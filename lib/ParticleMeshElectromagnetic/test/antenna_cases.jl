# WP-026: the laser antenna's runs and their analysis, shared by antenna.jl and
# scripts/em2d/antenna.jl. A run is the 2D module's step with a GaussianLaser in vacuum (or with
# a plasma), recording on the host the discrete-time Fourier transform at ω₀ = 1 of Ey and Ez on
# a few probe columns, and optionally snapshots of the fields.

using ParticleMeshElectromagnetic: Antenna, antenna_dy, yee_b_half!

"""
    antenna_problem(T; ncx = 16, ncy = 8, length = (300, 120), waist = 6, duration = 8, position = 10, focus = 28,
                    polarization = 0, chirp = 0, a0 = 1, tend = 160, species = (), courant = 95//100, phase = 0,
                    center = nothing) -> EMProblem{2}

A box of `length` (c/ω₀) with `ncx` cells per laser wavelength 2π along x and `ncy` along y,
Δt at `courant` of the Yee limit, `round(tend/Δt)` steps, and one [`GaussianLaser`](@ref) whose
envelope peaks at the antenna plane at t = 6·duration (below Float64 roundoff at t = 0), its axis
at the box's middle, or at `center` rows of Δy above y = 0 when given. Every laser parameter may
be a dual (A3).
"""
function antenna_problem(::Type{T}; ncx = 16, ncy = 8, length = (300, 120), waist = 6, duration = 8, position = 10,
                         focus = 28, polarization = 0, chirp = 0, a0 = 1, tend = 160, species = (), courant = 95 // 100,
                         phase = 0, center = nothing) where {T}
    h = (2π / ncx, 2π / ncy)
    n = (round(Int, length[1] / h[1]), round(Int, length[2] / h[2]))
    L = (T(n[1] * h[1]), T(n[2] * h[2]))
    dt = T(courant) / sqrt(1 / (L[1] / n[1])^2 + 1 / (L[2] / n[2])^2)
    laser = GaussianLaser(; a0 = T(a0), waist = T(waist), duration = T(duration), delay = 6 * T(duration), focus = T(focus),
                          position = Float64(position), center = isnothing(center) ? Float64(value_of(L[2])) / 2 : center * Float64(value_of(h[2])), chirp = T(chirp),
                          polarization = T(polarization), phase = T(phase))
    return EMProblem("antenna", L, n, dt, round(Int, tend / value_of(dt)), species, 1, (1, 0); lasers = (laser,))
end

"""
    antenna_run(prob; mode = Reference(), probes = (), snaps = (), be = CPU(), workgroupsize = nothing)
        -> (; st, prob, A, ip, shots)

Run `prob` through the module's step. `A[q]` holds the discrete-time Fourier transforms at
ω = 1, `Σ_n E(tₙ)·e^{i tₙ}·Δt` over the times tₙ = nΔt, n = 0…nsteps, of Ey and Ez on the probe
column `ip[q]` nearest to `probes[q]` (interior rows); `shots` the copies of the fields after the
steps listed in `snaps`, with their times.
"""
function antenna_run(prob; mode = Reference(), probes = (), snaps = (), be = CPU(), workgroupsize = nothing)
    st = EMState(prob, EMNumerics(; mode, workgroupsize, current = Esirkepov()); backend = be)
    initialize!(st, prob)
    p = params(prob)
    (; grid, fields) = st
    g, n, h = guards(grid), ncells(grid), cellsize(grid)
    T = eltype(st.ρ)
    dt = prob.dt
    ip = [round(Int, x / value_of(h[1])) + 1 for x in probes]
    rows = (g + 1):(g + n[2])
    A = [(Ey = zeros(Complex{T}, n[2]), Ez = zeros(Complex{T}, n[2])) for _ in ip]
    function accumulate!(k)
        w = cis(k * dt) * dt
        Ey, Ez = Array(fields.Ey), Array(fields.Ez)
        for (q, i) in enumerate(ip)
            A[q].Ey .+= w .* view(Ey, g + i, rows)
            A[q].Ez .+= w .* view(Ez, g + i, rows)
        end
    end
    accumulate!(0)
    shots = []
    for k in 1:prob.nsteps
        step!(st, p, k)
        isempty(ip) || accumulate!(k)
        k in snaps && push!(shots, (t = k * dt, f = map(Array, fields)))
    end
    return (; st, prob, A, ip, shots)
end

"The antenna of a run, as the state placed it."
run_antenna(r) = only(r.st.aux.antennas)

"""
    incident_dtft(r) -> (; Ey, Ez)

The discrete-time Fourier transform at ω = 1 of the incident field at the antenna plane (the
E nodes of column `ia`), over the same times as [`antenna_run`](@ref): Ey on the rows of Ey, Ez
on those of Ez, from [`laser_field`](@ref) in the run's element type.
"""
function incident_dtft(r)
    a = run_antenna(r)
    l = a.laser
    T = typeof(a.xa)
    n2 = Int(a.n2)
    dt = r.prob.dt
    sθ, cθ = sincos(l.polarization)
    Ey, Ez = zeros(Complex{T}, n2), zeros(Complex{T}, n2)
    for k in 0:r.prob.nsteps, j in 1:n2
        J = Int32(j) + a.g
        s = k * dt - l.delay
        w = cis(k * dt) * dt
        Ey[j] += w * laser_field(l, a.xa - l.focus, antenna_dy(a, J, one(T) / 2), s)[1] * cθ
        Ez[j] += w * laser_field(l, a.xa - l.focus, antenna_dy(a, J, zero(T)), s)[1] * sθ
    end
    return (; Ey, Ez)
end

"The wavenumbers of the DFT along y of `m` rows of height `dy`, in FFT order."
ky_modes(m, dy) = [2π * (q <= m ÷ 2 + 1 ? q - 1 : q - 1 - m) / (m * dy) for q in 1:m]
dft(v) = (m = length(v); [sum(v[j] * cis(-2π * (q - 1) * (j - 1) / m) for j in 1:m) for q in 1:m])
idft(V) = (m = length(V); [sum(V[q] * cis(2π * (q - 1) * (j - 1) / m) for q in 1:m) / m for j in 1:m])

"""
    yee_kx(ω, ky, dx, dy, dt) -> complex k_x

The wavenumber along x of the Yee mode of frequency ω and transverse wavenumber ky,
`sin²(k_xΔx/2)/Δx² = sin²(ωΔt/2)/Δt² − sin²(k_yΔy/2)/Δy²`, on the branch that travels or decays
along +x (Im k_x ≥ 0): the exact discrete relation of T-320.
"""
function yee_kx(ω, ky, dx, dy, dt)
    σ = (sin(ω * dt / 2) / dt)^2 - (sin(ky * dy / 2) / dy)^2
    kx = 2 / dx * asin(dx * sqrt(complex(σ)))
    return imag(kx) < 0 ? conj(kx) : kx
end

"""
    yee_axis(dx, dt; ω = 1) -> (; k, keff, vg)

The Yee numbers of ω along x on the axis (k_y = 0): the wavenumber `k`, `sin(kΔx/2)/Δx = sin(ωΔt/2)/Δt`;
`keff = sin(kΔx)/Δx`, with which the Yee relation reads `k_x ≈ k − k_y²/(2·keff)` for small k_y
(the paraxial beam of the scheme diffracts with keff in place of ω); and the group velocity
`vg = cos(kΔx/2)/cos(ωΔt/2)`.
"""
function yee_axis(dx, dt; ω = 1.0)
    k = 2 / dx * asin(dx / dt * sin(ω * dt / 2))
    return (; k, keff = sin(k * dx) / dx, vg = cos(k * dx / 2) / cos(ω * dt / 2))
end

"""
    angular_error(r; component = :Ey) -> Vector of (; x, err)

For every probe column at or past the antenna plane: the relative L² difference between the
measured ω = 1 transform of `component` and the Yee angular-spectrum propagation of the incident
field's transform from the plane, `Â(x, k_y) = Â_inc(x_a, k_y)·exp(i·k_x(1, k_y)·(x − x_a))`
([`yee_kx`](@ref), evanescent modes included), in Float64.
"""
function angular_error(r; component = :Ey)
    a = run_antenna(r)
    h = Float64.(value_of.(a.dx))
    dt = Float64(value_of(r.prob.dt))
    inc = map(v -> value_of.(real.(v)) .+ im .* value_of.(imag.(v)), incident_dtft(r))
    S = dft(getfield(inc, component))
    ky = ky_modes(length(S), h[2])
    xa = Float64(value_of(a.xa))
    out = []
    for (q, i) in enumerate(r.ip)
        x = (i - 1) * h[1]
        x < xa - h[1] / 2 && continue
        pred = idft(S .* [cis(yee_kx(1.0, k, h[1], h[2], dt) * (x - xa)) for k in ky])
        meas = value_of.(real.(getfield(r.A[q], component))) .+ im .* value_of.(imag.(getfield(r.A[q], component)))
        push!(out, (; x, err = sqrt(sum(abs2, meas .- pred) / sum(abs2, pred))))
    end
    return out
end

"""
    beam_profile(r; component = :Ey) -> Vector of (; x, w, y, phase)

On each probe column the centroid `y = ȳ` (from the laser's axis), and the 1/e field radius
`w = 2·√⟨(y − ȳ)²⟩`, of the intensity |Â(x, y)|² of the ω = 1 transform, and its phase on the
row nearest the axis with the carrier `k·(x − x_a)` of [`yee_axis`](@ref) removed. The row
coordinates are those of the staggering, independent of the antenna's code: Ey rows at
(j − ½)Δy, Ez rows at (j − 1)Δy. The axis must lie in the middle of the box (no wrap).
"""
function beam_profile(r; component = :Ey)
    a = run_antenna(r)
    h = Float64.(value_of.(a.dx))
    dt = Float64(value_of(r.prob.dt))
    yk = yee_axis(h[1], dt)
    xa = Float64(value_of(a.xa))
    s = component === :Ey ? 0.5 : 0.0
    m = Int(a.n2)
    y = ((1:m) .- 1 .+ s) .* h[2] .- Float64(a.laser.center)
    out = []
    for (q, i) in enumerate(r.ip)
        x = (i - 1) * h[1]
        A = value_of.(real.(getfield(r.A[q], component))) .+ im .* value_of.(imag.(getfield(r.A[q], component)))
        I = abs2.(A)
        ȳ = sum(y .* I) / sum(I)
        w = 2 * sqrt(sum((y .- ȳ) .^ 2 .* I) / sum(I))
        j0 = argmin(abs.(y))
        push!(out, (; x, w, y = ȳ, phase = angle(A[j0] * cis(-yk.k * (x - xa)))))
    end
    return out
end

"""
    angular_profile(r; component = :Ey) -> Vector of (; x, w, phase)

What [`beam_profile`](@ref) measures, computed instead from the Yee angular-spectrum propagation
of the incident field's ω = 1 transform ([`angular_error`](@ref)'s prediction): the beam a
perfect antenna launches into the scheme, non-paraxial terms and the scheme's dispersion
included.
"""
function angular_profile(r; component = :Ey)
    a = run_antenna(r)
    h = Float64.(value_of.(a.dx))
    dt = Float64(value_of(r.prob.dt))
    yk = yee_axis(h[1], dt)
    xa = Float64(value_of(a.xa))
    inc = map(v -> value_of.(real.(v)) .+ im .* value_of.(imag.(v)), incident_dtft(r))
    S = dft(getfield(inc, component))
    ky = ky_modes(length(S), h[2])
    s = component === :Ey ? 0.5 : 0.0
    y = ((1:Int(a.n2)) .- 1 .+ s) .* h[2] .- Float64(a.laser.center)
    j0 = argmin(abs.(y))
    out = []
    for i in r.ip
        x = (i - 1) * h[1]
        A = idft(S .* [cis(yee_kx(1.0, k, h[1], h[2], dt) * (x - xa)) for k in ky])
        I = abs2.(A)
        ȳ = sum(y .* I) / sum(I)
        push!(out, (; x, w = 2 * sqrt(sum((y .- ȳ) .^ 2 .* I) / sum(I)), phase = angle(A[j0] * cis(-yk.k * (x - xa)))))
    end
    return out
end

"""
    paraxial_beam(r, x) -> (; w, gouy)

The paraxial Gaussian beam of the Yee scheme at x: the antenna launches the profile of a beam
of complex parameter q_a = (x_a − x_f) − i·z_R (z_R = waist²/2, the continuum's at ω₀ = 1); the
scheme diffracts it with `keff` ([`yee_axis`](@ref)), so q(x) = keff·q_a + (x − x_a): the focus
moves to x_a + keff·(x_f − x_a), the Rayleigh length to keff·z_R, the waist stays. Returns the
1/e field radius `w = waist·|q(x)|/(keff·z_R)` and the 2D Gouy phase `arg (q(x_a)/q(x))^(1/2)`
accumulated since the plane.
"""
function paraxial_beam(r, x)
    a = run_antenna(r)
    l = a.laser
    h1 = Float64(value_of(a.dx[1]))
    yk = yee_axis(h1, Float64(value_of(r.prob.dt)))
    xa, xf, w0 = Float64(value_of(a.xa)), Float64(value_of(l.focus)), Float64(value_of(l.waist))
    zR = w0^2 / 2
    qa = yk.keff * complex(xa - xf, -zR)
    q = qa + (x - xa)
    return (; w = w0 * abs(q) / (yk.keff * zR), gouy = angle(sqrt(qa / q)))
end

"The six field components of a field `NamedTuple` (E, then B)."
eb(f) = (f.Ex, f.Ey, f.Ez, f.Bx, f.By, f.Bz)

"""
    conserved_energy(f, grid, dt; cols) -> W

The energy the Yee scheme keeps in vacuum (T-323), ½·Σ(Eⁿ)² + ½·Σ Bⁿ⁻¹ᐟ²·Bⁿ⁺¹ᐟ² times ΔxΔy, over
the interior columns `cols` (E and B of the same index) of the fields `f` at an integer time (a
state's fields or a snapshot); Bⁿ±¹ᐟ² from Bⁿ by half steps of ∓Δt/2 with the curl of Eⁿ
([`yee_b_half!`](@ref) on copies), so `cols` must not hold the B column behind an antenna plane
while the antenna still injects.
"""
function conserved_energy(f, grid, dt; cols)
    g, n, h = guards(grid), ncells(grid), cellsize(grid)
    fp, fm = map(copy, f), map(copy, f)
    yee_b_half!(fp, grid, dt / 2)
    yee_b_half!(fm, grid, -dt / 2)
    I1, I2 = g .+ cols, (g + 1):(g + n[2])
    W = zero(eltype(f.Ex))
    for c in (f.Ex, f.Ey, f.Ez)
        W += sum(abs2, Array(c)[I1, I2])
    end
    for k in (:Bx, :By, :Bz)
        W += sum(Array(getfield(fp, k))[I1, I2] .* Array(getfield(fm, k))[I1, I2])
    end
    return W * h[1] * h[2] / 2
end

"""
    forward_window(r, xmax) -> (fwd, back)

The interior columns from the antenna plane up to x = `xmax`, where the forward pulse is, and
the others (behind the plane, and past `xmax`), where the backward leak is once it has wrapped
round the periodic box. A run separates the two when, at the time of the measurement t, the
pulse lies within `xmax` and the wrapped leak, emitted from the plane at most a time t earlier
and travelling at c, lies beyond it: L − t > xmax − x_a.
"""
function forward_window(r, xmax)
    a = run_antenna(r)
    n1 = ncells(r.st.grid)[1]
    i2 = round(Int, xmax / value_of(a.dx[1])) + 1
    fwd = Int(a.ia):i2
    return fwd, setdiff(1:n1, fwd)
end

"The energy ½Σ(E² + B²)ΔxΔy of the integer-time fields `f` (host arrays) over the interior columns `cols`."
function plain_energy(f, grid; cols)
    g, n, h = guards(grid), ncells(grid), cellsize(grid)
    I1, I2 = g .+ cols, (g + 1):(g + n[2])
    return sum(c -> sum(abs2, c[I1, I2]), eb(f)) * h[1] * h[2] / 2
end

"""
    incident_flux(r) -> Φ

The discrete Poynting flux of the antenna's incident field through its plane over the run, the
flux of the Yee scheme's energy through a plane (`Δt·Δy·Σ ½(Eⁿ + Eⁿ⁺¹)·Bⁿ⁺¹ᐟ²`, E on the plane and
B half a cell behind it, Ey·Bz − Ez·By) with the incident values the antenna injects: E at x_a and
times tₙ, B at x_a − Δx/2 and t_{n+1/2} + ht.
"""
function incident_flux(r)
    a = run_antenna(r)
    l = a.laser
    T = typeof(a.xa)
    dt = r.prob.dt
    sθ, cθ = sincos(l.polarization)
    ξe, ξb = a.xa - l.focus, a.xa - a.hx - l.focus
    Φ = zero(T)
    for k in 1:r.prob.nsteps, j in 1:Int(a.n2)
        J = Int32(j) + a.g
        t = (k - 1) * dt
        yc, yn = antenna_dy(a, J, one(T) / 2), antenna_dy(a, J, zero(T))
        e0c, e1c = laser_field(l, ξe, yc, t - l.delay)[1], laser_field(l, ξe, yc, t + dt - l.delay)[1]
        e0n, e1n = laser_field(l, ξe, yn, t - l.delay)[1], laser_field(l, ξe, yn, t + dt - l.delay)[1]
        bp = laser_field(l, ξb, yc, t + dt / 2 + a.ht - l.delay)[2]
        bs = laser_field(l, ξb, yn, t + dt / 2 + a.ht - l.delay)[3]
        Φ += (e0c + e1c) / 2 * bp * cθ^2 + (e0n + e1n) / 2 * bs * sθ^2
    end
    return Φ * dt * a.dx[2]
end

"""
    energy_centroid(f, grid; cols = all) -> X

The centroid along x of the energy density of the integer-time fields `f` (host arrays) over the
interior columns `cols`, each component at its own stagger (`Node()` columns at (i − 1)Δx,
`Center()` at (i − ½)Δx).
"""
function energy_centroid(f, grid; cols = 1:ncells(grid)[1])
    g, n, h = guards(grid), ncells(grid), cellsize(grid)
    I2 = (g + 1):(g + n[2])
    L = field_layout(Val(2))
    num, den = 0.0, 0.0
    for key in (:Ex, :Ey, :Ez, :Bx, :By, :Bz)
        c = getfield(f, key)
        s = getfield(L, key)[1] isa Center ? 0.5 : 0.0
        for i in cols
            e = Float64(value_of(sum(abs2, view(c, g + i, I2))))
            num += (i - 1 + s) * Float64(value_of(h[1])) * e
            den += e
        end
    end
    return num / den
end

"""
    spectral_group_velocity(f, grid, dt; cols = all) -> v

The group velocity along x of the Yee scheme averaged over the modes of the fields `f` with
their energy as the weight: the 2D DFT of the interior of each component (zero outside the
columns `cols`, which hold the pulse), the weight of a mode
(k_x, k_y) the sum of its squared moduli, its frequency from the exact discrete relation and
`v_x = |sin(k_xΔx)|/Δx / (sin(ωΔt)/Δt)`, the x derivative of the relation (a pulse travelling
along +x, each pair of conjugate modes one wave). The energy centroid of a pulse that has left its
source moves at this velocity (T-404).
"""
function spectral_group_velocity(f, grid, dt; cols = 1:ncells(grid)[1])
    g, n = guards(grid), ncells(grid)
    h = Float64.(value_of.(cellsize(grid)))
    dt = Float64(value_of(dt))
    I1, I2 = (g + 1):(g + n[1]), (g + 1):(g + n[2])
    Fx = [cis(-2π * (p - 1) * (i - 1) / n[1]) for p in 1:n[1], i in 1:n[1]]
    Fy = [cis(-2π * (j - 1) * (q - 1) / n[2]) for j in 1:n[2], q in 1:n[2]]
    W = zeros(n[1], n[2])
    mask = [i in cols for i in 1:n[1]]
    for c in eb(f)
        W .+= abs2.(Fx * (mask .* Float64.(value_of.(c[I1, I2]))) * Fy)
    end
    kx, ky = ky_modes(n[1], h[1]), ky_modes(n[2], h[2])
    num, den = 0.0, 0.0
    for p in 1:n[1], q in 1:n[2]
        sw = sqrt((sin(kx[p] * h[1] / 2) / h[1])^2 + (sin(ky[q] * h[2] / 2) / h[2])^2)
        sw == 0 && continue
        ω = 2 / dt * asin(dt * sw)
        num += W[p, q] * abs(sin(kx[p] * h[1])) / h[1] / (sin(ω * dt) / dt)
        den += W[p, q]
    end
    return num / den
end
