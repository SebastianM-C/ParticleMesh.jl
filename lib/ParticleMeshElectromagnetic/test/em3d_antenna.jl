# WP-073: the laser antenna at N = 3, T-1020 to T-1028 (T-1029 is their rerun on CUDA, in
# gpu.jl). The antenna is the 2D total-field/scattered-field plane of WP-026 across a 3D box: the
# same four corrections on the planes ia − 1 (B) and ia (E), every component of the incident
# wave at its own place across the plane, the incident field that of the round paraxial Gaussian
# beam with its B to order k_⊥² (`laser_field` with dy and dz). After antenna.jl (its helpers
# `yee_kx`, `ky_modes`, `forward_window`) and em3d_open.jl (`alive_3d`).

using ForwardDiff: ForwardDiff
using ParticleMeshElectromagnetic: Antenna, AntennaPlane, laser_field, antenna_d, yee_b_half!

"The antenna of a 3D run: the problem's laser on the plane where the state placed it."
run_antenna_3d(r) = Antenna(only(params(r.prob).lasers), only(antenna_planes(r.st)))

# The places of Ey (y Center, z Node) and Ez (y Node, z Center) on the plane, as the stagger offsets (y, z).
const EY_AT, EZ_AT = (1 // 2, 0 // 1), (0 // 1, 1 // 2)

"The transverse distances (dy, dz) from the laser's axis of row j, layer k (interior indices) at the offsets `at`."
function plane_d(a, j, k, at)
    T = typeof(a.xa)
    c = a.laser.center
    return antenna_d(a, 2, c[1], Int32(a.g + j), T(at[1])), antenna_d(a, 3, c[2], Int32(a.g + k), T(at[2]))
end

"""
    incident_dtft_3d(r) -> (; Ey, Ez)

The discrete-time Fourier transform at ω = 1 of the incident Ey and Ez at the antenna plane, each
at its own places, over the times of [`antenna3d_run`](@ref), from [`laser_field`](@ref).
"""
function incident_dtft_3d(r)
    a = run_antenna_3d(r)
    l = a.laser
    T = typeof(a.xa)
    n2, n3 = Int(a.n[2]), Int(a.n[3])
    dt = r.prob.dt
    sθ, cθ = sincos(l.polarization)
    Ey, Ez = zeros(Complex{T}, n2, n3), zeros(Complex{T}, n2, n3)
    for j in 1:n2, k in 1:n3
        dy, dz = plane_d(a, j, k, EY_AT)
        ey, ez = plane_d(a, j, k, EZ_AT)
        for m in 0:r.prob.nsteps
            s = m * dt - l.delay
            w = cis(m * dt) * dt
            Ey[j, k] += w * laser_field(l, a.xa - l.focus, dy, dz, s)[1] * cθ
            Ez[j, k] += w * laser_field(l, a.xa - l.focus, ey, ez, s)[1] * sθ
        end
    end
    return (; Ey, Ez)
end

"The DFT matrix of `m` points and its inverse's, for the transverse transforms of the probe planes."
dft_matrix(m) = [cis(-2π * (q - 1) * (j - 1) / m) for q in 1:m, j in 1:m]
dft2(A) = dft_matrix(size(A, 1)) * A * transpose(dft_matrix(size(A, 2)))
idft2(S) = conj.(dft_matrix(size(S, 1))) * S * transpose(conj.(dft_matrix(size(S, 2)))) / length(S)

"The Yee wavenumber along x of the mode (ω, k_y, k_z) on the branch that travels or decays along +x (T-320 in 3D)."
function yee_kx_3d(ω, ky, kz, dx, dy, dz, dt)
    σ = (sin(ω * dt / 2) / dt)^2 - (sin(ky * dy / 2) / dy)^2 - (sin(kz * dz / 2) / dz)^2
    kx = 2 / dx * asin(dx * sqrt(complex(σ)))
    return imag(kx) < 0 ? conj(kx) : kx
end

host(v) = value_of.(real.(v)) .+ im .* value_of.(imag.(v))

"""
    angular_error_3d(r; component = :Ey, against = component) -> Vector of (; x, err)

For every probe plane past the antenna: the L² norm of the measured ω = 1 transform of
`component` minus the Yee angular-spectrum propagation of the incident transform of `component`
from the plane, relative to the propagated incident transform of `against` (`yee_kx_3d`,
evanescent modes included), in Float64. With `against ≠ component` it measures a component the
antenna should not launch.
"""
function angular_error_3d(r; component = :Ey, against = component)
    a = run_antenna_3d(r)
    h = Float64.(value_of.(a.dx))
    dt = Float64(value_of(r.prob.dt))
    inc = map(host, incident_dtft_3d(r))
    S, Sref = dft2(getfield(inc, component)), dft2(getfield(inc, against))
    ky, kz = ky_modes(size(S, 1), h[2]), ky_modes(size(S, 2), h[3])
    xa = Float64(value_of(a.xa))
    out = []
    for (q, i) in enumerate(r.ip)
        x = (i - 1) * h[1]
        x < xa - h[1] / 2 && continue
        P = [cis(yee_kx_3d(1.0, ky[p], kz[s], h[1], h[2], h[3], dt) * (x - xa)) for p in eachindex(ky), s in eachindex(kz)]
        pred, ref = idft2(S .* P), idft2(Sref .* P)
        meas = host(getfield(r.A[q], component))
        push!(out, (; x, err = sqrt(sum(abs2, meas .- pred) / sum(abs2, ref))))
    end
    return out
end

"""
    conserved_energy_3d(f, grid, dt; cols) -> W

`conserved_energy` in 3D: ½·Σ(Eⁿ)² + ½·Σ Bⁿ⁻¹ᐟ²·Bⁿ⁺¹ᐟ² times ΔxΔyΔz over the interior columns `cols`.
"""
function conserved_energy_3d(f, grid, dt; cols)
    g, n, h = guards(grid), ncells(grid), cellsize(grid)
    fp, fm = map(copy, f), map(copy, f)
    yee_b_half!(fp, grid, dt / 2)
    yee_b_half!(fm, grid, -dt / 2)
    I = (g .+ cols, (g + 1):(g + n[2]), (g + 1):(g + n[3]))
    W = sum(c -> sum(abs2, Array(c)[I...]), (f.Ex, f.Ey, f.Ez))
    W += sum(k -> sum(Array(getfield(fp, k))[I...] .* Array(getfield(fm, k))[I...]), (:Bx, :By, :Bz))
    return W * prod(h) / 2
end

"The energy ½Σ(E² + B²)ΔxΔyΔz of the integer-time host fields `f` over the interior columns `cols`."
function plain_energy_3d(f, grid; cols)
    g, n, h = guards(grid), ncells(grid), cellsize(grid)
    I = (g .+ cols, (g + 1):(g + n[2]), (g + 1):(g + n[3]))
    return sum(c -> sum(abs2, c[I...]), (f.Ex, f.Ey, f.Ez, f.Bx, f.By, f.Bz)) * prod(h) / 2
end

"The interior columns from the plane to x = `xmax` and the others (`forward_window` at N = 3)."
function forward_window_3d(r, xmax)
    a = run_antenna_3d(r)
    i2 = round(Int, xmax / value_of(a.dx[1])) + 1
    fwd = Int(a.ia):i2
    return fwd, setdiff(1:ncells(r.st.grid)[1], fwd)
end

"""
    incident_flux_3d(r) -> Φ

`incident_flux` in 3D: the discrete Poynting flux `Δt·Δy·Δz·Σ (½(Eyⁿ + Eyⁿ⁺¹)·Bzⁿ⁺¹ᐟ² − ½(Ezⁿ + Ezⁿ⁺¹)·Byⁿ⁺¹ᐟ²)`
of the incident values the antenna injects, each at its own place.
"""
function incident_flux_3d(r)
    a = run_antenna_3d(r)
    l = a.laser
    T = typeof(a.xa)
    dt = r.prob.dt
    sθ, cθ = sincos(l.polarization)
    ξe, ξb = a.xa - l.focus, a.xa - a.hx - l.focus
    Φ = zero(T)
    for j in 1:Int(a.n[2]), k in 1:Int(a.n[3])
        y1, z1 = plane_d(a, j, k, EY_AT)
        y2, z2 = plane_d(a, j, k, EZ_AT)
        for m in 1:r.prob.nsteps
            t = (m - 1) * dt
            ey = (laser_field(l, ξe, y1, z1, t - l.delay)[1] + laser_field(l, ξe, y1, z1, t + dt - l.delay)[1]) / 2 * cθ
            ez = (laser_field(l, ξe, y2, z2, t - l.delay)[1] + laser_field(l, ξe, y2, z2, t + dt - l.delay)[1]) / 2 * sθ
            _, bp, _, xy = laser_field(l, ξb, y1, z1, t + dt / 2 + a.ht - l.delay)
            _, _, bs, xz = laser_field(l, ξb, y2, z2, t + dt / 2 + a.ht - l.delay)
            Φ += ey * (bp * cθ - xy * sθ) - ez * (xz * cθ - bs * sθ)
        end
    end
    return Φ * dt * a.dx[2] * a.dx[3]
end

# --- T-1020 ---------------------------------------------------------------------------------

# The incident field is the round paraxial Gaussian beam in three dimensions: with U = e(phase =
# 0) + i·e(phase = −π/2) at s = 0, 2i·∂U/∂ξ + ∂²U/∂y² + ∂²U/∂z² = 0, |U| = a0·(1 + ξ²/z_R²)^(−1/2)·
# exp(−r²/w²) and arg U on the axis −atan(ξ/z_R) (the 3D Gouy phase, twice the 2D one). Its B
# amplitudes are e ∓ ½∂²_y e ± ½∂²_z e and ∂_y∂_z e, by ForwardDiff through laser_field, for
# AxisTime and WavefrontTime; WavefrontTime is AxisTime on the axis; and the flux
# ∫∫∫ (Ey·Bz − Ez·By) dy dz dt of a host quadrature matches laser_energy, π^{3/2}a0²w²τ/(4√2), for
# three polarizations and both envelopes.
@testset "T-1020 the 3D antenna's incident field is the round paraxial Gaussian beam with its B, and laser_energy its flux [A2, A3, WP-073]" begin
    mk(; kw...) = GaussianLaser(; merge((; a0 = 0.7, waist = 5.0, duration = 6.0, delay = 30.0, focus = 3.0, position = 0.0,
                                         center = (0.0, 0.0), chirp = 0.004, polarization = 0.3, phase = 0.2), kw)...)
    l = mk()
    l0 = mk(; phase = 0.0)
    lq = mk(; phase = -π / 2)
    zR = l.waist^2 / 2
    U(ξ, y, z) = complex(laser_field(l0, ξ, y, z, 0.0)[1], laser_field(lq, ξ, y, z, 0.0)[1])
    worst_pde = worst_mod = worst_b = 0.0
    for ξ in (-2zR, -0.7zR, 0.0, 0.4zR, 1.5zR), (y, z) in ((0.0, 0.0), (1.3, -0.4), (-4.0, 2.5), (7.5, 0.0))
        dξ = ForwardDiff.derivative(x -> real(U(x, y, z)), ξ) + im * ForwardDiff.derivative(x -> imag(U(x, y, z)), ξ)
        lap(f) = ForwardDiff.derivative(a -> ForwardDiff.derivative(b -> f(ξ, b, z), a), y) +
                 ForwardDiff.derivative(a -> ForwardDiff.derivative(b -> f(ξ, y, b), a), z)
        d2 = lap((x, a, b) -> real(U(x, a, b))) + im * lap((x, a, b) -> imag(U(x, a, b)))
        u = U(ξ, y, z)
        worst_pde = max(worst_pde, abs(2im * dξ + d2) / (abs(u) / l.waist^2))
        w = l.waist * sqrt(1 + (ξ / zR)^2)
        worst_mod = max(worst_mod, abs(abs(u) - l.a0 / sqrt(1 + (ξ / zR)^2) * exp(-(y^2 + z^2) / w^2)) / l.a0)
        y == z == 0 && (worst_mod = max(worst_mod, abs(angle(u) + atan(ξ / zR))))
        for env in (AxisTime(), WavefrontTime()), s in (-7.0, -0.3, 0.0, 2.2)
            le = mk(; envelope = env)
            e, bp, bs, byz = laser_field(le, ξ, y, z, s)
            E(a, b) = laser_field(le, ξ, a, b, s)[1]
            eyy = ForwardDiff.derivative(a -> ForwardDiff.derivative(b -> E(b, z), a), y)
            ezz = ForwardDiff.derivative(a -> ForwardDiff.derivative(b -> E(y, b), a), z)
            eyz = ForwardDiff.derivative(a -> ForwardDiff.derivative(b -> E(b, a), y), z)
            worst_b = max(worst_b, abs(bp - (e - eyy / 2 + ezz / 2)) / l.a0, abs(bs - (e + eyy / 2 - ezz / 2)) / l.a0,
                          abs(byz - eyz) / l.a0)
            env isa WavefrontTime && y == z == 0 && (worst_b = max(worst_b, abs(e - laser_field(l, ξ, 0.0, 0.0, s)[1]) / l.a0))
        end
    end
    @info "T-1020" worst_pde worst_mod worst_b
    @test worst_pde <= 100 * eps()
    @test worst_mod <= 100 * eps()
    @test worst_b <= 100 * eps()
    for s in (-5.0, 0.0, 1.7)
        @test laser_field(l, 0.0, 0.0, 0.0, s)[1] ≈ l.a0 * exp(-(s / l.duration)^2) * cos(s + l.chirp * s^2 - l.phase) rtol = 1e-14
    end
    # the flux through the plane at ξ = −z_R (duration 8: the exp(−duration²/2) term below roundoff)
    # (steps of 0.6 across, far below the beam radius 7.1 there; ±30 leaves exp(−36) of the power out)
    h, ht = 0.6, 0.25
    for env in (AxisTime(), WavefrontTime()), θ in (0.0, π / 2, 0.6)
        lθ = mk(; duration = 8.0, chirp = 0.0, polarization = θ, envelope = env)
        sθ, cθ = sincos(θ)
        Φ = 0.0
        for y in -30:h:30, z in -30:h:30, s in -56:ht:56
            e, bp, bs, byz = laser_field(lθ, -zR, y, z, s)
            Φ += e * cθ * (bp * cθ - byz * sθ) - e * sθ * (byz * cθ - bs * sθ)
        end
        Φ *= h * h * ht
        @info "T-1020 flux" env θ rel = Φ / laser_energy(lθ) - 1
        @test Φ / laser_energy(lθ) - 1 ≈ 0 atol = 1e-11                # the sum of 4.6·10⁶ terms: 1.3·10⁻¹² measured
    end
    @test laser_energy(mk(; duration = 8.0)) ≈ π^(3 / 2) * 0.7^2 * 25 * 8 / (4 * sqrt(2)) rtol = 1e-14
end

# --- T-1021 ---------------------------------------------------------------------------------

# One way, as T-401: behind the plane only what the incident field fails to satisfy of the
# discrete equations leaves, a measured fraction of the launched energy (the box is long enough
# that the leak, wrapped round the periodic x, and the pulse lie in separate columns at the end);
# in the A8 matrix with a0, focus and chirp seeded as the dual's partials, the fields linear in
# a0 (a0·∂E/∂a0 = E to roundoff) and the focus and chirp partials nonzero; with no antenna the
# 3D solve with the empty AntennaPlanes is the plain solve bit for bit.
@testset "T-1021 the 3D antenna launches one way, is linear in a0, and without antennas the step is the plain step bit for bit [A2, A3, A8, WP-073]" begin
    C, Cl = emC("T-1021"), emC("T-1021", "C_linear")
    for T in TYPES, θ in (0, 3 // 5)
        D = T === D3
        a0 = D ? mkdual(1, 1, 0, 0) : one(T)
        focus = D ? mkdual(14, 0, 1, 0) : T(14)
        chirp = D ? mkdual(1 // 100, 0, 0, 1) : T(1 // 100)
        # at t = 60 the pulse lies within x ≤ 55, the wrapped leak beyond x = 6 + 100 − 60 = 46 only after t = 60 − 15
        prob = antenna3d_problem(T; length = (100, 24, 24), position = 6, focus, chirp, a0, polarization = θ, tend = 60)
        r = antenna3d_run(prob)
        fwd, back = forward_window_3d(r, 58)
        f = map(Array, r.st.fields)
        frac = Float64(value_of(plain_energy_3d(f, r.st.grid; cols = back) / conserved_energy_3d(f, r.st.grid, prob.dt; cols = fwd)))
        @info "T-1021" T θ frac
        record!("T-1021 θ=$θ", T, frac)
        @test frac <= C
        if D
            worst = 0.0
            for c in (f.Ex, f.Ey, f.Ez, f.Bx, f.By, f.Bz)
                m = maximum(x -> abs(value_of(x)), c)
                d = maximum(x -> abs(ForwardDiff.partials(x)[1] - value_of(x)), c)
                worst = max(worst, m == 0 ? (d == 0 ? 0.0 : Inf) : d / (eps(Float32) * m * sqrt(prob.nsteps)))
            end
            @info "T-1021 linear" θ worst
            record!("T-1021 linear", T, worst)
            @test worst <= Cl
            @test any(x -> !iszero(ForwardDiff.partials(x)[2]), f.Ey) && any(x -> !iszero(ForwardDiff.partials(x)[3]), f.Ey)
        end
    end
    for T in TYPES
        grid = PeriodicGrid((10, 8, 7), (T(1 // 2), T(2 // 3), T(3 // 5)), 3)
        f1 = em_fields(CPU(), T, grid)
        rng = Xoshiro(73)
        for c in values(f1)
            interior_view(c, grid) .= T.(rand(rng, size(interior_view(c, grid))...) .- 1 // 2)
            fill_guards!(c, grid)
        end
        f2 = map(copy, f1)
        dt = T(1 // 4)
        for k in 1:5
            solve_fields!(Yee(), f1, grid, dt)
            solve_fields!(Yee(), f2, grid, dt, (AntennaPlanes((), nothing),), (; lasers = ()), k)
        end
        @test all(same_bits(getfield(f1, k), getfield(f2, k)) for k in keys(f1))
    end
end

# --- T-1022 ---------------------------------------------------------------------------------

# The ω₀ = 1 component of the launched beam on probe planes from the antenna to past the focus is
# the Yee angular-spectrum propagation of the incident field's ω₀ component at the plane, in two
# transverse wavenumbers (`yee_kx_3d`, evanescent modes included): the antenna launches a free
# beam of the scheme. The one approximation at ω₀ is the antenna's B, the expansion of the plane-
# wave ratios to order k_⊥²; its next terms are of order k_⊥⁴, so the error scales as 1/w0⁴ (the
# scale of the test, as T-402). The component the incident field does not have (Ez for θ = 0, Ey
# for θ = π/2) stays at that order too: a wrong cross term ∂_y∂_z u of the B would launch it at
# order k_⊥².
@testset "T-1022 the launched 3D beam is the Yee angular-spectrum propagation of the incident field, to the antenna's order k_⊥⁴ [A2, A7, WP-073]" begin
    C, Cx = emC("T-1022"), emC("T-1022", "C_cross")
    w0 = 4
    # the third run on a cross-section whose y and z differ in cells and cell size, the axis off the
    # diagonal, so that an exchange of y and z anywhere shows
    for (θ, main, cross, box) in ((0.0, :Ey, :Ez, (; length = (140, 30, 30))), (π / 2, :Ez, :Ey, (; length = (140, 30, 30))),
                                  (0.0, :Ey, :Ez, (; length = (140, 30, 26), nct = (6, 5), center = (16.5, 11.0))))
        # at t = 130 the pulse lies within 64 ≤ x ≤ 112, past the probes and not yet wrapped round the
        # periodic x; the backward leak reaches the probes after t = 140
        prob = antenna3d_problem(Float64; waist = w0, duration = 8, position = 6, focus = 16, polarization = θ, tend = 130, box...)
        r = antenna3d_run(prob; probes = (6, 12, 20, 30))
        e = angular_error_3d(r; component = main)
        x = angular_error_3d(r; component = cross, against = main)
        worst, worst_x = maximum(p -> p.err, e) * w0^4, maximum(p -> p.err, x) * w0^4
        @info "T-1022" θ box errors = join([round(p.err; sigdigits = 3) for p in e], " ") cross = join([round(p.err; sigdigits = 3) for p in x], " ")
        tag = haskey(box, :nct) ? " y≠z" : ""
        record!("T-1022 θ=$(round(θ; digits = 2))$tag", Float64, worst)
        record!("T-1022 cross θ=$(round(θ; digits = 2))$tag", Float64, worst_x)
        @test worst <= C
        @test worst_x <= Cx
    end
end

# --- T-1023 ---------------------------------------------------------------------------------

# The energy of the launched pulse, measured once it is past the plane (the scheme's conserved
# energy over the forward columns), is the discrete Poynting flux of the incident field through
# the plane to the antenna's order k_⊥⁴, and that flux is laser_energy's to the discretization's
# order Δ² (T-405 in 3D), for both envelopes and two polarizations.
@testset "T-1023 the energy the 3D pulse carries matches the Poynting flux through the antenna and laser_energy [A2, A7, WP-073]" begin
    C, Cf = emC("T-1023"), emC("T-1023", "C_flux")
    boxes = ((; length = (100, 24, 24)), (; length = (100, 24, 20), nct = (6, 5), center = (13.0, 9.0)))
    for env in (AxisTime(), WavefrontTime()), θ in (0.0, 0.6), box in boxes
        env isa WavefrontTime && haskey(box, :nct) && continue
        prob = antenna3d_problem(Float64; position = 6, polarization = θ, envelope = env, tend = 60, box...)
        r = antenna3d_run(prob)
        a = run_antenna_3d(r)
        fwd, _ = forward_window_3d(r, 58)
        W = conserved_energy_3d(map(Array, r.st.fields), r.st.grid, prob.dt; cols = fwd)
        Φ = incident_flux_3d(r)
        rel = abs(W / Φ - 1) * 4^4
        relf = abs(Φ / laser_energy(a.laser) - 1) / (prob.dt^2 + a.dx[1]^2)
        @info "T-1023" env θ box rel relf
        tag = haskey(box, :nct) ? " y≠z" : ""
        record!("T-1023 $(nameof(typeof(env))) θ=$θ$tag", Float64, rel)
        record!("T-1023 flux $(nameof(typeof(env))) θ=$θ$tag", Float64, relf)
        @test rel <= C
        @test relf <= Cf
    end
end

# --- T-1024 ---------------------------------------------------------------------------------

"""
    antenna_divergences_3d(st, p, nk, dt) -> (; gauss, gauss_col, divb, divb_col, scale, bscale)

`antenna_divergences` in 3D: the Gauss residual at every interior node and ∇·B at every cell
centre, the maximum off the antenna's plane (E plane `ia`, B cell plane `ia − 1`) and on it after
subtracting the surface charges the corrections have accumulated over `nk` steps: per step
`(Δt/Δx)·(D⁻_y Bz_inc − D⁻_z By_inc)` on the E plane, and `(Δt/2Δx)·(D⁺_z Ey_inc − D⁺_y Ez_inc)` on the
B plane for the incident E at both ends of the step.
"""
# The antenna's cross term is the mean of the two images, zero, on a row at the antipode of the axis.
odd_mean(a, d) = any(q -> abs(2 * abs(d[q - 1]) - a.len[q]) * 1000 < a.dx[q], (2, 3)) ? 0 : 1

function antenna_divergences_3d(st, p, nk, dt)
    (; grid, numerics) = st
    T = eltype(st.ρ)
    a = Antenna(only(p.lasers), only(antenna_planes(st)))
    l = a.laser
    ρd = field(CPU(), T, grid)
    for (s, ws) in zip(st.species, st.ws)
        deposit!(numerics.mode, ρd, s, numerics.shape, (Node(), Node(), Node()), grid, ws)
    end
    fold_guards!(ρd, grid)
    g, n, h = guards(grid), ncells(grid), cellsize(grid)
    I = CartesianIndices(ntuple(d -> (g + 1):(g + n[d]), 3))
    e = ntuple(d -> CartesianIndex(ntuple(b -> Int(b == d), 3)), 3)
    ρ = Array(ρd)
    E = map(Array, (st.fields.Ex, st.fields.Ey, st.fields.Ez))
    B = map(Array, (st.fields.Bx, st.fields.By, st.fields.Bz))
    ρ̄ = sum(ρ[I]) / prod(n)
    sθ, cθ = sincos(l.polarization)
    n2, n3 = n[2], n[3]
    qe, qb = zeros(T, n2, n3), zeros(T, n2, n3)
    ξe, ξb = a.xa - l.focus, a.xa - a.hx - l.focus
    for m in 1:nk
        t = (m - 1) * dt
        bz = [(d = plane_d(a, j, k, EY_AT); f = laser_field(l, ξb, d..., t + dt / 2 + a.ht - l.delay); f[2] * cθ - f[4] * odd_mean(a, d) * sθ)
              for j in 1:n2, k in 1:n3]
        by = [(d = plane_d(a, j, k, EZ_AT); f = laser_field(l, ξb, d..., t + dt / 2 + a.ht - l.delay); f[4] * odd_mean(a, d) * cθ - f[3] * sθ)
              for j in 1:n2, k in 1:n3]
        for j in 1:n2, k in 1:n3
            qe[j, k] += dt / a.dx[1] * ((bz[j, k] - bz[mod1(j - 1, n2), k]) / a.dx[2] - (by[j, k] - by[j, mod1(k - 1, n3)]) / a.dx[3])
        end
        for tt in (t, t + dt)
            ey = [laser_field(l, ξe, plane_d(a, j, k, EY_AT)..., tt - l.delay)[1] * cθ for j in 1:n2, k in 1:n3]
            ez = [laser_field(l, ξe, plane_d(a, j, k, EZ_AT)..., tt - l.delay)[1] * sθ for j in 1:n2, k in 1:n3]
            for j in 1:n2, k in 1:n3
                qb[j, k] += dt / 2 / a.dx[1] * ((ey[j, mod1(k + 1, n3)] - ey[j, k]) / a.dx[3] - (ez[mod1(j + 1, n2), k] - ez[j, k]) / a.dx[2])
            end
        end
    end
    ia = Int(a.ia)
    gauss = gauss_col = divb = divb_col = 0.0
    for J in I
        i, j, k = J[1] - g, J[2] - g, J[3] - g
        r = sum(d -> (E[d][J] - E[d][J - e[d]]) / h[d], 1:3) - (ρ[J] - ρ̄)
        b = sum(d -> (B[d][J + e[d]] - B[d][J]) / h[d], 1:3)
        if i == ia
            gauss_col = max(gauss_col, Float64(absmax(r - qe[j, k])))
        else
            gauss = max(gauss, Float64(absmax(r)))
        end
        if i == ia - 1
            divb_col = max(divb_col, Float64(absmax(b - qb[j, k])))
        else
            divb = max(divb, Float64(absmax(b)))
        end
    end
    hv = map(value_of, h)
    Em = Float64(maximum(c -> maximum(absmax, c[I]), E))
    Bm = Float64(maximum(c -> maximum(absmax, c[I]), B))
    return (; gauss, gauss_col, divb, divb_col, scale = sum(n) * Float64(maximum(absmax, ρ[I])) + sum(2 ./ hv) * Em,
            bscale = sum(2 ./ hv) * Bm)
end

# Gauss's law and ∇·B with a drifting plasma crossing the antenna plane, as T-406: off the plane
# both hold to roundoff, on it the residual is the surface charge of the antenna's sheets,
# computed on the host from the same incident field; the polarization 0.6 drives all four
# corrections. Both deposition modes, the A8 matrix.
function body_T1024(be)
    C = emC("T-1024")
    for T in TYPES, mode in (Reference(), Fast())
        tenth = one(T) / 10
        sp = (SpeciesDescription{T}(:electron, -one(T), one(T), T(1 // 20), zero(T), (1, 0, 0), (2, 1, 1), (3tenth, tenth, -tenth / 2),
                                    (tenth / 2, tenth / 2, tenth / 2), 1),)
        prob = antenna3d_problem(T; ncx = 8, nct = 6, length = (20, 14, 14), waist = 3, duration = 3, position = 5, focus = 10,
                                 polarization = 3 // 5, a0 = 1 // 2, tend = 25, species = sp)
        st = EMState(prob, EMNumerics(; mode, current = Esirkepov()); backend = be)
        initialize!(st, prob)
        p = params(prob)
        worst = worst_col = worst_b = worst_bcol = 0.0
        for k in 1:prob.nsteps
            step!(st, p, k)
            k % 10 == 0 || k == prob.nsteps || continue
            d = antenna_divergences_3d(st, p, k, prob.dt)
            s = base_eps(T) * sqrt(k + 1)
            worst = max(worst, d.gauss / (s * d.scale))
            worst_col = max(worst_col, d.gauss_col / (s * d.scale))
            worst_b = max(worst_b, d.divb / (s * d.bscale))
            worst_bcol = max(worst_bcol, d.divb_col / (s * d.bscale))
        end
        @info "T-1024" T mode worst worst_col worst_b worst_bcol
        record!("T-1024 $(nameof(typeof(mode)))", T, max(worst, worst_col, worst_b, worst_bcol), be)
        @test worst <= C
        @test worst_col <= C
        @test worst_b <= C
        @test worst_bcol <= C
    end
end

@testset "T-1024 with the 3D antenna Gauss's law and ∇·B hold to roundoff off its planes and on them up to its surface charges [A2, A4, A8, WP-073]" begin
    body_T1024(CPU())
end

# --- T-1025 ---------------------------------------------------------------------------------

# A3: the laser's parameters enter through the 3D antenna; the gradient of the intensity of the
# launched beam's ω₀ component on the axis of a probe plane past the focus against focus and
# waist, by ForwardDiff through the whole run, agrees with the finite-difference floor of A7
# (fd_scan: Float64, Reference).
function antenna3d_observable(x, mode, seed; which)
    T = typeof(x)
    kw = which === :focus ? (; focus = x, waist = 3) : (; focus = 12, waist = x)
    prob = antenna3d_problem(T; ncx = 10, nct = 5, length = (30, 18, 18), duration = 3, position = 5, tend = 40, kw...)
    r = antenna3d_run(prob; mode, probes = (20,))
    A = r.A[1].Ey
    return abs2(A[size(A, 1) ÷ 2, size(A, 2) ÷ 2 + 1])
end

@testset "T-1025 laser parameters enter through the 3D antenna: gradients in focus and waist agree with the finite-difference floor [A3, A7, WP-073]" begin
    C = emC("T-1025")
    hs = [3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4]
    for (which, x0) in ((:focus, 12.0), (:waist, 3.0))
        f = (x, mode, seed) -> antenna3d_observable(x, mode, seed; which)
        ad = ForwardDiff.derivative(x -> f(x, Reference(), 1), x0)
        s = fd_scan(f, x0, which === :waist ? hs ./ 10 : hs; seed = 1, ad)
        @info "T-1025" which ad rel = s.rel[s.istar] floor = s.floor_rel
        record!("T-1025 $which", Float64, s.rel[s.istar] / s.floor_rel)
        @test abs(ad) > 0
        @test s.rel[s.istar] <= C * max(s.floor_rel, 1e-10)
    end
end

# --- T-1026 ---------------------------------------------------------------------------------

# The mirror y ↔ z: on a box whose y and z are alike, with the axis on the diagonal, the run
# polarized along z (θ = π/2) is the run polarized along y (θ = 0) mirrored, E as a vector and B
# as a pseudovector: Ey′(x, y, z) = Ez(x, z, y), Ez′ = Ey, Ex′ = Ex, and B′ = −(the same map of B).
# The Yee staggers map onto each other under the exchange (Ey's (Node, Center, Node) onto Ez's
# (Node, Node, Center)), so the two runs agree to roundoff (cos(π/2) is 6·10⁻¹⁷, not zero); a
# sign or a stagger of the cross term ∂_y∂_z u, or of the Ez sheet, breaks it. Float64 and Float32,
# both envelopes, focus off the plane. And the shift: an axis moved by whole cells, across the
# periodic seams, moves the run with it to roundoff, on 23 transverse cells, where a row lies on the
# antipode of the axis and the odd cross term takes the mean of the two images there (with the
# nearest image alone it is 3·10⁻³ of the field there).
function body_T1026(be)
    C = emC("T-1026")
    for T in (Float32, Float64), env in (AxisTime(), WavefrontTime())
        kw = (; length = (30, 18, 18), waist = 3, duration = 3, position = 5, focus = 12, tend = 30, envelope = env, center = (8.0, 8.0))
        ry = antenna3d_run(antenna3d_problem(T; polarization = 0, kw...); be)
        rz = antenna3d_run(antenna3d_problem(T; polarization = π / 2, kw...); be)
        fy, fz = map(Array, ry.st.fields), map(Array, rz.st.fields)
        sw(a) = permutedims(a, (1, 3, 2))
        pairs_ = ((fz.Ex, sw(fy.Ex), 1), (fz.Ey, sw(fy.Ez), 1), (fz.Ez, sw(fy.Ey), 1), (fz.Bx, sw(fy.Bx), -1), (fz.By, sw(fy.Bz), -1),
                  (fz.Bz, sw(fy.By), -1))
        m = maximum(p -> maximum(abs, p[2]), pairs_)
        worst = maximum(p -> maximum(abs, p[1] .- p[3] .* p[2]), pairs_) / (eps(T) * m * sqrt(ry.prob.nsteps))
        @info "T-1026" T env worst
        record!("T-1026 $(nameof(typeof(env)))", T, worst, be)
        @test m > 0 && maximum(abs, fy.By) > 0                     # the cross term launches a By
        @test worst <= C
    end
    h = 2π / 6                                                     # antenna3d_problem's transverse cell
    for T in (Float32, Float64)
        kw = (; length = (40, 24, 24), position = 6, tend = 30, polarization = 3 // 5)
        r0 = antenna3d_run(antenna3d_problem(T; center = (15h, 13h), kw...); be)
        r1 = antenna3d_run(antenna3d_problem(T; center = (2h, 2h), kw...); be)
        g, n = guards(r0.st.grid), ncells(r0.st.grid)
        @test n[2] == n[3] == 23
        I = ((g + 1):(g + n[1]), (g + 1):(g + n[2]), (g + 1):(g + n[3]))
        f0, f1 = map(a -> Array(a)[I...], r0.st.fields), map(a -> Array(a)[I...], r1.st.fields)
        m = maximum(c -> maximum(abs, getfield(f0, c)), (:Ex, :Ey, :Ez, :Bx, :By, :Bz))
        worst = maximum(c -> maximum(abs, circshift(getfield(f0, c), (0, -13, -11)) .- getfield(f1, c)), (:Ex, :Ey, :Ez, :Bx, :By, :Bz)) /
                (eps(T) * m * sqrt(r0.prob.nsteps))
        @info "T-1026 shift" T worst
        record!("T-1026 shift", T, worst, be)
        @test worst <= C
    end
end

@testset "T-1026 the 3D antenna is symmetric under y ↔ z and under shifts of its axis by whole cells across the seams [A2, A8, WP-073]" begin
    body_T1026(CPU())
end

# --- T-1027 ---------------------------------------------------------------------------------

@testset "T-1027 the 3D step with an antenna is bitwise across reruns, launch shapes and 1 and 4 threads in Reference mode [A4, WP-073]" begin
    ref = antenna3d_step_digests()
    @test length(ref) == 2 * length(TYPES) && allunique(last.(ref))
    @test antenna3d_step_digests() == ref
    for wgs in (32, 256)
        @test antenna3d_step_digests(; workgroupsize = wgs) == ref
    end
    for nt in (1, 4)
        fields = split.(em3d_worker_lines(nt, "EM3DA "))
        got = Dict((String(f[3]), String(f[4])) => parse(UInt64, f[5]) for f in fields)
        @test all(f -> parse(Int, f[2]) == nt, fields)
        @test length(got) == length(ref)
        for (label, T, d) in ref
            @test get(got, (label, type_label(T)), nothing) == d
        end
    end
end

# --- T-1028 ---------------------------------------------------------------------------------

# The checks of a laser in 3D: a laser's center has one coordinate at N = 2 and two at N = 3, a
# ShapedLaser (one-dimensional profiles) is refused at N = 3, the plane must keep the guard width
# from the periodic seam, a laser moved off the state's plane is refused by check_params, the
# plane is isbits with the box's numbers, the antennas sit in the state's modifiers after the
# PMLs and before the absorbing layers, and the tunables of a 3D laser round-trip.
@testset "T-1028 input checks and the antenna planes of the 3D state [A2, A3, A10, WP-073]" begin
    T = Float64
    ok = antenna3d_problem(T)
    l = only(ok.lasers)
    @test l.center isa NTuple{2,Float64} && ParticleMeshElectromagnetic.transverse_axes(l) == 2
    mk(las; N = 3) = N == 3 ? EMProblem("x", ok.length, ok.cells, ok.dt, 2, (), 1, (1, 0, 0); lasers = (las,)) :
                     EMProblem("x", ok.length[1:2], ok.cells[1:2], ok.dt, 2, (), 1, (1, 0); lasers = (las,))
    flat = GaussianLaser(; a0 = 1, waist = 3, duration = 3, delay = 18, focus = 5, position = 6, center = 4.0)
    round_ = GaussianLaser(; a0 = 1, waist = 3, duration = 3, delay = 18, focus = 5, position = 6, center = (4.0, 5.0))
    @test_throws ArgumentError mk(flat)                       # a 2D beam in 3D
    @test_throws ArgumentError mk(round_; N = 2)              # a 3D beam in 2D
    @test mk(round_) isa EMProblem{3} && mk(flat; N = 2) isa EMProblem{2}
    shaped = ShapedLaser(; a0 = 1, delay = 18, focus = 5, position = 6, center = 4.0,
                         profile = GaussianBeam(; waist = 3), envelope = GaussianEnvelope(; duration = 3))
    @test_throws ArgumentError mk(shaped)
    @test_throws ArgumentError ShapedLaser(; a0 = 1, delay = 18, focus = 5, position = 6, center = (4.0, 5.0),
                                           profile = GaussianBeam(; waist = 3), envelope = GaussianEnvelope(; duration = 3))
    seam = GaussianLaser(; a0 = 1, waist = 3, duration = 3, delay = 18, focus = 5, position = 0.5, center = (4.0, 5.0))
    @test_throws ArgumentError EMState(mk(seam), EMNumerics())
    st = EMState(ok, EMNumerics())
    plane = only(antenna_planes(st))
    @test plane isa AntennaPlane{T,3} && isbits(plane)
    @test plane.n == Int32.(ncells(st.grid)) && plane.len[2] === ok.length[2] && plane.len[3] === ok.length[3]
    moved = EMParams(params(ok); lasers = (GaussianLaser(; a0 = 1, waist = 4, duration = 5, delay = 30, focus = 14, position = 12,
                                                          center = l.center),))
    @test_throws ArgumentError check_params(st, moved)
    @test check_params(st, params(ok)) === params(ok)
    # the modifiers' order with open axes: the PMLs first, the antennas, the layers last
    open_ok = antenna3d_problem(T; boundaries = (:periodic, :open, :open))
    kinds(s) = map(m -> nameof(typeof(m)), s.modifiers)
    @test kinds(EMState(open_ok, EMNumerics(; absorber = PML(; cells = 4)))) == (:PMLFields, :PMLFields, :AntennaPlanes)
    @test kinds(EMState(open_ok, EMNumerics(; absorber = AbsorbingLayer(; cells = 4, strength = 1)))) ==
          (:AntennaPlanes, :LayerFactors, :LayerFactors)
    # a 3D laser's tunables round-trip with its center
    @test with_tunables(params(ok), tunables(params(ok))) === params(ok)
    @test with_tunables(ok, tunables(ok)) === ok
    # on a cross-section whose y and z differ, with the axis off the diagonal: the center keeps its
    # order through the problem, the parameters and the tunables; the plane holds each axis' own
    # cells, cell size and length; and the plane's offsets are the grid's own positions less the
    # center, to the nearest periodic image
    ns = antenna3d_problem(T; length = (60, 24, 20), nct = (6, 5), center = (13.0, 9.0))
    @test only(ns.lasers).center === (13.0, 9.0) && only(params(ns).lasers).center === (13.0, 9.0)
    @test only(with_tunables(params(ns), tunables(params(ns))).lasers).center === (13.0, 9.0)
    sn = EMState(ns, EMNumerics())
    pn = only(antenna_planes(sn))
    g, n, h = guards(sn.grid), ncells(sn.grid), cellsize(sn.grid)
    @test n[2] != n[3] && h[2] != h[3]
    @test pn.n == Int32.(n) && pn.dx == h && pn.len == ns.length
    an = Antenna(only(params(ns).lasers), pn)
    near(d, len) = d - len * round(d / len)
    off = 0.0
    for d in (2, 3), j in 1:n[d], s in (0.0, 0.5)
        c = an.laser.center[d - 1]
        off = max(off, abs(ParticleMeshElectromagnetic.antenna_d(an, d, c, Int32(g + j), s) - near((j - 1 + s) * h[d] - c, ns.length[d])))
    end
    @test off <= 1e-12
end
