# WP-027: the open transverse boundary's runs and their analysis, shared by boundaries.jl and
# scripts/em2d/absorbing_layer.jl.
#
# The reflection runs launch, in vacuum, a slab packet: one wavenumber k_x along the periodic x
# (one period of the box) and a Gaussian envelope along y, built from the scheme's own upward
# eigenmodes, so it travels one way to roundoff. It hits the top layer at the angle θ of its
# carrier, tan θ = k_x/k_y0. What comes back into the interior is the reflection (going down,
# from the top layer) and the transmission through both layers and the seam (going up, out of the
# bottom layer); at the measurement time the two are packets in the upper and lower halves of the
# interior. Their energies over the incident energy are compared with the reflection and
# transmission of the same discrete scheme in the frequency domain (`layer_rt`): per mode, the
# time-harmonic step reduces along y to a three-point recurrence, solved through the layer slab
# with outgoing waves on the far side.

using ParticleMeshElectromagnetic: LayerFactors, layer_sigma, yee_b_half!

"""
    slab_geometry(; theta, cells = 16, strength = 2.0, order = 2, ncx = 16, ncy = 8, width = 3, courant = 95//100, margin = 8)

The box of a reflection run at the carrier angle `theta` (radians from the layer's normal), for a
carrier |k| = 1 (ω ≈ ω₀): along x one period of k_x = sin θ (or 8 cells at θ = 0) with about `ncx`
cells per wavelength 2π, along y `ncy` cells per wavelength; the packet's field envelope has the
1/e half width `width` wavelengths. The interior holds the packet's start and, at the measurement
time, the reflected and transmitted packets apart, each `margin` envelope widths from the layers
and from the split between them (an energy tail of exp(−margin²) of the packet beyond): its
height is 2·(2·margin·w + L) + 4w with w = width·2π and L the layer thickness. Returns a NamedTuple of the sizes, the carrier and the times.
"""
function slab_geometry(; theta, cells = 16, strength = 2.0, order = 2, ncx = 16, ncy = 8, width = 3, courant = 95 // 100, margin = 8)
    kx = sin(theta)
    ky0 = cos(theta)
    if kx == 0
        nx, Lx = 8, 8 * 2π / ncx
    else
        Lx = 2π / kx
        nx = max(8, round(Int, ncx * Lx / 2π))
    end
    dy = 2π / ncy
    w = width * 2π
    thick = cells * dy
    Hi = 2 * (2margin * w + thick) + 4w
    ny = 2cells + ceil(Int, Hi / dy)
    Ly = ny * dy
    dx = Lx / nx
    dt = courant / sqrt(1 / dx^2 + 1 / dy^2)
    # interior [thick, Ly − thick); start the packet in its middle, measure when the reflection is
    # a distance D below the top edge: D between the margins of the transmitted and reflected packets.
    ylo, yhi = thick, Ly - thick
    y0 = (ylo + yhi) / 2
    D = ((margin * w + 2thick) + (Hi / 2 + thick - margin * w)) / 2
    vy = ky0                       # the packet's speed along y, to the continuum's order (within the margins)
    tmeas = ((yhi - y0) + D) / vy
    split = ylo + (Hi - 2thick) / 2
    return (; theta, kx, ky0, nx, ny, Lx, Ly, dx, dy, dt, w, cells, strength, order, thick, ylo, yhi, y0, tmeas, split, Hi, margin)
end

"The vacuum problem and the numerics of a reflection run of `geo` in element type `T` (open axis 2)."
function slab_problem(::Type{T}, geo; nsteps = round(Int, geo.tmeas / geo.dt)) where {T}
    prob = EMProblem("slab", (T(geo.Lx), T(geo.Ly)), (geo.nx, geo.ny), T(geo.dt), nsteps, (), 1, (1, 1);
                     boundaries = (:periodic, :open))
    num = EMNumerics(; current = NoCurrent(), absorber = AbsorbingLayer(; cells = geo.cells, strength = geo.strength, order = geo.order))
    return prob, num
end

"The modes of the slab packet: wavenumbers k_y = 2πm/Ly with their complex amplitudes, the upward ones within 8/w of the carrier."
function slab_modes(geo)
    out = Tuple{Float64,ComplexF64}[]
    for m in 1:(geo.ny ÷ 2 - 1)
        ky = 2π * m / geo.Ly
        abs(ky - geo.ky0) * geo.w <= 8 || continue
        push!(out, (ky, exp(-(ky - geo.ky0)^2 * geo.w^2 / 2) * cis(-ky * geo.y0)))
    end
    return out
end

"The Yee ω of the mode (k_x, k_y) on the run's grid: sin(ωΔt/2)/Δt = √(Σ sin²(kΔ/2)/Δ²)."
yee_omega(geo, ky) = 2 / geo.dt * asin(geo.dt * sqrt(sin(geo.kx * geo.dx / 2)^2 / geo.dx^2 + sin(ky * geo.dy / 2)^2 / geo.dy^2))

"""
    slab_init!(st, geo, pol)

Write the slab packet into the state's fields at t = 0 (polarization `:TE`, Ez with Bx, By, or
`:TM`, Bz with Ex, Ey): the sum of the scheme's upward eigenmodes of [`slab_modes`](@ref), with E at
t = 0 and B at t = Δt/2 from the discrete Faraday and Ampère relations, then B taken back to t = 0
by the solver's own half step, so the state holds Eⁿ, Bⁿ as the step expects. TE: Ez = Re Σ A·e^{i k·r},
B^{1/2} = (κ_y, −κ_x)/|κ|·Ez; TM: Bz^{1/2} = Re Σ A·e^{i(k·r − ωΔt/2)}, E = (−κ_y, κ_x)/|κ|·Bz^{1/2}
taken at t = 0, with κ_d = 2 sin(k_dΔ_d/2)/Δ_d.
"""
function slab_init!(st, geo, pol)
    (; grid, fields) = st
    T = eltype(st.ρ)
    g, n = guards(grid), ncells(grid)
    modes = slab_modes(geo)
    host = Dict(c => zeros(Float64, n[1] + 2g, n[2] + 2g) for c in (:Ex, :Ey, :Ez, :Bx, :By, :Bz))
    κx = 2 * sin(geo.kx * geo.dx / 2) / geo.dx
    # the position of array index (i, j) at stagger (sx, sy): ((i − g − 1 + sx)·Δx, (j − g − 1 + sy)·Δy)
    function fill_comp!(a, sx, sy, coef)
        for j in axes(a, 2), i in axes(a, 1)
            x = (i - g - 1 + sx) * geo.dx
            y = (j - g - 1 + sy) * geo.dy
            v = 0.0im
            for (ky, A) in modes
                v += coef(ky) * A * cis(geo.kx * x + ky * y)
            end
            a[i, j] = real(v)
        end
    end
    κn(ky) = (κy = 2 * sin(ky * geo.dy / 2) / geo.dy; (κy, hypot(κx, κy)))
    ph(ky) = cis(-yee_omega(geo, ky) * geo.dt / 2)
    if pol === :TE
        fill_comp!(host[:Ez], 0, 0, ky -> 1.0)
        fill_comp!(host[:Bx], 0, 1 / 2, ky -> ((κy, κ) = κn(ky); κy / κ * ph(ky)))
        fill_comp!(host[:By], 1 / 2, 0, ky -> ((κy, κ) = κn(ky); -κx / κ * ph(ky)))
    else
        fill_comp!(host[:Bz], 1 / 2, 1 / 2, ky -> ph(ky))
        fill_comp!(host[:Ex], 1 / 2, 0, ky -> ((κy, κ) = κn(ky); -κy / κ))
        fill_comp!(host[:Ey], 0, 1 / 2, ky -> ((κy, κ) = κn(ky); κx / κ))
    end
    for c in (:Ex, :Ey, :Ez, :Bx, :By, :Bz)
        copyto!(getfield(fields, c), T.(host[c]))
    end
    yee_b_half!(fields, grid, -st.aux.dt / 2)       # B^{1/2} → B⁰ = B^{1/2} + (Δt/2)∇×E⁰
    return st
end

"""
    slab_energy(st, geo; rows) -> W

½Σ(E² + B²)·ΔxΔy over the interior columns and the array `rows` (Float64 on the host).
"""
function slab_energy(st, rows)
    (; grid, fields) = st
    g, n, h = guards(grid), ncells(grid), cellsize(grid)
    cols = (g + 1):(g + n[1])
    W = 0.0
    for c in (:Ex, :Ey, :Ez, :Bx, :By, :Bz)
        a = Array(getfield(fields, c))
        W += sum(abs2 ∘ Float64 ∘ value_of, view(a, cols, rows))
    end
    return W / 2 * Float64(value_of(h[1])) * Float64(value_of(h[2]))
end

"The array rows of the interior (no layer row), of its lower half and of its upper half, split at `geo.split`."
function slab_rows(st, geo)
    g, n = guards(st.grid), ncells(st.grid)
    jlo, jhi = g + geo.cells + 1, g + n[2] - geo.cells
    js = g + 1 + round(Int, geo.split / geo.dy)
    return (; interior = jlo:jhi, lower = jlo:(js - 1), upper = js:jhi)
end

"""
    slab_run(T, geo, pol; history = false, be = CPU(), workgroupsize = nothing) -> (; R, Tr, W0, st, hist)

A reflection run: the slab packet in vacuum to the measurement time; `R` and `Tr` the energies of
the upper and lower halves of the interior over the initial energy `W0` (the reflected and the
transmitted packets), `hist` the interior energy every 10 steps when `history`.
"""
function slab_run(::Type{T}, geo, pol; history = false, be = CPU(), workgroupsize = nothing) where {T}
    prob, num = slab_problem(T, geo)
    num = EMNumerics(; current = num.current, absorber = num.absorber, workgroupsize)
    st = EMState(prob, num; backend = be)
    initialize!(st, prob)
    slab_init!(st, geo, pol)
    rows = slab_rows(st, geo)
    W0 = slab_energy(st, rows.interior)
    p = params(prob)
    hist = Tuple{Float64,Float64,Float64}[]
    for k in 1:prob.nsteps
        step!(st, p, k)
        if history && k % 10 == 0
            push!(hist, (k * geo.dt, slab_energy(st, rows.lower) / W0, slab_energy(st, rows.upper) / W0))
        end
    end
    return (; R = slab_energy(st, rows.upper) / W0, Tr = slab_energy(st, rows.lower) / W0, W0, st, hist)
end

# --- the discrete theory ---------------------------------------------------------------------

"The layer's σ at the half row `p` (height p·Δy/2, any integer: wrapped into the box), the kernel's profile in Float64."
sigma_at(geo, p::Integer) = layer_sigma(Float64(geo.strength), geo.order, mod(p, 2 * geo.ny), geo.ny, geo.cells)

"""
    layer_rt(geo, pol, ky) -> (r, t)

The reflection and transmission amplitudes of the absorbing slab (the top layer, the seam and the
bottom layer) for the scheme's mode (k_x, k_y) at its Yee ω, from the time-harmonic step: with
every field ∝ e^{i k_x x}·λⁿ, λ = e^{−iωΔt}, the x-differences become iκ_x, and a sub-step of
length τ with its symmetric damping becomes X′ = a·X + b·U(X) at the component's row, a = e^{−στ},
b = e^{−στ/2}, U the undamped update's increment. Eliminating the components without a
y-difference leaves a three-point recurrence along y, for Ez on the Node() rows (TE) or for the
half-step Bz on the Center() rows (TM):

    TE: ((λ − a_E)/b_E − Δt·iκ_x·γ_j)·Ez_j + (Δt/Δy²)·(β_{j+½}(Ez_{j+1} − Ez_j) − β_{j−½}(Ez_j − Ez_{j−1})) = 0
        β = −b_B h(1 + a_B)/(1 − a_B²/λ) on Center() rows, γ = b_B·h·iκ_x(1 + a_B)/(1 − a_B²/λ) on Node() rows
    TM: (1 − a_B²/λ)·u_j + b_B h(1 + a_B)·(iκ_x η_j u_j − (ε_{j+1}(u_{j+1} − u_j) − ε_j(u_j − u_{j−1}))/Δy²) = 0
        u = Bz^{n+½}, ε = b_E Δt/(λ − a_E) on Node() rows, η = −b_E Δt iκ_x/(λ − a_E) on Center() rows

(h = Δt/2; B's sub-steps have τ = h, E's τ = Δt). Outgoing above the slab (the upward root μ₊ of
the vacuum recurrence), iterated down through the slab, and split below it into μ₊ʲ (incident)
and μ₋ʲ (reflected). Independent of the kernels: it shares only the profile `layer_sigma`, which
it evaluates at integer half rows as the kernel does.
"""
function layer_rt(geo, pol, ky)
    dt, dy = geo.dt, geo.dy
    h = dt / 2
    ω = yee_omega(geo, ky)
    λ = cis(-ω * dt)
    κx = 2 * sin(geo.kx * geo.dx / 2) / geo.dx
    ia = im * κx
    # (a, b) = (e^{−στ}, e^{−στ/2}) at the half row p
    ab(p, τ) = (σ = sigma_at(geo, p); (exp(-σ * τ), exp(-σ * τ / 2)))
    # the unwrapped rows through the slab: from 3 rows below the top layer to 3 rows above the bottom
    # one; row j sits at the half row 2j (Node(), TE) or 2j + 1 (Center(), TM), j0 = first row.
    j0 = geo.ny - geo.cells - 3
    j1 = geo.ny + geo.cells + 3
    if pol === :TE
        β(pc) = ((a, b) = ab(pc, h); -b * h * (1 + a) / (1 - a^2 / λ))
        γ(pn) = ((a, b) = ab(pn, h); b * h * ia * (1 + a) / (1 - a^2 / λ))
        coefs = j -> begin
            pn = 2j
            cp, cm = dt / dy^2 * β(pn + 1), dt / dy^2 * β(pn - 1)
            aE, bE = ab(pn, dt)
            (cp, (λ - aE) / bE - dt * ia * γ(pn) - cp - cm, cm)
        end
    else
        ε(pn) = ((a, b) = ab(pn, dt); b * dt / (λ - a))
        η(pc) = ((a, b) = ab(pc, dt); -b * dt * ia / (λ - a))
        coefs = j -> begin
            pc = 2j + 1
            aB, bB = ab(pc, h)
            c = bB * h * (1 + aB)
            ep, em = ε(pc + 1), ε(pc - 1)
            (-c * ep / dy^2, (1 - aB^2 / λ) + c * (ia * η(pc) + ep / dy^2 + em / dy^2), -c * em / dy^2)
        end
    end
    # the vacuum roots of c₊μ² + c₀μ + c₋ = 0, the upward one with arg μ > 0
    cp, c0, cm = coefs(j0)
    disc = sqrt(c0^2 - 4 * cp * cm)
    μs = ((-c0 + disc) / (2cp), (-c0 - disc) / (2cp))
    μu, μd = angle(μs[1]) > 0 ? μs : (μs[2], μs[1])
    # outgoing above: F_j = μu^(j − j1) for j ≥ j1 − 1; iterate down to j0 − 1
    Fp, F = μu, one(ComplexF64)                       # F_{j+1}, F_j at j = j1
    for j in j1:-1:j0
        cp, c0, cm = coefs(j)
        Fm = -(cp * Fp + c0 * F) / cm
        Fp, F = F, Fm
    end
    # now F = F_{j0−1}, Fp = F_{j0}: split into a·μu^(j−j0) + b·μd^(j−j0)
    # j = j0: a + b = Fp; j = j0 − 1: a/μu + b/μd = F
    b = (F - Fp / μu) / (1 / μd - 1 / μu)
    a = Fp - b
    # amplitudes relative to the incident one at a common row: |t| = |μu^(j1−j0)|/|a| = 1/|a| (|μu| = 1)
    return (b / a, 1 / a)
end

"""
    slab_prediction(geo, pol) -> (; R, Tr)

The energy fractions the discrete theory predicts for the slab packet: Σ|A|²·|r|² over Σ|A|² (and
|t|²), each mode's weight the energy ½Σ(E² + B²) of its eigenvector at integer times,
1 + cos²(ωΔt/2) per unit amplitude (E and B of unit amplitude at the half steps, Bⁿ their mean).
"""
function slab_prediction(geo, pol)
    num_r = num_t = den = 0.0
    for (ky, A) in slab_modes(geo)
        r, t = layer_rt(geo, pol, ky)
        w = abs2(A) * (1 + cos(yee_omega(geo, ky) * geo.dt / 2)^2)
        num_r += w * abs2(r)
        num_t += w * abs2(t)
        den += w
    end
    return (; R = num_r / den, Tr = num_t / den)
end

# --- the plasma with an open axis 2 -------------------------------------------------------------

"""
    open_plasma_2d(T; cells = (10, 16), ppc = (3, 3), nsteps = 40, layer = 4, strength = 1, order = 2, seed = 7)
        -> (problem, absorber)

The two mobile species of `plasma_step_2d` (thermal electrons with the drift (0.3, −0.25, 0.1)
and an oblique perturbation, positive particles of mass 4 drifting along (−0.2, 0.3, −0.15)) on
a box of 2π × 4π, open along y with an absorbing layer of `layer` cells: within the 40 steps a
few hundred particles leave through either edge, others cross the periodic x seam.
"""
function open_plasma_2d(::Type{T}; cells = (10, 16), ppc = (3, 3), nsteps = 40, layer = 4, strength = 1, order = 2,
                        seed = 7, amplitude = one(T) / 10) where {T}
    L = (2 * T(π), 4 * T(π))
    dt = 9 * inv(sqrt(sum(d -> (cells[d] / L[d])^2, 1:2))) / 10
    tenth = one(T) / 10
    e = SpeciesDescription{T}(:electron, -one(T), one(T), one(T), T(amplitude), (2, 1), ppc, (3tenth, -5tenth / 2, tenth),
                              (tenth, tenth, tenth), 1)
    p = SpeciesDescription{T}(:positive, one(T), 4 * one(T), one(T) / 2, tenth / 2, (1, -1), ppc, (-2tenth, 3tenth, -3tenth / 2),
                              (tenth / 2, tenth / 2, tenth / 2), 2)
    prob = EMProblem("open-plasma-2d", L, cells, T(dt), nsteps, (e, p), seed, (2, 1); boundaries = (:periodic, :open))
    return prob, AbsorbingLayer(; cells = layer, strength, order)
end

"The initialized state of `open_plasma_2d` in `mode` with the `Esirkepov()` current."
function open_state(prob, absorber; mode = Reference(), workgroupsize = nothing, be = CPU())
    st = EMState(prob, EMNumerics(; mode, workgroupsize, current = Esirkepov(), absorber); backend = be)
    initialize!(st, prob)
    return st
end

"""
    open_step_digests(; workgroupsize = nothing, be = CPU()) -> Vector{(T, digest)}

T-454: a Reference run of [`open_plasma_2d`](@ref) (20 steps, particles retired at both edges,
the layer damping) per element type, one digest of every particle array, field and history each.
"""
function open_step_digests(; workgroupsize = nothing, be = CPU())
    out = Tuple{DataType,UInt64}[]
    for T in TYPES
        prob, ab = open_plasma_2d(T; nsteps = 20)
        st = open_state(prob, ab; workgroupsize, be)
        run!(st, params(prob), prob.nsteps)
        push!(out, (T, state_digest_2d(st)))
    end
    return out
end
