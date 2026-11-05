# WP-034: the back-edge PML's problems, runs and discrete theory (T-570 to T-576, scripts/em2d/pml_back.jl).
# After window_cases.jl, whose `window_lwfa` and `static_twin` it uses.

using ParticleMeshElectromagnetic: pml_profile, pml_sigma_max

"A window state with `back` as the back edge's absorber (and `absorber` on an open y axis)."
function back_state(prob; back = nothing, mode = Reference(), workgroupsize = nothing, absorber = nothing)
    st = EMState(prob, EMNumerics(; mode, workgroupsize, headroom = 1 / 4, back_absorber = back, absorber))
    initialize!(st, prob)
    return st
end

# The back-edge tests' problem: `window_lwfa` with its antenna at x = 10 (column 14) and 72 columns,
# so that the antenna stays out of a layer of up to 8 cells while it emits (the state refuses a
# plane that the layer would reach earlier), and the pulse keeps T-472's distance to the front.
const BACK_POSITION = 10
const BACK_CELLS = (72, 16)
back_lwfa(::Type{T}; kw...) where {T} = window_lwfa(T; position = BACK_POSITION, cells = BACK_CELLS, kw...)

"""
    back_comparison(back; nsteps = 200, extra = 100, position, cells) -> (; shifts, before, far, near, history)

T-472's vacuum pulse in the window with `back` on its back edge, against two references. Its
static twin with the same back edge gives `before` (every column, before the window moves) and
`far` (columns 25 … n₁ after), T-472's comparison of the shift's mechanics. A free-space reference,
a static box with `extra` more columns behind the antenna (moved by as many) and a bare back too far
away for its reflection to come back within the run, gives `near`: the largest difference on the
eight columns behind the layer (cells + 1 … cells + 8), over the whole run. All three are relative
to the reference's largest field at that step; `history` is `near` per step.
"""
function back_comparison(back; nsteps = 200, extra = 100, position = BACK_POSITION, cells = BACK_CELLS)
    prob = window_lwfa(Float64; plasma = false, nsteps, position, cells)
    n = prob.cells
    h = prob.length[1] / n[1]
    win = back_state(prob; back)
    shifts = window_shifts(win, prob.nsteps)
    ref = back_state(static_twin(prob, n[1] + shifts + 8); back)
    free = back_state(window_lwfa(Float64; plasma = false, nsteps, position = position + extra * h,
                                  cells = (n[1] + shifts + 8 + extra, n[2]), start = Inf))
    p = params(prob)
    g = guards(win.grid)
    J = (g + 1):(g + n[2])
    cb = back === nothing ? 0 : back.cells
    before = near = far = 0.0
    history = zeros(prob.nsteps)
    for k in 1:prob.nsteps
        step!(win, p, k)
        step!(ref, p, k)
        step!(free, p, k)
        s = window_shifts(win, k)
        for c in (:Ex, :Ey, :Ez, :Bx, :By, :Bz)
            a, b, f = Array(getfield(win.fields, c)), Array(getfield(ref.fields, c)), Array(getfield(free.fields, c))
            m = maximum(abs, b[(g + 1 + s):(g + n[1] + s), J])
            m > 0 || continue
            d(r) = maximum(abs, a[g .+ r, J] .- b[g .+ s .+ r, J]) / m
            r = (cb + 1):(cb + 8)
            history[k] = max(history[k], maximum(abs, a[g .+ r, J] .- f[g .+ extra .+ s .+ r, J]) / m)
            near = max(near, history[k])
            if s == 0
                before = max(before, d(1:n[1]))
            else
                far = max(far, d(25:n[1]))
            end
        end
    end
    return (; shifts, before, far, near, history)
end

"""
    back_long_run(back; start, nsteps, every = 50, seed = 3) -> (; t, w)

Random fields (standard normal, seed `seed`) in the empty window of `back_lwfa` (no laser, no
plasma) with `back` on its back edge, and the field energy over the window against its start,
every `every` steps: a window that never moves (`start = Inf`) or one that moves from `start`.
"""
function back_long_run(back; start, nsteps, every = 50, seed = 3)
    prob = back_lwfa(Float64; plasma = false, laser = false, nsteps, start)
    st = back_state(prob; back)
    rng = Xoshiro(seed)
    for c in (:Ex, :Ey, :Ez, :Bx, :By, :Bz)
        a = getfield(st.fields, c)
        a .= randn(rng, size(a))
        fill_guards!(a, st.grid)
    end
    g, n = guards(st.grid), ncells(st.grid)
    W() = sum(c -> sum(abs2, view(Array(getfield(st.fields, c)), (g + 1):(g + n[1]), (g + 1):(g + n[2]))), (:Ex, :Ey, :Ez, :Bx, :By, :Bz)) / 2
    W0 = W()
    p = params(prob)
    t, w = [0.0], [1.0]
    for k in 1:nsteps
        step!(st, p, k)
        k % every == 0 && (push!(t, k * Float64(prob.dt)); push!(w, W() / W0))
    end
    return (; t, w)
end

# --- a packet into the back edge of a static window, and its discrete theory (T-576) ----------
#
# WP-027's slab packet turned on its side: in a window that never moves (start = Inf), periodic
# along y, a packet travels along −x into the back-edge PML, behind which the window's zero x-guards
# (its field-free outside) reflect what the layer leaves. Nothing is transmitted, so the energy that
# comes back is the reflection of the layer and its termination together, and the scheme's
# time-harmonic step predicts it (`back_rt`) as `pml_rt` predicts the y-axis slab.

"""
    back_slab_geometry(; theta, cells = 8, ncx = 8, ncy = 16, width = 3, courant = 95//100, margin = 8)

The box of a back-edge reflection run at the carrier angle `theta` from the layer's normal (−x),
|k| = 1: along y one period of k_y = sin θ (or 8 cells at θ = 0) at about `ncy` cells per
wavelength, along x `ncx` cells per wavelength (Δx ≠ Δy). The packet (field envelope of 1/e half
width `width` wavelengths) starts at x₀ = thick + 2·margin·w and the front is 2·margin·w beyond it;
the run ends when the reflection is back at x₀, and the energy from `split` = thick + margin·w to
the front is the reflected energy.
"""
function back_slab_geometry(; theta, cells = 8, ncx = 8, ncy = 16, width = 3, courant = 95 // 100, margin = 8)
    ky = sin(theta)
    kx0 = cos(theta)
    if ky == 0
        ny, Ly = 8, 8 * 2π / ncy
    else
        Ly = 2π / ky
        ny = max(8, round(Int, ncy * Ly / 2π))
    end
    dx = 2π / ncx
    dy = Ly / ny
    w = width * 2π
    thick = cells * dx
    x0 = thick + 2margin * w
    nx = ceil(Int, (x0 + 2margin * w) / dx)
    Lx = nx * dx
    dt = courant / sqrt(1 / dx^2 + 1 / dy^2)
    tmeas = 2 * (x0 - thick) / kx0
    return (; theta, ky, kx0, nx, ny, Lx, Ly, dx, dy, dt, w, cells, thick, x0, split = thick + margin * w, tmeas, margin)
end

"The modes of the packet: k_x = 2πm/Lx (the packet travels along −x) with their amplitudes, within 8/w of the carrier."
function back_slab_modes(geo)
    out = Tuple{Float64,ComplexF64}[]
    for m in 1:(geo.nx ÷ 2 - 1)
        kx = 2π * m / geo.Lx
        abs(kx - geo.kx0) * geo.w <= 8 || continue
        push!(out, (kx, exp(-(kx - geo.kx0)^2 * geo.w^2 / 2) * cis(kx * geo.x0)))
    end
    return out
end

"The Yee ω of the mode (k_x, k_y) of the back-edge run."
back_omega(geo, kx) = 2 / geo.dt * asin(geo.dt * sqrt(sin(kx * geo.dx / 2)^2 / geo.dx^2 + sin(geo.ky * geo.dy / 2)^2 / geo.dy^2))

"The static window of a back-edge run in element type `T`, with `pml` on its back edge, initialized and holding the packet."
function back_slab_state(::Type{T}, geo, pml, pol; nsteps = round(Int, geo.tmeas / geo.dt)) where {T}
    prob = EMProblem("back-slab", (T(geo.Lx), T(geo.Ly)), (geo.nx, geo.ny), T(geo.dt), nsteps, (), 1, (1, 1);
                     boundaries = (:window, :periodic), window = MovingWindow(; start = T(Inf)))
    st = EMState(prob, EMNumerics(; current = NoCurrent(), back_absorber = pml))
    initialize!(st, prob)
    back_slab_init!(st, geo, pol)
    return st, prob
end

# The packet as `slab_init!` writes WP-027's, with the wave vector (−k_x, k_y): TE Ez with
# B^{1/2} = (κ_y, −κ_x)/|κ|·Ez, TM Bz^{1/2} with E = (−κ_y, κ_x)/|κ|·Bz^{1/2}, κ_d = 2 sin(k_dΔ_d/2)/Δ_d
# (signed), then B taken back to t = 0 by the solver's half step.
function back_slab_init!(st, geo, pol)
    (; grid, fields) = st
    T = eltype(st.ρ)
    g, n = guards(grid), ncells(grid)
    modes = back_slab_modes(geo)
    host = Dict(c => zeros(Float64, n[1] + 2g, n[2] + 2g) for c in (:Ex, :Ey, :Ez, :Bx, :By, :Bz))
    κy = 2 * sin(geo.ky * geo.dy / 2) / geo.dy
    function fill_comp!(a, sx, sy, coef)
        for j in axes(a, 2), i in axes(a, 1)
            (g < i <= g + n[1]) || continue          # the x-guards stay zero (the field-free outside)
            x = (i - g - 1 + sx) * geo.dx
            y = (j - g - 1 + sy) * geo.dy
            v = 0.0im
            for (kx, A) in modes
                v += coef(kx) * A * cis(-kx * x + geo.ky * y)
            end
            a[i, j] = real(v)
        end
    end
    κn(kx) = (κx = -2 * sin(kx * geo.dx / 2) / geo.dx; (κx, hypot(κx, κy)))
    ph(kx) = cis(-back_omega(geo, kx) * geo.dt / 2)
    if pol === :TE
        fill_comp!(host[:Ez], 0, 0, kx -> 1.0)
        fill_comp!(host[:Bx], 0, 1 / 2, kx -> ((κx, κ) = κn(kx); κy / κ * ph(kx)))
        fill_comp!(host[:By], 1 / 2, 0, kx -> ((κx, κ) = κn(kx); -κx / κ * ph(kx)))
    else
        fill_comp!(host[:Bz], 1 / 2, 1 / 2, kx -> ph(kx))
        fill_comp!(host[:Ex], 1 / 2, 0, kx -> ((κx, κ) = κn(kx); -κy / κ))
        fill_comp!(host[:Ey], 0, 1 / 2, kx -> ((κx, κ) = κn(kx); κx / κ))
    end
    for c in (:Ex, :Ey, :Ez, :Bx, :By, :Bz)
        a = getfield(fields, c)
        copyto!(a, T.(host[c]))
        fill_guards!(a, grid)
    end
    ParticleMeshElectromagnetic.yee_b_half!(fields, grid, -st.aux.dt / 2)
    return st
end

"½Σ(E² + B²)·ΔxΔy over the array columns `cols` and every interior row (Float64 on the host)."
function back_slab_energy(st, cols)
    g, n, h = guards(st.grid), ncells(st.grid), cellsize(st.grid)
    W = 0.0
    for c in (:Ex, :Ey, :Ez, :Bx, :By, :Bz)
        W += sum(abs2 ∘ Float64 ∘ value_of, view(Array(getfield(st.fields, c)), cols, (g + 1):(g + n[2])))
    end
    return W / 2 * Float64(value_of(h[1])) * Float64(value_of(h[2]))
end

"""
    back_slab_run(T, geo, pml, pol) -> (; R, W0, k)

The packet of `geo` into the back edge with `pml`: `R` the energy from `geo.split` to the front at
the measurement time over the interior's initial energy `W0` (the columns off the layer).
"""
function back_slab_run(::Type{T}, geo, pml, pol) where {T}
    st, prob = back_slab_state(T, geo, pml, pol)
    g, n = guards(st.grid), ncells(st.grid)
    W0 = back_slab_energy(st, (g + geo.cells + 1):(g + n[1]))
    p = params(prob)
    for k in 1:prob.nsteps
        step!(st, p, k)
    end
    js = g + 1 + round(Int, geo.split / geo.dx)
    return (; R = back_slab_energy(st, js:(g + n[1])) / W0, W0, k = prob.nsteps)
end

"""
    back_rt(geo, pml, pol, kx) -> r

The reflection amplitude of the back edge (the layer of `pml` and the field-free outside behind
it) for the incident mode (−k_x, k_y) at its Yee ω, from the time-harmonic step as in `pml_rt`
with x the layer's axis: G = h(2/κ + 2c/(1 − b/λ))/(1 − 1/λ) at B's Center() columns, S = 1/κ +
λc/(λ − b) at E's Node() columns, the tangential difference giving Δt²κ_y². The termination is
the window's zero x-guards: TE (Ez on Node() columns j ≥ 0) without the By of the guard column
behind node 0, TM (the half-step Bz on Center() columns j ≥ 0) with that guard's Bz zero. The
recurrence runs from the termination into the vacuum, where it splits into the incident and the
reflected roots. Independent of the kernels except for `pml_profile`.
"""
function back_rt(geo, pml, pol, kx)
    dt, dx = geo.dt, geo.dx
    h = dt / 2
    λ = cis(-back_omega(geo, kx) * dt)
    κt = 2 * sin(geo.ky * geo.dy / 2) / geo.dy
    sm = pml_sigma_max(pml, Float64(dx))
    function bc(p)
        σ, κ, α = pml_profile(sm, pml.kappa_max, pml.alpha_max, (pml.order, pml.kappa_order, pml.alpha_order), p, geo.nx, geo.cells)
        b = exp(-(σ / κ + α) * dt)
        c = σ > 0 ? σ * (b - 1) / (κ * (σ + κ * α)) : 0.0
        return (b, c, κ)
    end
    G(pc) = ((b, c, κ) = bc(pc); h * (2 / κ + 2c / (1 - b / λ)) / (1 - 1 / λ))
    S(pn) = ((b, c, κ) = bc(pn); 1 / κ + λ * c / (λ - b))
    if pol === :TE
        A = (λ - 1) + dt^2 * κt^2 / (1 - 1 / λ)
        coefs = j -> begin
            s = dt * S(2j) / dx^2
            cp, cm = -s * G(2j + 1), -s * G(2j - 1)
            (cp, A - cp - (j == 0 ? 0 : cm), cm)
        end
    else
        A = 1 + dt^2 * κt^2 / ((1 - 1 / λ) * (λ - 1))
        coefs = j -> begin
            s = dt * G(2j + 1) / (dx^2 * (λ - 1))
            cp, cm = -s * S(2j + 2), -s * S(2j)
            (cp, A - cp - cm, cm)
        end
    end
    j1 = geo.cells + 3
    Fm, F = 0.0im, 1.0 + 0.0im                     # F₋₁ (no term at TE's node 0; zero guard for TM), F₀
    for j in 0:j1
        cp, c0, cm = coefs(j)
        Fp = -(c0 * F + cm * Fm) / cp
        Fm, F = F, Fp
    end
    # F = F_{j1+1}, Fm = F_{j1}: split into a·μ_in^j + b·μ_out^j about j1
    cp, c0, cm = coefs(j1 + 1)
    disc = sqrt(c0^2 - 4 * cp * cm)
    μs = ((-c0 + disc) / (2cp), (-c0 - disc) / (2cp))
    μout, μin = angle(μs[1]) > 0 ? μs : (μs[2], μs[1])
    b = (F - μin * Fm) / (μout - μin)
    a = Fm - b
    isfinite(b / a) || error("back_rt: the recurrence is singular for this profile")
    return b / a
end

"The energy fraction the back edge's theory predicts for the packet, weighted as `pml_prediction`."
function back_prediction(geo, pml, pol)
    num = den = 0.0
    for (kx, A) in back_slab_modes(geo)
        w = abs2(A) * (1 + cos(back_omega(geo, kx) * geo.dt / 2)^2)
        num += w * abs2(back_rt(geo, pml, pol, kx))
        den += w
    end
    return num / den
end
