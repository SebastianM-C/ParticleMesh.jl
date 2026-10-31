"""
    ParticleMeshCore.Electrostatic

The electrostatic 1D periodic PIC module of the spike (WP-010). It uses the engine pieces
(species, shapes, sort, deposition, gather, guards, loading) and owns the physics: what is
deposited (charge `q·w`), the field solve (a discrete Gauss law by a prefix sum), the force
(`qE`), the pusher (leapfrog, nonrelativistic, velocities in units of c) and the order of
the step.

Normalized units (A1): time in 1/ω₀, length in c/ω₀, density in n_c, field in mₑcω₀/e. Then
Gauss's law reads ∂E/∂x = ρ and the equation of motion dv/dt = (q/m)E; for electrons of density
n₀ the plasma frequency is ω_p = √n₀ (in units of ω₀).

Layout: charge density `ρ` and the gathered field `Enode` on the nodes of the shape's natural
stagger (`Node()` for order 1, `Center()` for order 2), both with guards; `Ehalf[j]` is the field
half a cell to the right of node `j`. The solve makes
`(Ehalf[j] - Ehalf[j-1])/dx = ρ[j] - mean(ρ)` exactly up to roundoff (the mean is the immobile,
uniform neutralizing background) and `mean(Ehalf) = 0`; `Enode[j]` is the average of the two
neighbouring half-node values, which with the same shape for gather and deposit gives the
momentum-conserving scheme (no self-force).
"""
module Electrostatic

using KernelAbstractions
using KernelAbstractions: @kernel, @index, get_backend
using ..ParticleMeshCore
using ..ParticleMeshCore: ALIVE, gather_at, advance, Workspace, Stagger, Shape

export ESConfig, ESState, ESParams, natural_stagger, initialize!, step!, run!
export CosineDensity, landau_case, two_stream_case, field_energy_history, mode_amplitude_history
export kinetic_energy, linear_fit, fit_damping, fit_growth

"Static configuration of the module: shape, stagger and deposition mode, all type-level."
struct ESConfig{S<:Shape,G<:Stagger,M}
    shape::S
    stagger::G
    mode::M
end

natural_stagger(::Shape{1}) = Node()
natural_stagger(::Shape{2}) = Center()
ESConfig(shape::Shape, mode) = ESConfig(shape, natural_stagger(shape), mode)

"Run parameters: the time step and the wavenumber of the recorded mode, both of type `T`."
struct ESParams{T}
    dt::T
    kmode::T
end

"""
    ESState

Everything a step reads and writes: the species and their workspaces (tuples), the grid, the
fields (`ρ`, `Enode` with guards; `Ehalf` interior only) and the diagnostic histories (field
energy and the real and imaginary parts of the Fourier amplitude of `Ehalf` at `kmode`), one
entry per step.
"""
struct ESState{C<:ESConfig,G,SP<:Tuple,WS<:Tuple,F,H}
    config::C
    grid::G
    species::SP
    ws::WS
    ρ::F
    Ehalf::F
    Enode::F
    W::H
    Ere::H
    Eim::H
end

function ESState(config::ESConfig, grid::PeriodicGrid{1}, species::Tuple, nsteps::Integer)
    T = eltype(species[1])
    backend = get_backend(species[1])
    ws = map(s -> Workspace(s, grid), species)
    ρ = field(backend, T, grid)
    Enode = field(backend, T, grid)
    Ehalf = KernelAbstractions.zeros(backend, T, ncells(grid))
    hist() = KernelAbstractions.zeros(backend, T, nsteps)
    return ESState(config, grid, species, ws, ρ, Ehalf, Enode, hist(), hist(), hist())
end

# Prefix-sum Gauss solve in one work item: deterministic, exact Gauss law up to roundoff, no
# FFT (which would not take `Dual`). O(n) and serial; a GPU version would use a scan.
@kernel function gauss_kernel!(Ehalf, Enode, @Const(ρ), dx, n::Int32, g::Int32)
    T = eltype(Ehalf)
    @inbounds begin
        ρsum = zero(T)
        for j in Int32(1):n
            ρsum += ρ[g + j]
        end
        ρ̄ = ρsum / T(n)
        acc = zero(T)
        Esum = zero(T)
        for j in Int32(1):n
            acc += (ρ[g + j] - ρ̄) * dx
            Ehalf[j] = acc
            Esum += acc
        end
        Ē = Esum / T(n)
        for j in Int32(1):n
            Ehalf[j] -= Ē
        end
        for j in Int32(1):n
            jm = ifelse(j == Int32(1), n, j - Int32(1))
            Enode[g + j] = (Ehalf[jm] + Ehalf[j]) / 2
        end
    end
end

@kernel function diagnostics_kernel!(W, Ere, Eim, @Const(Ehalf), dx, kmode, x0, n::Int32, step::Int32)
    T = eltype(Ehalf)
    @inbounds begin
        w = zero(T)
        re = zero(T)
        im = zero(T)
        for j in Int32(1):n
            e = Ehalf[j]
            x = x0 + T(j - Int32(1)) * dx
            w += e * e
            re += e * cos(kmode * x)
            im -= e * sin(kmode * x)
        end
        W[step] = w * dx / 2
        Ere[step] = re * 2 / T(n)
        Eim[step] = im * 2 / T(n)
    end
end

@kernel function push_kernel!(s, @Const(Enode), shape, stagger, g::Int32, qmdt, dt_dx, n::Int32)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE
        c = s.cell[1][i]
        δ = s.offset[1][i]
        v = s.u[1][i] + qmdt * gather_at(Enode, c, δ, shape, stagger, g)
        s.u[1][i] = v
        c′, δ′ = advance(c, δ, v * dt_dx, n)
        s.cell[1][i] = c′
        s.offset[1][i] = δ′
    end
end

@kernel function kick_kernel!(s, @Const(Enode), shape, stagger, g::Int32, qmdt)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE
        s.u[1][i] += qmdt * gather_at(Enode, s.cell[1][i], s.offset[1][i], shape, stagger, g)
    end
end

"Deposit the charge of every species and solve for the field; the fields then belong to the current positions."
function fields!(st::ESState)
    (; config, grid) = st
    mode = config.mode
    for (s, ws) in zip(st.species, st.ws)
        effective_mode(mode, eltype(s)) isa Reference && sort_by_cell!(s, grid, ws)
    end
    fill!(st.ρ, zero(eltype(st.ρ)))
    for (s, ws) in zip(st.species, st.ws)
        deposit!(mode, st.ρ, s, config.shape, config.stagger, grid, ws)
    end
    fold_guards!(st.ρ, grid)
    backend = get_backend(st.ρ)
    gauss_kernel!(backend)(st.Ehalf, st.Enode, st.ρ, cellsize(grid), Int32(ncells(grid)),
                           Int32(guards(grid)); ndrange = 1)
    fill_guards!(st.Enode, grid)
    return st
end

"""
    initialize!(state, params)

Fields at t = 0 and the half-step-back kick `v ← v - (q/m)E dt/2`, so that the leapfrog starts
with velocities at t = -dt/2.
"""
function initialize!(st::ESState, p::ESParams)
    fields!(st)
    (; config, grid) = st
    for s in st.species
        T = eltype(s)
        kick_kernel!(get_backend(s))(s, st.Enode, config.shape, config.stagger, Int32(guards(grid)),
                                     -s.q / s.m * p.dt / 2; ndrange = capacity(s))
    end
    return st
end

"""
    step!(state, params, k)

One leapfrog step, the module's order: (sort if the mode needs it) deposit, fold, solve, fill;
record the diagnostics of step `k` (fields at t = (k-1)dt); kick and drift every species.
A function of the state and the parameters only, with every buffer in the state.
"""
function step!(st::ESState, p::ESParams, k::Integer)
    fields!(st)
    (; config, grid) = st
    backend = get_backend(st.ρ)
    T = eltype(st.ρ)
    dx = cellsize(grid)
    n = Int32(ncells(grid))
    x0 = (one(T) / 2 + ParticleMeshCore.stagger_offset(config.stagger, T)) * dx
    diagnostics_kernel!(backend)(st.W, st.Ere, st.Eim, st.Ehalf, dx, p.kmode, x0, n, Int32(k); ndrange = 1)
    for s in st.species
        push_kernel!(get_backend(s))(s, st.Enode, config.shape, config.stagger, Int32(guards(grid)),
                                     s.q / s.m * p.dt, p.dt / dx, n; ndrange = capacity(s))
    end
    return st
end

"""
    run!(state, params, nsteps; first = 1)

The plain loop: `step!` for `k = first:first+nsteps-1`. Call `initialize!` once before.
"""
function run!(st::ESState, p::ESParams, nsteps::Integer; first::Integer = 1)
    for k in first:(first + nsteps - 1)
        step!(st, p, k)
    end
    return st
end

field_energy_history(st::ESState) = Array(st.W)
mode_amplitude_history(st::ESState) = hypot.(Array(st.Ere), Array(st.Eim))

"Kinetic energy Σ m w |v|²/2 · dx over every species (host reduction, for diagnostics outside the step)."
function kinetic_energy(st::ESState)
    dx = cellsize(st.grid)
    sum(st.species) do s
        alive = Array(s.status) .== ALIVE
        v2 = Array(s.u[1]) .^ 2 .+ Array(s.u[2]) .^ 2 .+ Array(s.u[3]) .^ 2
        s.m * sum(Array(s.w)[alive] .* v2[alive]) * dx / 2
    end
end

"Least-squares slope and intercept of `y` against `x` (host analysis, Float64)."
function linear_fit(x, y)
    x̄, ȳ = sum(x) / length(x), sum(y) / length(y)
    slope = sum((x .- x̄) .* (y .- ȳ)) / sum((x .- x̄) .^ 2)
    return slope, ȳ - slope * x̄
end

"""
    fit_damping(t, A; ω_guess, tmax) -> (; γ, ω, tpeaks, Apeaks)

Damping rate and frequency of an oscillating, damped mode amplitude `A(t) ∝ |cos(ωt+φ)| e^{γt}`
(`γ < 0` for damping): the peaks of `A` are the maxima within ±0.4π/ω_guess, restricted to
`t ≤ tmax`; `γ` is the slope of log A over the peaks, `ω = π / (mean peak spacing)`. `ω_guess`
only sets the peak window.
"""
function fit_damping(t, A; ω_guess, tmax)
    dt = t[2] - t[1]
    h = max(1, round(Int, 0.4π / ω_guess / dt))
    idx = Int[]
    for k in (1 + h):(length(A) - h)
        t[k] <= tmax || break
        A[k] == maximum(@view A[(k - h):(k + h)]) && push!(idx, k)
    end
    length(idx) >= 2 || return (; γ = NaN, ω = NaN, tpeaks = t[idx], Apeaks = A[idx])
    γ, _ = linear_fit(Float64.(t[idx]), log.(Float64.(A[idx])))
    ω = π / ((t[idx[end]] - t[idx[1]]) / (length(idx) - 1))
    return (; γ, ω, tpeaks = t[idx], Apeaks = A[idx])
end

"""
    fit_growth(t, A; lo = 1/50, hi = 3/10) -> (; γ, window, tsat, Asat)

Growth rate of an exponentially growing amplitude. The saturation peak is the first sample that
reaches half the global maximum of `A` and is the largest within the following ±1 time unit
around it; the fit window is the contiguous run of samples before it with
`lo·Asat < A ≤ hi·Asat` (the linear phase, away from the initial transient and from
saturation); `γ` is the least-squares slope of log A over that window.
"""
function fit_growth(t, A; lo = 1 / 50, hi = 3 / 10)
    dt = t[2] - t[1]
    h = max(1, round(Int, 1 / dt))
    half = maximum(A) / 2
    ksat = findfirst(k -> A[k] >= half && A[k] == maximum(@view A[max(1, k - h):min(end, k + h)]), eachindex(A))
    Asat = A[ksat]
    start = ksat
    while start > 1 && A[start - 1] > lo * Asat
        start -= 1
    end
    sel = [k for k in start:ksat if A[k] <= hi * Asat]
    length(sel) >= 2 || return (; γ = NaN, window = (NaN, NaN), tsat = t[ksat], Asat)
    γ, _ = linear_fit(Float64.(t[sel]), log.(Float64.(A[sel])))
    return (; γ, window = (t[sel[1]], t[sel[end]]), tsat = t[ksat], Asat)
end

"""
    CosineDensity()

`n(x, p) = p[1]·(1 + p[2]·cos(p[3]·x))`: background density, relative amplitude and wavenumber.
"""
struct CosineDensity end
@inline (::CosineDensity)(x, p) = p[1] * (1 + p[2] * cos(p[3] * x))

"""
    landau_case(T; backend, n = 64, ppc = 1024, n0 = 1, vth = 1/20, kλD = 1/2, α = 1/20,
                dt_ωp = 1/20, tmax_ωp = 40, shape = Quadratic, mode = Reference(), seed = 1)

Landau damping of the fundamental mode of a box of one wavelength, `k λ_D = kλD`, with electrons
of density `n0·(1 + α cos kx)` (perturbation through the weights, A3), Maxwellian velocities of
thermal speed `vth` (in c) from Philox draws, and an immobile neutralizing background. Returns
`(state, params, info)`; `info` holds the derived quantities (ω_p, λ_D, k, dt, steps).
"""
function landau_case(::Type{T}; backend = CPU(), n = 64, ppc = 1024, n0 = 1, vth = 1 // 20,
                     kλD = 1 // 2, α = 1 // 20, dt_ωp = 1 // 20, tmax_ωp = 40, shape = Quadratic,
                     mode = Reference(), seed = 1) where {T}
    ωp = sqrt(T(n0))
    λD = T(vth) / ωp
    k = T(kλD) / λD
    L = 2 * T(π) / k
    grid = PeriodicGrid(n, L / n, guard_width(shape))
    dt = T(dt_ωp) / ωp
    nsteps = round(Int, tmax_ωp / dt_ωp)
    e = Species(backend, T, n * ppc; q = -1, m = 1)
    vth3 = (T(vth), T(vth), T(vth))
    load_quiet!(e, grid, CosineDensity(), (T(n0), T(α), k), ppc; vth = vth3, seed, stream = 1)
    st = ESState(ESConfig(shape, mode), grid, (e,), nsteps)
    info = (; ωp, λD, k, L, dt, nsteps, n, ppc, vth = T(vth), kλD = T(kλD), α = T(α), n0 = T(n0))
    return st, ESParams(dt, k), info
end

"""
    two_stream_case(T; backend, n = 64, ppc = 512, n0 = 1, v0 = 1/10, kv0_ωp = 3/5, α = 1/1000,
                    vth = 0, dt_ωp = 1/20, tmax_ωp = 40, shape = Quadratic, mode = Reference(), seed = 1)

Two counter-streaming electron beams at `±v0`, each of density `n0/2·(1 + α cos kx)`, in a box
of one wavelength with `k v0 / ω_p = kv0_ωp`, cold by default, immobile neutralizing background.
"""
function two_stream_case(::Type{T}; backend = CPU(), n = 64, ppc = 512, n0 = 1, v0 = 1 // 10,
                         kv0_ωp = 3 // 5, α = 1 // 1000, vth = 0, dt_ωp = 1 // 20, tmax_ωp = 40,
                         shape = Quadratic, mode = Reference(), seed = 1) where {T}
    ωp = sqrt(T(n0))
    k = T(kv0_ωp) * ωp / T(v0)
    L = 2 * T(π) / k
    grid = PeriodicGrid(n, L / n, guard_width(shape))
    dt = T(dt_ωp) / ωp
    nsteps = round(Int, tmax_ωp / dt_ωp)
    vth3 = (T(vth), T(vth), T(vth))
    p = (T(n0) / 2, T(α), k)
    b1 = Species(backend, T, n * ppc; q = -1, m = 1)
    b2 = Species(backend, T, n * ppc; q = -1, m = 1)
    load_quiet!(b1, grid, CosineDensity(), p, ppc; vdrift = (T(v0), zero(T), zero(T)), vth = vth3, seed, stream = 1)
    load_quiet!(b2, grid, CosineDensity(), p, ppc; vdrift = (-T(v0), zero(T), zero(T)), vth = vth3, seed, stream = 2)
    st = ESState(ESConfig(shape, mode), grid, (b1, b2), nsteps)
    info = (; ωp, k, L, dt, nsteps, n, ppc, v0 = T(v0), kv0_ωp = T(kv0_ωp), α = T(α), n0 = T(n0), vth = T(vth))
    return st, ESParams(dt, k), info
end

end
