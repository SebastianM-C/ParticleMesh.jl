# The module's step: composed here from engine pieces, in the module's own order. The engine owns
# no System, State or step! (architecture-draft.md §0.7); this file is where the electromagnetic
# step order lives.

"The current density's y component per particle, `q·w·u_y/γ` (a `deposit_flux!` quantity)."
struct CurrentY end
"The current density's z component per particle, `q·w·u_z/γ`."
struct CurrentZ end
"The kinetic energy density per particle, `m·w·(γ − 1)`, written `m·w·|u|²/(γ + 1)` (no cancellation for small u)."
struct KineticEnergy end

@inline particle_gamma(s, i) = @inbounds lorentz_factor(s.u[1][i], s.u[2][i], s.u[3][i])
@inline (::CurrentY)(s, i) = @inbounds s.q * s.w[i] * s.u[2][i] / particle_gamma(s, i)
@inline (::CurrentZ)(s, i) = @inbounds s.q * s.w[i] * s.u[3][i] / particle_gamma(s, i)
@inline function (::KineticEnergy)(s, i)
    @inbounds begin
        u1, u2, u3 = s.u[1][i], s.u[2][i], s.u[3][i]
        u² = u1 * u1 + u2 * u2 + u3 * u3
        return s.m * s.w[i] * u² / (lorentz_factor(u1, u2, u3) + 1)
    end
end

"E and B of the layout at offset `δ` of cell `c` (guards filled), each on its own stagger."
@inline function fields_at(f, c::Int32, δ::T, shape, g::Int32) where {T}
    E = (gather_at(f.Ex, c, δ, shape, FIELD_LAYOUT.Ex, g),
         gather_at(f.Ey, c, δ, shape, FIELD_LAYOUT.Ey, g),
         gather_at(f.Ez, c, δ, shape, FIELD_LAYOUT.Ez, g))
    B = (gather_at(f.Bx, c, δ, shape, FIELD_LAYOUT.Bx, g),
         gather_at(f.By, c, δ, shape, FIELD_LAYOUT.By, g),
         gather_at(f.Bz, c, δ, shape, FIELD_LAYOUT.Bz, g))
    return E, B
end

# Store the old position (the extras deposit_flux! reads), push the momentum in Eⁿ, Bⁿ at xⁿ,
# and move the particle by Δt·uₓ/γ (less than one cell for Δt < Δx, as deposit_flux! requires).
@kernel function push_kernel!(s, f, pusher, shape, g::Int32, qm_h, dt_dx, n::Int32)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE
        c = s.cell[1][i]
        δ = s.offset[1][i]
        s.extras.old_cell[i] = c
        s.extras.old_offset[i] = δ
        E, B = fields_at(f, c, δ, shape, g)
        u = push_momentum(pusher, (s.u[1][i], s.u[2][i], s.u[3][i]), E, B, qm_h)
        s.u[1][i] = u[1]
        s.u[2][i] = u[2]
        s.u[3][i] = u[3]
        c′, δ′ = advance(c, δ, u[1] / lorentz_factor(u) * dt_dx, n)
        s.cell[1][i] = c′
        s.offset[1][i] = δ′
    end
end

# The momentum update alone (no move): the half step back of initialize!, with qm_h = −(q/m)Δt/4.
@kernel function kick_kernel!(s, f, pusher, shape, g::Int32, qm_h)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE
        E, B = fields_at(f, s.cell[1][i], s.offset[1][i], shape, g)
        u = push_momentum(pusher, (s.u[1][i], s.u[2][i], s.u[3][i]), E, B, qm_h)
        s.u[1][i] = u[1]
        s.u[2][i] = u[2]
        s.u[3][i] = u[3]
    end
end

# The loader draws velocities; the module's u is the momentum γv (per unit mass, in c).
@kernel function velocity_to_momentum_kernel!(s)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE
        T = eltype(s.w)
        v1, v2, v3 = s.u[1][i], s.u[2][i], s.u[3][i]
        γ = one(T) / sqrt(one(T) - (v1 * v1 + v2 * v2 + v3 * v3))
        s.u[1][i] = γ * v1
        s.u[2][i] = γ * v2
        s.u[3][i] = γ * v3
    end
end

"""
    EMState

Everything a step reads and writes, allocated once: the numerics, the grid, the species and
their workspaces (tuples), the fields of [`FIELD_LAYOUT`](@ref) (with guards), the scratch
fields `ρ` (the charge for the initial field) and `K` (the kinetic energy density), and the
diagnostic histories, one entry per step (see [`histories`](@ref)).
"""
struct EMState{NU<:EMNumerics,G,SP<:Tuple,WS<:Tuple,F<:NamedTuple,A,H<:NamedTuple}
    numerics::NU
    grid::G
    species::SP
    ws::WS
    fields::F
    ρ::A
    K::A
    hist::H
end

const HISTORY_KEYS = (:WEx, :WEy, :WEz, :WBx, :WBy, :WBz, :K, :Ex_re, :Ex_im, :Ey_re, :Ey_im, :Bz_re, :Bz_im)

"""
    EMState(problem, numerics; backend = CPU())

Allocate the state of `problem` (an [`EMProblem`](@ref)) for `numerics`: the grid with
[`em_guard_width`](@ref) guards, one species per [`SpeciesDescription`](@ref) at capacity
`cells·ppc` with the extras `(old_cell = Int32, old_offset = T)` that `deposit_flux!` reads,
the workspaces, the fields and the histories for `problem.nsteps` steps. The particles are
loaded by [`initialize!`](@ref).
"""
function EMState(problem::EMProblem{T}, numerics::EMNumerics; backend = CPU()) where {T}
    n = problem.cells
    grid = PeriodicGrid(n, problem.length / n, em_guard_width(numerics.shape))
    species = map(problem.species) do d
        Species(backend, T, n * d.ppc; q = d.charge, m = d.mass, extras = (old_cell = Int32, old_offset = T))
    end
    ws = map(s -> Workspace(s, grid), species)
    hist = NamedTuple{HISTORY_KEYS}(ntuple(_ -> KernelAbstractions.zeros(backend, T, problem.nsteps), length(HISTORY_KEYS)))
    return EMState(numerics, grid, species, ws, em_fields(backend, T, grid), field(backend, T, grid),
                   field(backend, T, grid), hist)
end

sorts(st::EMState, s) = effective_mode(st.numerics.mode, eltype(s)) isa Reference

"Deposit the kinetic energy density at the current momenta into `st.K` (positions sorted if the mode needs it)."
function kinetic!(st::EMState)
    (; numerics, grid) = st
    fill!(st.K, zero(eltype(st.K)))
    for (s, ws) in zip(st.species, st.ws)
        deposit!(numerics.mode, st.K, s, numerics.shape, RHO_STAGGER, grid, ws; quantity = KineticEnergy(),
                 workgroupsize = numerics.workgroupsize)
    end
    return st
end

"""
    initialize!(state, problem) -> state

The state at t = 0 from the problem description: each species loaded by the engine's quiet
start (`load_quiet!`, weights `density·(1 + amplitude·cos(k·x))/ppc`, so the parameters enter
through the weights, A3; velocities `drift + thermal·z` from Philox draws keyed by the problem's
seed and the species' stream), the velocities converted to momenta `u = γv`; the transverse
fields zero; Ex from the charge, as Smilei solves at t = 0: `cumsum_periodic!` of the deposited
ρ, so that (Ex[j] − Ex[j−1])/Δx = ρ[j] − mean(ρ) to roundoff, the mean being the immobile
neutralizing background; then the momenta half a step back, `uⁿ⁻¹ᐟ²` from `u⁰` by the pusher
with −Δt/2 in E⁰, B⁰ (half-step centring, A2), and the kinetic energy density at t = −Δt/2.
"""
function initialize!(st::EMState, problem::EMProblem)
    (; numerics, grid) = st
    T = eltype(st.ρ)
    wgs = numerics.workgroupsize
    for (s, d) in zip(st.species, problem.species)
        fill!(s.status, ParticleMeshCore.FREE)
        k = 2 * T(π) * d.mode / problem.length
        load_quiet!(s, grid, CosineProfile(), (d.density, d.amplitude, k), d.ppc; vdrift = d.drift, vth = d.thermal,
                    seed = problem.seed, stream = d.stream, workgroupsize = wgs)
        launcher(velocity_to_momentum_kernel!, get_backend(s), wgs)(s; ndrange = capacity(s))
    end
    foreach(a -> fill!(a, zero(T)), st.fields)
    for (s, ws) in zip(st.species, st.ws)
        sorts(st, s) && sort_by_cell!(s, grid, ws; workgroupsize = wgs)
    end
    fill!(st.ρ, zero(T))
    for (s, ws) in zip(st.species, st.ws)
        deposit!(numerics.mode, st.ρ, s, numerics.shape, RHO_STAGGER, grid, ws; workgroupsize = wgs)
    end
    fold_guards!(st.ρ, grid; workgroupsize = wgs)
    cumsum_periodic!(st.fields.Ex, st.ρ, grid; workgroupsize = wgs)
    g = Int32(guards(grid))
    for s in st.species
        launcher(kick_kernel!, get_backend(s), wgs)(s, st.fields, numerics.pusher, numerics.shape, g,
                                                    -s.q / s.m * problem.dt / 4; ndrange = capacity(s))
    end
    kinetic!(st)
    return st
end

"""
    step!(state, params, k) -> state

Step `k` (from t = (k−1)Δt to kΔt), the module's step order:

1. diagnostics of step `k`: the field energies and mode amplitudes of Eⁿ, Bⁿ, and the kinetic
   energy at n − 1/2 (deposited at the end of the previous step);
2. per species: store the old cell and offset as extras, push the momentum in Eⁿ, Bⁿ gathered
   at xⁿ (the pusher of the numerics) and move the particle, xⁿ⁺¹ = xⁿ + Δt·uₓ/γ;
3. sort by cell where the deposition mode needs it (the extras follow the particles);
4. zero Jx, Jy, Jz and add every species' current with the engine's `deposit_flux!` (Esirkepov
   for Jx from the old and new positions, `q·w·u_{y,z}/γ` for Jy and Jz), then fold the guards;
5. deposit the kinetic energy density at n + 1/2 (for the next step's diagnostics);
6. the mesh solve of the numerics (Yee: B half step, E full step with J, B half step).

Gauss's law then holds to roundoff after every step, from continuity (T-212). A function of the
state and the parameters only, every buffer in the state, no host-side branch on device data
(§0.4).
"""
function step!(st::EMState, p::EMParams, k::Integer)
    (; numerics, grid, fields) = st
    T = eltype(st.ρ)
    wgs = numerics.workgroupsize
    n = Int32(ncells(grid))
    g = Int32(guards(grid))
    dx = cellsize(grid)
    diagnostics_kernel!(get_backend(st.ρ))(st.hist, fields, st.K, dx, p.kmode, n, g, Int32(length(st.K)), Int32(k);
                                           ndrange = 1)
    for s in st.species
        launcher(push_kernel!, get_backend(s), wgs)(s, fields, numerics.pusher, numerics.shape, g, s.q / s.m * p.dt / 2,
                                                    p.dt / dx, n; ndrange = capacity(s))
    end
    for (s, ws) in zip(st.species, st.ws)
        sorts(st, s) && sort_by_cell!(s, grid, ws; workgroupsize = wgs)
    end
    fill!(fields.Jx, zero(T))
    fill!(fields.Jy, zero(T))
    fill!(fields.Jz, zero(T))
    for (s, ws) in zip(st.species, st.ws)
        ParticleMeshCore.deposit_flux!(numerics.mode, fields.Jx, fields.Jy, fields.Jz, s, numerics.shape, RHO_STAGGER,
                                       grid, ws; dt = p.dt, quantity_y = CurrentY(), quantity_z = CurrentZ(),
                                       workgroupsize = wgs)
    end
    fold_guards!(fields.Jx, grid; workgroupsize = wgs)
    fold_guards!(fields.Jy, grid; workgroupsize = wgs)
    fold_guards!(fields.Jz, grid; workgroupsize = wgs)
    kinetic!(st)
    solve_fields!(numerics.solver, fields, grid, p.dt; workgroupsize = wgs)
    return st
end

"""
    run!(state, params, nsteps; first = 1)

The plain loop: `step!` for `k = first:first+nsteps-1`. Call [`initialize!`](@ref) once before.
"""
function run!(st::EMState, p::EMParams, nsteps::Integer; first::Integer = 1)
    for k in first:(first + nsteps - 1)
        step!(st, p, k)
    end
    return st
end
