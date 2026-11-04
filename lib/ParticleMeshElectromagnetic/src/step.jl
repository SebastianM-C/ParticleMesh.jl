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
# and move the particle by Δt·uₓ/γ (less than one cell below the Courant limit, as deposit_flux! requires).
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
    check_subluminal(s, d::SpeciesDescription) -> s

Refuse a load of species `d` into `s` in which a drawn speed reaches c, before the velocities
become momenta (γ = 1/√(1 − v²) is not finite there; #257). The loader draws each velocity
component as `drift + thermal·z` with `z` standard normal, in velocity, so the Gaussian tail
passes c with a probability that grows with the thermal speed. One device reduction, the
largest |v|² over the alive slots, then one host comparison of that scalar (§0.4); an
`ArgumentError` names the species, the largest drawn |v| and the cause. A NaN velocity is
refused too. Called by `initialize!` at N = 1 and 2, never inside `step!`.
"""
function check_subluminal(s, d)
    T = eltype(s.w)
    v2 = mapreduce((a, b, c, st) -> ifelse(st == ALIVE, a * a + b * b + c * c, zero(T)), max, s.u[1], s.u[2], s.u[3],
                   s.status; init = zero(T))
    v2 < one(T) ||
        throw(ArgumentError("species $(d.name): a loaded speed reaches c, the largest drawn |v| is $(sqrt(v2)) (in c); " *
                            "the thermal spread is drawn in velocity, drift + thermal·z with z standard normal, " *
                            "so its Gaussian tail can pass c: lower the thermal speed or the drift"))
    return s
end

"""
    EMState{N}

Everything a step reads and writes, allocated once: the numerics, the grid (`PeriodicGrid{N}`),
the species and their workspaces (tuples), the fields of [`field_layout`](@ref)`(Val(N))` (with
guards), the scratch field `ρ` (the charge for the initial field, then the scratch of the
current filter), the kinetic energy buffer `K`,
the diagnostic histories, one entry per step (see [`histories`](@ref)), and `aux`, what only
one dimension needs (empty at N = 1; at N = 2 the spectral workspace and potential of the
initial solve, the diagnostics' phase tables and partial sums, see [`EMState`](@ref)'s 2D
constructor). At N = 1 `K` is the kinetic energy density, a field; at N = 2 it holds the kinetic
energy of fixed chunks of particle slots, in order (its size is fixed by the capacities).
"""
struct EMState{N,NU<:EMNumerics,G<:PeriodicGrid{N},SP<:Tuple,WS<:Tuple,F<:NamedTuple,A,KB,H<:NamedTuple,X<:NamedTuple}
    numerics::NU
    grid::G
    species::SP
    ws::WS
    fields::F
    ρ::A
    K::KB
    hist::H
    aux::X
end

Base.ndims(::EMState{N}) where {N} = N

const HISTORY_KEYS = (:WEx, :WEy, :WEz, :WBx, :WBy, :WBz, :K, :Ex_re, :Ex_im, :Ey_re, :Ey_im, :Bz_re, :Bz_im)

"""
    EMState(problem, numerics; backend = CPU())

Allocate the state of `problem` (an [`EMProblem`](@ref)) for `numerics`: the grid with
[`em_guard_width`](@ref) guards, one species per [`SpeciesDescription`](@ref) at capacity
`cells·ppc` with the extras `(old_cell = Int32, old_offset = T)` that `deposit_flux!` reads,
the workspaces, the fields and the histories for `problem.nsteps` steps. The particles are
loaded by [`initialize!`](@ref). Throws an `ArgumentError` naming the limit unless
`problem.dt < courant_limit(numerics.solver, grid)` ([`check_courant`](@ref)): the solver is then
stable and every particle moves less than one cell per step, as `deposit_flux!` needs.
"""
function EMState(problem::EMProblem{1,T}, numerics::EMNumerics; backend = CPU()) where {T}
    numerics.current isa Esirkepov ||
        throw(ArgumentError("the 1D step deposits the charge-conserving current: EMNumerics needs current = Esirkepov() at N = 1, got $(numerics.current)"))
    numerics.absorber === nothing || throw(ArgumentError("the 1D box is periodic: EMNumerics takes no absorber at N = 1"))
    n = problem.cells
    grid = PeriodicGrid(n, problem.length / n, em_guard_width(numerics.shape))
    check_courant(numerics.solver, grid, problem.dt)
    species = map(problem.species) do d
        Species(backend, T, n * d.ppc; q = d.charge, m = d.mass, extras = (old_cell = Int32, old_offset = T))
    end
    ws = map(s -> Workspace(s, grid), species)
    hist = NamedTuple{HISTORY_KEYS}(ntuple(_ -> KernelAbstractions.zeros(backend, T, problem.nsteps), length(HISTORY_KEYS)))
    return EMState(numerics, grid, species, ws, em_fields(backend, T, grid), field(backend, T, grid),
                   field(backend, T, grid), hist, NamedTuple())
end

# The mode the depositions of `s` run in: the step sorts when it is `Reference()` and passes it,
# so `Fast()` on a type without atomics does not sort a second time inside the deposition.
deposit_mode(st::EMState, s) = effective_mode(st.numerics.mode, eltype(s))
sorts(st::EMState, s) = deposit_mode(st, s) isa Reference

"Deposit the kinetic energy density at the current momenta into `st.K` (positions sorted if the mode needs it)."
function kinetic!(st::EMState{1})
    (; numerics, grid) = st
    fill!(st.K, zero(eltype(st.K)))
    for (s, ws) in zip(st.species, st.ws)
        deposit!(deposit_mode(st, s), st.K, s, numerics.shape, RHO_STAGGER, grid, ws; quantity = KineticEnergy(),
                 workgroupsize = numerics.workgroupsize)
    end
    return st
end

# The uniform background fields of the problem, added to every entry (guards included, so they
# stay filled) of each component whose background is nonzero; a zero background leaves the
# component's bits as they are (−0.0 + 0.0 would be +0.0). The test is on host values.
function add_background!(f, E0, B0)
    for (a, v) in ((f.Ex, E0[1]), (f.Ey, E0[2]), (f.Ez, E0[3]), (f.Bx, B0[1]), (f.By, B0[2]), (f.Bz, B0[3]))
        iszero(v) || (a .+= v)
    end
    return f
end

"The values of `initialize!`'s `momenta_at`; anything else is an `ArgumentError` (a host keyword, not device data, §0.4)."
check_momenta_at(m::Symbol) = m in (:t0, :t_minus_half) ? m :
    throw(ArgumentError("momenta_at must be :t0 or :t_minus_half, got :$m"))

"""
    initialize!(state, problem; momenta_at = :t0) -> state

The state at t = 0 from the problem description: each species loaded by the engine's quiet
start (`load_quiet!`, weights `density·(1 + amplitude·cos(k·x))/ppc`, so the parameters enter
through the weights, A3; velocities `drift + thermal·z` from Philox draws keyed by the problem's
seed and the species' stream), the velocities converted to momenta `u = γv` (a load in which a
drawn speed reaches c is an `ArgumentError` naming the species, `check_subluminal`); the fields zero; Ex from the charge, as Smilei solves at t = 0: `cumsum_periodic!` of the deposited
ρ, so that (Ex[j] − Ex[j−1])/Δx = ρ[j] − mean(ρ) to roundoff, the mean being the immobile
neutralizing background; the background fields `E0` and `B0` of the problem added where they
are nonzero (so a problem without them keeps the bits of the fields); then the momenta at
t = −Δt/2 (half-step centring, A2), and the kinetic energy density there.

`momenta_at` says at which time the loaded momenta are taken:

- `:t0` (the default): the loaded momenta are `u⁰`, and `initialize!` moves them half a step
  back, `uⁿ⁻¹ᐟ²` from `u⁰` by the pusher with −Δt/2 in E⁰, B⁰;
- `:t_minus_half`: the loaded momenta are already `u⁻¹ᐟ²` and are kept as loaded, with no kick.
  This is the convention of Smilei and PIConGPU, which the 1D cross-code runs use; the two
  starts differ by O(Δt·E⁰) in the momenta.

Positions, weights and fields are the same in both; anything else is an `ArgumentError`.
"""
function initialize!(st::EMState{1}, problem::EMProblem{1}; momenta_at::Symbol = :t0)
    check_momenta_at(momenta_at)
    (; numerics, grid) = st
    T = eltype(st.ρ)
    wgs = numerics.workgroupsize
    for (s, d) in zip(st.species, problem.species)
        fill!(s.status, ParticleMeshCore.FREE)
        k = 2 * T(π) * d.mode / problem.length
        load_quiet!(s, grid, CosineProfile(), (d.density, d.amplitude, k), d.ppc; vdrift = d.drift, vth = d.thermal,
                    seed = problem.seed, stream = d.stream, workgroupsize = wgs)
        check_subluminal(s, d)
        launcher(velocity_to_momentum_kernel!, get_backend(s), wgs)(s; ndrange = capacity(s))
    end
    foreach(a -> fill!(a, zero(T)), st.fields)
    for (s, ws) in zip(st.species, st.ws)
        sorts(st, s) && sort_by_cell!(s, grid, ws; workgroupsize = wgs)
    end
    fill!(st.ρ, zero(T))
    for (s, ws) in zip(st.species, st.ws)
        deposit!(deposit_mode(st, s), st.ρ, s, numerics.shape, RHO_STAGGER, grid, ws; workgroupsize = wgs)
    end
    fold_guards!(st.ρ, grid; workgroupsize = wgs)
    cumsum_periodic!(st.fields.Ex, st.ρ, grid; workgroupsize = wgs)
    add_background!(st.fields, problem.E0, problem.B0)
    if momenta_at === :t0
        g = Int32(guards(grid))
        for s in st.species
            launcher(kick_kernel!, get_backend(s), wgs)(s, st.fields, numerics.pusher, numerics.shape, g,
                                                        -s.q / s.m * problem.dt / 4; ndrange = capacity(s))
        end
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
   for Jx from the old and new positions, `q·w·u_{y,z}/γ` for Jy and Jz), fold the guards, then
   apply the numerics' [`BinomialFilter`](@ref) (nothing with zero passes; the scratch is `ρ`);
5. deposit the kinetic energy density at n + 1/2 (for the next step's diagnostics);
6. the mesh solve of the numerics (Yee: B half step, E full step with J, B half step).

`k` must lie within the `nsteps` the state was allocated for (an `ArgumentError` otherwise: the
diagnostics write entry `k`). Gauss's law then holds to roundoff after every step, from
continuity (T-212); with filter passes its filtered form does (T-441). A function of the
state and the parameters only, every buffer in the state, no host-side branch on device data
(§0.4).
"""
function step!(st::EMState{1}, p::EMParams, k::Integer)
    (; numerics, grid, fields) = st
    # k is a host integer (not device data): the diagnostics kernel writes entry k of the
    # histories under @inbounds, so a k past their length must not reach it.
    1 <= k <= length(st.hist.K) || throw(ArgumentError("step $k outside the $(length(st.hist.K)) steps the state was allocated for"))
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
        ParticleMeshCore.deposit_flux!(deposit_mode(st, s), fields.Jx, fields.Jy, fields.Jz, s, numerics.shape, RHO_STAGGER,
                                       grid, ws; dt = p.dt, quantity_y = CurrentY(), quantity_z = CurrentZ(),
                                       workgroupsize = wgs)
    end
    fold_guards!(fields.Jx, grid; workgroupsize = wgs)
    fold_guards!(fields.Jy, grid; workgroupsize = wgs)
    fold_guards!(fields.Jz, grid; workgroupsize = wgs)
    filter_current!(numerics.filter, fields, grid, st.ρ; workgroupsize = wgs)
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
