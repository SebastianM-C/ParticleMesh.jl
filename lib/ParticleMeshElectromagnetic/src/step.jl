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
    check_subluminal(s, species) -> s

Refuse a load of `species` (anything with a `name`: a description, or a state's setup of it) into `s` in which a drawn speed reaches c, before the velocities
become momenta (γ = 1/√(1 − v²) is not finite there; #257). The loader draws each velocity
component as `drift + thermal·z` with `z` standard normal, in velocity, so the Gaussian tail
passes c with a probability that grows with the thermal speed. One device reduction, the
largest |v|² over the alive slots, then one host comparison of that scalar (§0.4); an
`ArgumentError` names the species, the largest drawn |v| and the cause. A NaN velocity is
refused too. Called by `initialize!` at N = 1 and 2, never inside `step!`.
"""
function check_subluminal(s, d)
    T = eltype(s.w)
    z = zero(T)   # a value, not the type, in the closure: a Type field is not isbits on the device
    v2 = mapreduce((a, b, c, st) -> ifelse(st == ALIVE, a * a + b * b + c * c, z), max, s.u[1], s.u[2], s.u[3],
                   s.status; init = zero(T))
    v2 < one(T) ||
        throw(ArgumentError("species $(d.name): a loaded speed reaches c, the largest drawn |v| is $(sqrt(v2)) (in c); " *
                            "the thermal spread is drawn in velocity, drift + thermal·z with z standard normal, " *
                            "so its Gaussian tail can pass c: lower the thermal speed or the drift"))
    return s
end

"""
    load_species!(s, grid, ps::SpeciesParams, setup, seed; density_at, workgroupsize) -> s

The quiet start of a species into `s`, at N = 1 and 2, from its parameters `ps` (the
[`CosineProfile`](@ref)'s density, amplitude and wave vector through the weights, A3; the drift
and the thermal speeds through the momenta) and the state's `setup` of it (particles per cell,
Philox stream, kind of momentum distribution): `load_quiet!` with the species'
[`momentum_distribution`](@ref), keyed by `seed` and the stream. A Gaussian-velocity load is
checked ([`check_subluminal`](@ref)) and its velocities converted to `u = γv`; the two momentum
draws write `u` directly and need neither. A Maxwell–Jüttner load counts the particles whose
rejection tries all ran out (they keep their last, finite proposal), one device counter read once
on the host, and warns when there are any; the probability is near 1e−22 per particle. Called by
`initialize!`, never inside `step!`.
"""
function load_species!(s, grid, ps::SpeciesParams, setup, seed; density_at, workgroupsize)
    dist = momentum_distribution(setup.momentum, setup.name, ps.drift, ps.thermal)
    exhausted = dist isa MaxwellJuttner ? KernelAbstractions.zeros(get_backend(s), Int32, 1) : nothing
    load_quiet!(s, grid, CosineProfile(), density_params(ps), setup.ppc, dist; seed, stream = setup.stream, density_at, exhausted,
                workgroupsize)
    if !draws_momentum(dist)
        check_subluminal(s, setup)
        launcher(velocity_to_momentum_kernel!, get_backend(s), workgroupsize)(s; ndrange = capacity(s))
    end
    if exhausted !== nothing
        n = Int(only(Array(exhausted)))
        n > 0 && @warn "species $(setup.name): $n Maxwell–Jüttner draws ran out of rejection tries and keep their last proposal"
    end
    return s
end

"""
    EMState{N}

What a run evolves, with the cache, the scratch and the configuration a step needs next to it,
allocated once from a problem and its numerics; no parameter of [`EMParams`](@ref) is kept in it.

- Evolved by `initialize!` and `step!`, and nothing else is ([`evolved`](@ref) is the list, as a
  tuple of arrays): the `species` (the engine's `Species`, a tuple; of each the cells, offsets,
  momenta, weights, status and count), the `fields` of [`field_layout`](@ref)`(Val(N))` with
  guards, and at N = 2 the convolution fields of a PML (in `modifiers`) and in `aux` the
  window's counters `lost` and `fast` and the witness bunch's particles. With the
  step index and the parameters they are a complete restart ([`snapshot`](@ref),
  [`restore!`](@ref)).
- The cache: the kinetic energy buffer `K` that a step leaves for the next observation (at N = 1
  the kinetic energy density, a field; at N = 2 the kinetic energy of fixed chunks of particle
  slots, in order), a function of the evolved momenta, weights and status, rebuilt by
  `refresh_caches!` where the evolved arrays are written from outside.
- Scratch, written before it is read within a call: the species' workspaces `ws` and extras (the
  old position the push stores), the field `ρ` (the charge for the initial field, then the scratch
  of the current filter and of the window's shift), and at N = 2 in `aux` the spectral workspace
  and the potential `φ` of the initial solve and the refill's `flags`, and the PMLs' shift scratch.
- Configuration: the `numerics`, the `grid` (`PeriodicGrid{N}`), the time step `dt` (to time what
  the cell size is to space: the Courant check, the absorbers' tables, the antenna's half-cell
  delay, the window's shift count and the witness's injection step are made for it), the `setup`
  of each species (name, particles per cell, Philox stream, kind of momentum distribution), each
  species' charge and mass, and at N = 2 in `aux` the [`WindowFrame`](@ref) and the
  [`WitnessFrame`](@ref) (`nothing` where the problem has none). `aux` is empty at N = 1.
- `modifiers`: the tuple of [`FieldModifier`](@ref)s that the mesh solve walks
  ([`solve_fields!`](@ref)), in the order [`field_modifiers`](@ref) gives them: the placed
  absorber of an open axis (a PML's coefficient tables and convolution fields, or an absorbing
  layer's factors), the PML of a moving window's back edge, the antenna planes
  ([`AntennaPlanes`](@ref)) and the window's x edges. Each is held here and nowhere else in the
  state (`open_absorber`, `back_absorber` and `antenna_planes` find them). Empty for a periodic
  box without lasers, and at N = 1.
- `recorder`: the state's default observer, a [`HistoryRecorder`](@ref) for the problem's
  diagnostic `kmode` and `nsteps`. A step does not read it; `step!` hands it to
  [`observe!`](@ref) unless another observer is given.
"""
struct EMState{N,T,NU<:EMNumerics,G<:PeriodicGrid{N},SP<:Tuple,WS<:Tuple,F<:NamedTuple,A,KB,R,SU<:Tuple,MD<:Tuple{Vararg{FieldModifier}},X<:NamedTuple}
    numerics::NU
    grid::G
    dt::T
    species::SP
    ws::WS
    fields::F
    ρ::A
    K::KB
    recorder::R
    setup::SU
    modifiers::MD
    aux::X
end

"The placed absorber of a state's open axis (a [`LayerFactors`](@ref) or a [`PMLFields`](@ref)), or `nothing`."
open_absorber(st::EMState) = pick_modifier(Union{LayerFactors,PMLFields}, st.modifiers)
"The placed PML of a moving window's back edge (a [`BackPMLFields`](@ref)), or `nothing`."
back_absorber(st::EMState) = pick_modifier(BackPMLFields, st.modifiers)
"The antenna planes of a state, one [`AntennaPlane`](@ref) per laser; `()` without lasers."
antenna_planes(st::EMState) = antenna_planes(pick_modifier(AntennaPlanes, st.modifiers))
antenna_planes(::Nothing) = ()

"""
    with_modifiers(state, modifiers::Tuple) -> EMState

`state` with another tuple of [`FieldModifier`](@ref)s, which its steps then walk, and everything
else the same arrays (no copy: the two states are one run, use one of them). How a modifier that
the state's constructor does not know joins a state, at the place in the tuple its order in the
solve asks for: `with_modifiers(state, (state.modifiers..., m))`.
"""
with_modifiers(st::EMState, modifiers::Tuple) =
    EMState(st.numerics, st.grid, st.dt, st.species, st.ws, st.fields, st.ρ, st.K, st.recorder, st.setup, modifiers, st.aux)

Base.ndims(::EMState{N}) where {N} = N
Base.eltype(::EMState{N,T}) where {N,T} = T

"""
    EMState(problem, numerics; backend = CPU())

Allocate the state of `problem` (an [`EMProblem`](@ref)) for `numerics`: the grid with
[`em_guard_width`](@ref) guards, one species per [`SpeciesDescription`](@ref) at capacity
`cells·ppc` with the extras `(old_cell = Int32, old_offset = T)` that `deposit_flux!` reads,
the workspaces, the fields, and the default [`HistoryRecorder`](@ref) for `problem.kmode` and
`problem.nsteps` steps. The particles are loaded by [`initialize!`](@ref). Throws an
`ArgumentError` naming the limit unless `problem.dt < courant_limit(numerics.solver, grid)`
([`check_courant`](@ref)): the solver is then stable and every particle moves less than one cell
per step, as `deposit_flux!` needs.
"""
function EMState(problem::EMProblem{1,T}, numerics::EMNumerics; backend = CPU()) where {T}
    numerics.current isa Esirkepov ||
        throw(ArgumentError("the 1D step deposits the charge-conserving current: EMNumerics needs current = Esirkepov() at N = 1, got $(numerics.current)"))
    numerics.absorber === nothing || throw(ArgumentError("the 1D box is periodic: EMNumerics takes no absorber at N = 1"))
    numerics.back_absorber === nothing ||
        throw(ArgumentError("the 1D box is periodic: EMNumerics takes no back_absorber at N = 1 (the moving window is 2D)"))
    n = problem.cells
    grid = PeriodicGrid(n, problem.length / n, em_guard_width(numerics.shape))
    check_courant(numerics.solver, grid, problem.dt)
    species = map(problem.species) do d
        Species(backend, T, n * d.ppc; q = d.charge, m = d.mass, extras = (old_cell = Int32, old_offset = T))
    end
    ws = map(s -> Workspace(s, grid), species)
    recorder = HistoryRecorder(backend, T, grid, problem.length, problem.kmode, problem.nsteps; workgroupsize = numerics.workgroupsize)
    return EMState(numerics, grid, problem.dt, species, ws, em_fields(backend, T, grid), field(backend, T, grid),
                   field(backend, T, grid), recorder, map(species_setup, problem.species), (), NamedTuple())
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
    check_params(state, params) -> params

Whether `params` fit `state` (an `ArgumentError` naming the misfit otherwise): the same element
type, one [`SpeciesParams`](@ref) per species, and at N = 2 one laser per antenna plane, each
within half a cell of its plane's column (the column is fixed when the state is built), with a
moving window every pulse over before the window passes its plane, a witness's parameters exactly
when the state has a bunch, and a profile only with a window; and what an [`EMProblem`](@ref)
refuses of the same numbers: a thermal spread that is not isotropic and positive for a species
the state loads as Maxwell–Jüttner, and in a moving window a perturbation amplitude, background
fields, or, when the window starts later than t = 0, a drift or a thermal speed. `initialize!` and `step!` call it
first, so parameters that were made for another state stop the run instead of giving another
run silently. The checks are host comparisons of host scalars and of types (no device data,
§0.4 rule 2), a few per step; a compiler that traces the parameters would hoist them out of the
step, which is why they are one function.
"""
function check_params(st::EMState{N,T}, p::EMParams) where {N,T}
    eltype(p) === T ||
        throw(ArgumentError("the parameters have element type $(eltype(p)), the state $T: build the state from a problem of the parameters' element type"))
    length(p.species) == length(st.species) ||
        throw(ArgumentError("$(length(p.species)) species in the parameters, $(length(st.species)) in the state"))
    all(ps -> ps isa SpeciesParams{T,N}, p.species) ||
        throw(ArgumentError("the species' parameters must be SpeciesParams{$T,$N}, got $(map(typeof, p.species))"))
    foreach(check_species_params, p.species, st.setup)
    check_aux_params(st, p)
    return p
end

# A species the state loads as Maxwell–Jüttner takes one positive thermal speed (the loader reads
# the first component only), as `SpeciesDescription` requires.
function check_species_params(ps::SpeciesParams, setup)
    th = ps.thermal
    setup.momentum === :maxwell_juttner && !(th[1] == th[2] == th[3] && th[1] > 0) &&
        throw(ArgumentError("species $(setup.name): the state loads it as Maxwell–Jüttner, which takes the same nonzero thermal speed in every component, got $th"))
    return nothing
end

function check_aux_params(::EMState{1}, p::EMParams)
    isempty(p.lasers) || throw(ArgumentError("laser antennas are implemented at N = 2 only"))
    p.witness === nothing || throw(ArgumentError("a witness bunch is implemented at N = 2 only"))
    p.profile === nothing || throw(ArgumentError("a density profile is implemented with the moving window only"))
    return nothing
end

"""
    check_setup(state, problem) -> problem

Whether `state` has the configuration of `problem` (an `ArgumentError` otherwise), what a state
takes from the problem it is built for besides the parameters: the cells and the cell sizes, the
time step, per species the name, the particles per cell, the Philox stream, the kind of momentum
distribution, the charge and the mass (as the state holds them, converted to its element type:
a description in another type than the problem's is the state's own problem), and at N = 2 which
boundaries are open or a window, the
window's start and the witness bunch's placement, charge and mass; and the diagnostic mode of
the state's own recorder, whose histories would otherwise be of another mode than the problem's. Called by
`initialize!(state, problem)`; the problem's parameters are checked by [`check_params`](@ref)
(the lasers' planes among them).
"""
function check_setup(st::EMState{N}, problem::EMProblem{M}) where {N,M}
    wrong(what) = throw(ArgumentError("the problem \"$(problem.name)\" does not have the configuration the state was built for ($what): build a new EMState for it"))
    N == M || wrong("N = $M against $N")
    st.recorder.kmode == problem.kmode ||
        throw(ArgumentError("the problem \"$(problem.name)\" has the diagnostic mode $(problem.kmode), the state's recorder $(st.recorder.kmode): the state's histories would be of another mode; observe with HistoryRecorder(problem, state), or build a new EMState"))
    ncells(st.grid) == problem.cells || wrong("cells $(problem.cells) against $(ncells(st.grid))")
    cellsize(st.grid) == problem.length ./ problem.cells || wrong("cell sizes $(problem.length ./ problem.cells) against $(cellsize(st.grid))")
    st.dt == problem.dt || wrong("dt $(problem.dt) against $(st.dt)")
    map(species_setup, problem.species) == st.setup ||
        wrong("the species' names, particles per cell, streams or momentum distributions $(map(species_setup, problem.species)) against $(st.setup)")
    all(map((s, d) -> s.q == oftype(s.q, d.charge) && s.m == oftype(s.m, d.mass), st.species, problem.species)) ||
        wrong("a species' charge or mass")
    check_aux_setup(st, problem, wrong)
    return problem
end

check_aux_setup(::EMState{1}, problem, wrong) = nothing

"""
    initialize!(state, params; momenta_at = :t0, density_at = :particle) -> state
    initialize!(state, problem; momenta_at = :t0, density_at = :particle) -> state

The state at t = 0 from the parameters ([`EMParams`](@ref); or from a problem description, whose
`params(problem)` are taken after [`check_setup`](@ref)): each species loaded by the engine's quiet
start (`load_quiet!`, weights `density·(1 + amplitude·cos(k·x))/ppc`, so the parameters enter
through the weights, A3; momenta from the species' distribution, [`momentum_distribution`](@ref),
by Philox draws keyed by the parameters' seed and the species' stream, [`load_species!`](@ref): a
Maxwell–Jüttner or Gaussian-momentum draw writes `u = γv` directly, a Gaussian-velocity draw is
converted, and refused if a drawn speed reaches c, `check_subluminal`); the fields zero; Ex from the charge, as Smilei solves at t = 0: `cumsum_periodic!` of the deposited
ρ, so that (Ex[j] − Ex[j−1])/Δx = ρ[j] − mean(ρ) to roundoff, the mean being the immobile
neutralizing background; the background fields `E0` and `B0` of the parameters added where they
are nonzero (so a run without them keeps the bits of the fields); then the momenta at
t = −Δt/2 (half-step centring, A2), and the kinetic energy density there. Everything a run
evolves starts here, so a state that has run before is the same as a new one after it; the
recorder's histories are not touched (entry `k` is overwritten when step `k` is observed again).

`momenta_at` says at which time the loaded momenta are taken:

- `:t0` (the default): the loaded momenta are `u⁰`, and `initialize!` moves them half a step
  back, `uⁿ⁻¹ᐟ²` from `u⁰` by the pusher with −Δt/2 in E⁰, B⁰;
- `:t_minus_half`: the loaded momenta are already `u⁻¹ᐟ²` and are kept as loaded, with no kick.
  This is the convention of Smilei and PIConGPU, which the 1D cross-code runs use; the two
  starts differ by O(Δt·E⁰) in the momenta.

Positions, weights and fields are the same in both; anything else is an `ArgumentError`.

`density_at` says where the loader evaluates the density profile (`load_quiet!`): `:particle`
(the default) at each particle's position, `:cell_centre` at the centre of its cell, so the
loaded profile is piecewise constant per cell, as Smilei and PIConGPU sample it, which lowers a
cosine's amplitude by about sinc(kΔx/2) ≈ 1 − (kΔx)²/24 against the per-particle load (#226: a
constant 1.00017 in the field-energy ratio of the 1D two-stream case). Positions and the momenta
draws are the same in both; anything else is an `ArgumentError`.
"""
function initialize!(st::EMState{1}, p::EMParams; momenta_at::Symbol = :t0, density_at::Symbol = :particle)
    check_momenta_at(momenta_at)
    ParticleMeshCore.check_density_at(density_at)
    check_params(st, p)
    (; numerics, grid) = st
    T = eltype(st.ρ)
    wgs = numerics.workgroupsize
    for (s, ps, setup) in zip(st.species, p.species, st.setup)
        fill!(s.status, ParticleMeshCore.FREE)
        load_species!(s, grid, ps, setup, p.seed; density_at, workgroupsize = wgs)
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
    add_background!(st.fields, p.E0, p.B0)
    if momenta_at === :t0
        g = Int32(guards(grid))
        for s in st.species
            launcher(kick_kernel!, get_backend(s), wgs)(s, st.fields, numerics.pusher, numerics.shape, g,
                                                        -s.q / s.m * st.dt / 4; ndrange = capacity(s))
        end
    end
    kinetic!(st)
    return st
end

initialize!(st::EMState, problem::EMProblem; kw...) = (check_setup(st, problem); initialize!(st, params(problem); kw...))

"""
    step!(state, params, k; observer = state.recorder) -> state

Step `k` (from t = (k−1)Δt to kΔt, Δt the state's), a function of the state and the parameters
([`EMParams`](@ref)) with every buffer in the state and no host-side branch on device data
(§0.4). First the `observer` sees the state at the start of the step ([`observe!`](@ref): Eⁿ,
Bⁿ and the kinetic energy at n − 1/2 that the previous step left in `K`; `nothing` observes
nothing); then the module's step order:

1. per species: store the old cell and offset as extras, push the momentum in Eⁿ, Bⁿ gathered
   at xⁿ (the pusher of the numerics) and move the particle, xⁿ⁺¹ = xⁿ + Δt·uₓ/γ;
2. sort by cell where the deposition mode needs it (the extras follow the particles);
3. zero Jx, Jy, Jz and add every species' current with the engine's `deposit_flux!` (Esirkepov
   for Jx from the old and new positions, `q·w·u_{y,z}/γ` for Jy and Jz), fold the guards, then
   apply the numerics' [`BinomialFilter`](@ref) (nothing with zero passes; the scratch is `ρ`);
4. deposit the kinetic energy density at n + 1/2 (for the next observation);
5. the mesh solve of the numerics (Yee: B half step, E full step with J, B half step).

Gauss's law then holds to roundoff after every step, from continuity (T-212); with filter passes
its filtered form does (T-441). At N = 1 a step reads no parameter (there is no source and no
refill), but parameters that do not fit the state are refused as at N = 2
([`check_params`](@ref)); the default observer, the state's [`HistoryRecorder`](@ref), refuses a
`k` outside the steps it was allocated for with an `ArgumentError`.
"""
function step!(st::EMState{1}, p::EMParams, k::Integer; observer = st.recorder)
    check_params(st, p)
    (; numerics, grid, fields, dt) = st
    T = eltype(st.ρ)
    wgs = numerics.workgroupsize
    n = Int32(ncells(grid))
    g = Int32(guards(grid))
    dx = cellsize(grid)
    observe!(observer, st, k)
    for s in st.species
        launcher(push_kernel!, get_backend(s), wgs)(s, fields, numerics.pusher, numerics.shape, g, s.q / s.m * dt / 2,
                                                    dt / dx, n; ndrange = capacity(s))
    end
    for (s, ws) in zip(st.species, st.ws)
        sorts(st, s) && sort_by_cell!(s, grid, ws; workgroupsize = wgs)
    end
    fill!(fields.Jx, zero(T))
    fill!(fields.Jy, zero(T))
    fill!(fields.Jz, zero(T))
    for (s, ws) in zip(st.species, st.ws)
        ParticleMeshCore.deposit_flux!(deposit_mode(st, s), fields.Jx, fields.Jy, fields.Jz, s, numerics.shape, RHO_STAGGER,
                                       grid, ws; dt, quantity_y = CurrentY(), quantity_z = CurrentZ(),
                                       workgroupsize = wgs)
    end
    fold_guards!(fields.Jx, grid; workgroupsize = wgs)
    fold_guards!(fields.Jy, grid; workgroupsize = wgs)
    fold_guards!(fields.Jz, grid; workgroupsize = wgs)
    filter_current!(numerics.filter, fields, grid, st.ρ; workgroupsize = wgs)
    kinetic!(st)
    solve_fields!(numerics.solver, fields, grid, dt, st.modifiers, p, k; workgroupsize = wgs)
    return st
end

"""
    run!(state, params, nsteps; first = 1, observer = state.recorder)

The plain loop: `step!` with `observer` for `k = first:first+nsteps-1`. Call
[`initialize!`](@ref) once before, or [`restore!`](@ref) a snapshot and continue with
`first = snapshot.k + 1`.
"""
function run!(st::EMState, p::EMParams, nsteps::Integer; first::Integer = 1, observer = st.recorder)
    for k in first:(first + nsteps - 1)
        step!(st, p, k; observer)
    end
    return st
end
