# The module's step: composed here from engine pieces, in the module's own order. The engine owns
# no System, State or step! (architecture-draft.md §0.7); this file is where the electromagnetic
# step order lives. `initialize!` and `step!` are one method each over the number N of resolved
# axes, the 1D case being the case of one axis (WP-057); `step!`'s documentation names what is
# per N.

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

"E and B of the layout at offset `δ` of cell `c` (one entry per resolved axis; guards filled), each on its own stagger."
@inline function fields_at(f, c::NTuple{N,Int32}, δ::NTuple{N,Any}, shape, g::Int32) where {N}
    L = gather_layout(Val(N))
    E = (gather_at(f.Ex, c, δ, shape, L.Ex, g), gather_at(f.Ey, c, δ, shape, L.Ey, g), gather_at(f.Ez, c, δ, shape, L.Ez, g))
    B = (gather_at(f.Bx, c, δ, shape, L.Bx, g), gather_at(f.By, c, δ, shape, L.By, g), gather_at(f.Bz, c, δ, shape, L.Bz, g))
    return E, B
end

# At N = 1 the gathers are the engine's 1D calls on the bare staggers of `FIELD_LAYOUT`, as the 1D
# step has always made them. A dispatch on N that stays: the N-axis gather gives the same bits at
# N = 1 (T-756), but compiles to another shape for duals with the quadratic shape (its per-axis
# weights are a `map` whose closure is not inlined there), and the 1D step keeps its code.
@inline function fields_at(f, c::NTuple{1,Int32}, δ::NTuple{1,Any}, shape, g::Int32)
    L = FIELD_LAYOUT
    c1, δ1 = c[1], δ[1]
    E = (gather_at(f.Ex, c1, δ1, shape, L.Ex, g), gather_at(f.Ey, c1, δ1, shape, L.Ey, g), gather_at(f.Ez, c1, δ1, shape, L.Ez, g))
    B = (gather_at(f.Bx, c1, δ1, shape, L.Bx, g), gather_at(f.By, c1, δ1, shape, L.By, g), gather_at(f.Bz, c1, δ1, shape, L.Bz, g))
    return E, B
end

# The particle's cell and offset after a move by Δt·u_d/γ along each resolved axis. At N = 1 the
# move is written as the 1D step has always written it, a dispatch on N that stays because the bits
# differ on a device: a device compiler may contract a multiplication and an addition into one
# rounding, which of them depends on the shape of the expression, and with duals at N = 1 the
# N-tuple form rounds the new offset's value otherwise than the 1D form does (CUDA; the CPU's bits
# are the same in both forms). T-756 holds both kernels against the 1D kernels of before WP-057
# on the backend they run on.
@inline function moved(c::NTuple{N}, δ::NTuple{N}, u, dt_dx, n) where {N}
    γ = lorentz_factor(u)
    return advance(c, δ, ntuple(d -> u[d] / γ * dt_dx[d], Val(N)), n)
end
@inline function moved(c::NTuple{1}, δ::NTuple{1}, u, dt_dx, n)
    c′, δ′ = advance(c[1], δ[1], u[1] / lorentz_factor(u) * dt_dx[1], n[1])
    return (c′,), (δ′,)
end

# The position before the move, where the engine's `deposit_flux!` reads it: the extras of
# `flux_extras(T, N)`, whose names are the engine's per N.
@inline function store_old!(x, i, c::NTuple{1}, δ::NTuple{1})
    @inbounds x.old_cell[i] = c[1]
    @inbounds x.old_offset[i] = δ[1]
    return nothing
end
@inline function store_old!(x, i, c::NTuple{2}, δ::NTuple{2})
    @inbounds x.old_cell_1[i] = c[1]
    @inbounds x.old_cell_2[i] = c[2]
    @inbounds x.old_offset_1[i] = δ[1]
    @inbounds x.old_offset_2[i] = δ[2]
    return nothing
end

@inline function set_position!(s, i, c::NTuple{N}, δ::NTuple{N}) where {N}
    ntuple(d -> (@inbounds s.cell[d][i] = c[d]; @inbounds s.offset[d][i] = δ[d]; nothing), Val(N))
    return nothing
end

# Store the old position per axis (the extras deposit_flux! reads), push the momentum in Eⁿ, Bⁿ at
# xⁿ, and move the particle by Δt·u_d/γ along each resolved axis (less than one cell per axis below
# the Courant limit, as deposit_flux! requires). One kernel over N: at N = 1 the cell, the offset,
# `dt_dx` and `n` are 1-tuples, and the gather (`fields_at`), the move (`moved`) and the extras
# (`store_old!`) are its three leaves that are per N.
@kernel function push_kernel!(s, f, pusher, shape, g::Int32, qm_h, dt_dx::NTuple{N}, n::NTuple{N,Int32}) where {N}
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE
        c = particle_cell(s, i, Val(N))
        δ = particle_offset(s, i, Val(N))
        store_old!(s.extras, i, c, δ)
        E, B = fields_at(f, c, δ, shape, g)
        u = push_momentum(pusher, (s.u[1][i], s.u[2][i], s.u[3][i]), E, B, qm_h)
        s.u[1][i] = u[1]
        s.u[2][i] = u[2]
        s.u[3][i] = u[3]
        c′, δ′ = moved(c, δ, u, dt_dx, n)
        set_position!(s, i, c′, δ′)
    end
end

# The momentum update alone (no move): the half step back of initialize!, with qm_h = −(q/m)Δt/4.
@kernel function kick_kernel!(s, f, pusher, shape, g::Int32, qm_h)
    i = @index(Global, Linear)
    @inbounds if s.status[i] == ALIVE
        vN = Val(length(s.cell))
        E, B = fields_at(f, particle_cell(s, i, vN), particle_offset(s, i, vN), shape, g)
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
open_absorber(st::EMState) = pick_modifier(OpenAbsorber, st.modifiers)
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

function check_aux_params(st::EMState, p::EMParams)
    antennas, window = antenna_planes(st), window_frame(st)
    length(p.lasers) == length(antennas) ||
        throw(ArgumentError("$(length(p.lasers)) lasers in the parameters, $(length(antennas)) antenna planes in the state$(ndims(st) == 1 ? " (laser antennas are implemented at N = 2 only)" : "")"))
    foreach((l, a) -> check_plane(a, l), p.lasers, antennas)
    window === nothing || check_window_antennas(window, back_absorber(st), map(Antenna, p.lasers, antennas))
    (p.witness === nothing) == (witness_frame(st) === nothing) ||
        throw(ArgumentError("the parameters $(p.witness === nothing ? "have no" : "have a") witness bunch, the state $(witness_frame(st) === nothing ? "has none" : "has one")"))
    p.profile === nothing || window !== nothing ||
        throw(ArgumentError("a density profile is implemented with the moving window only"))
    window === nothing || check_window_params(window, p)
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

# What a problem's boundaries, window and witness bunch ask of a state: the absorber of an open
# axis, the window's frame and the bunch's placement (none of them at N = 1, where no problem has one).
function check_aux_setup(st::EMState, problem, wrong)
    window = window_frame(st)
    (last(problem.boundaries) === :open) == (open_absorber(st) !== nothing) || wrong("axis 2 is $(last(problem.boundaries))")
    (problem.window === nothing ? window === nothing : window !== nothing && window.window == problem.window) ||
        wrong("the moving window $(problem.window)")
    w = witness_frame(st)
    b = problem.witness
    (b === nothing) == (w === nothing) || wrong("the witness bunch")
    b === nothing || (witness_placement(b, st.grid, st.dt) == (w.kinj, w.c0, w.nc, w.ppc) && w.species.q == oftype(w.species.q, b.charge) && w.species.m == oftype(w.species.m, b.mass)) ||
        wrong("the witness bunch's injection step, lattice, charge or mass")
    return nothing
end

"""
    initialize!(state, params; momenta_at = :t0, density_at = :particle) -> state
    initialize!(state, problem; momenta_at = :t0, density_at = :particle) -> state

The state at t = 0 from the parameters ([`EMParams`](@ref); or from a problem description, whose
`params(problem)` are taken after [`check_setup`](@ref)), one method for N = 1 and N = 2: each
species loaded by the engine's quiet start (`load_quiet!`, weights
`density·(1 + amplitude·cos(k·x))/prod(ppc)` with the parameters' wave vector, so the parameters
enter through the weights, A3; momenta from the species' distribution,
[`momentum_distribution`](@ref), by Philox draws keyed by the parameters' seed and the species'
stream, [`load_species!`](@ref): a Maxwell–Jüttner or Gaussian-momentum draw writes `u = γv`
directly, a Gaussian-velocity draw is converted, and refused if a drawn speed reaches c,
`check_subluminal`; with a moving window the window's own load of its plasma columns); the witness
bunch's buffer empty and the window's counters zero; the fields zero, and with them the
convolution fields of a PML (they are field state: ψ of a field that starts at zero is zero, and a
state that ran before must not carry the last run's into this one); the initial field from the
deposited charge (`initial_field!`, below); the background fields `E0` and `B0` of the parameters
added where they are nonzero (so a run without them keeps the bits of the fields); then the
momenta at t = −Δt/2 (half-step centring, A2), and the kinetic energy there. Everything a run
evolves starts here, so a state that has run before is the same as a new one after it; the
recorder's histories are not touched (entry `k` is overwritten when step `k` is observed again).

The initial field is where the two dimensions differ: at N = 1, Ex from the charge as Smilei
solves at t = 0, `cumsum_periodic!` of the deposited ρ, so that (Ex[j] − Ex[j−1])/Δx =
ρ[j] − mean(ρ) to roundoff, the mean being the immobile neutralizing background; at N = 2, Ex and
Ey by the engine's periodic solve, `φ` from `inverse_laplacian_periodic!` of ρ and `E = G` from
`gradient_periodic!`, so that the backward-difference divergence of E is ρ − mean(ρ) to roundoff;
with a moving window the plasma starts neutral against its immobile background, so E⁰ = 0 and no
solve is made (the charge in the x-guards is dropped).

`momenta_at` says at which time the loaded momenta are taken:

- `:t0` (the default): the loaded momenta are `u⁰`, and `initialize!` moves them half a step
  back, `uⁿ⁻¹ᐟ²` from `u⁰` by the pusher with −Δt/2 in E⁰, B⁰;
- `:t_minus_half`: the loaded momenta are already `u⁻¹ᐟ²` and are kept as loaded, with no kick.
  This is the convention of Smilei and PIConGPU, which the cross-code runs use; the two
  starts differ by O(Δt·E⁰) in the momenta.

Positions, weights and fields are the same in both; anything else is an `ArgumentError`.

`density_at` says where the loader evaluates the density profile (`load_quiet!`): `:particle`
(the default) at each particle's position, `:cell_centre` at the centre of its cell, so the
loaded profile is piecewise constant per cell, as Smilei and PIConGPU sample it, which lowers a
cosine's amplitude by about sinc(kΔx/2) ≈ 1 − (kΔx)²/24 against the per-particle load (#226: a
constant 1.00017 in the field-energy ratio of the 1D two-stream case). Positions and the momenta
draws are the same in both; anything else is an `ArgumentError`, and so is `:cell_centre` with a
moving window, whose loads and refills evaluate the density at each particle only.
"""
function initialize!(st::EMState{N}, p::EMParams; momenta_at::Symbol = :t0, density_at::Symbol = :particle) where {N}
    check_momenta_at(momenta_at)
    ParticleMeshCore.check_density_at(density_at)
    window = window_frame(st)
    window === nothing || density_at === :particle ||
        throw(ArgumentError("density_at = :$density_at: the moving window's loads and refills evaluate the density at each particle (:particle) only"))
    check_params(st, p)
    (; numerics, grid) = st
    T = eltype(st.ρ)
    wgs = numerics.workgroupsize
    for (i, (s, ps, setup)) in enumerate(zip(st.species, p.species, st.setup))
        fill!(s.status, ParticleMeshCore.FREE)
        window === nothing || (load_window_species!(window, i, s, grid, wgs, window_load(setup, ps, p), setup.name); continue)
        load_species!(s, grid, ps, setup, p.seed; density_at, workgroupsize = wgs)
    end
    clear_witness!(witness_frame(st))
    foreach(a -> fill!(a, zero(T)), st.fields)
    reset_convolution!(st)
    for (s, ws) in zip(st.species, st.ws)
        sorts(st, s) && sort_by_cell!(s, grid, ws; workgroupsize = wgs)
    end
    fill!(st.ρ, zero(T))
    for (s, ws) in zip(st.species, st.ws)
        deposit!(deposit_mode(st, s), st.ρ, s, numerics.shape, rho_stagger(Val(N)), grid, ws; workgroupsize = wgs)
    end
    zero_x_edges!(window, st.ρ; workgroupsize = wgs)
    fold_guards!(st.ρ, grid; workgroupsize = wgs)
    initial_field!(st)
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

# The initial field from the deposited, folded charge in `st.ρ`. A dispatch on N that stays: the
# 1D sum and the 2D spectral solve give different bits for the same charge.
initial_field!(st::EMState{1}) = (cumsum_periodic!(st.fields.Ex, st.ρ, st.grid; workgroupsize = st.numerics.workgroupsize); st)

initialize!(st::EMState, problem::EMProblem; kw...) = (check_setup(st, problem); initialize!(st, params(problem); kw...))

# The transverse components the engine's `deposit_flux!` takes as per-particle quantities, the
# currents along the axes that are not resolved: Jy and Jz at N = 1, Jz at N = 2 (the in-plane
# components are Esirkepov's). A dispatch on N that stays: it is the scheme's.
transverse_currents(::Val{1}) = (; quantity_y = CurrentY(), quantity_z = CurrentZ())
transverse_currents(::Val{2}) = (; quantity_z = CurrentZ())

# The current of the step, by the numerics' choice: none, or the charge-conserving one.
current!(::NoCurrent, st::EMState) = st

# Sort where the deposition mode needs it (the old positions in the extras follow the
# particles), zero Jx, Jy, Jz, add every species' current with the engine's `deposit_flux!`
# (Esirkepov from the old and new positions along the resolved axes, `q·w·u/γ` along the others),
# fold the guards (with a moving window, the x-guards dropped first) and apply the numerics'
# binomial filter (through the scratch field ρ, which only initialize! and the window shift read
# otherwise).
function current!(::Esirkepov, st::EMState{N}) where {N}
    (; numerics, grid, fields) = st
    T = eltype(st.ρ)
    wgs = numerics.workgroupsize
    for (s, ws) in zip(st.species, st.ws)
        sorts(st, s) && sort_by_cell!(s, grid, ws; workgroupsize = wgs)
    end
    fill!(fields.Jx, zero(T))
    fill!(fields.Jy, zero(T))
    fill!(fields.Jz, zero(T))
    for (s, ws) in zip(st.species, st.ws)
        deposit_flux!(deposit_mode(st, s), fields.Jx, fields.Jy, fields.Jz, s, numerics.shape, rho_stagger(Val(N)), grid, ws;
                      dt = st.dt, transverse_currents(Val(N))..., workgroupsize = wgs)
    end
    window = window_frame(st)
    for J in (fields.Jx, fields.Jy, fields.Jz)
        zero_x_edges!(window, J; workgroupsize = wgs)
        fold_guards!(J, grid; workgroupsize = wgs)
    end
    filter_current!(numerics.filter, fields, grid, st.ρ; workgroupsize = wgs, edges = window)
    return st
end

"""
    step!(state, params, k; observer = state.recorder) -> state

Step `k` (from t = (k−1)Δt to kΔt, Δt the state's), a function of the state and the parameters
([`EMParams`](@ref)) with every buffer in the state and no host-side branch on device data
(§0.4). One method for N = 1 and N = 2, the 1D case being the case of one resolved axis; what a
problem does not have (a witness bunch, a moving window, an open axis, field modifiers: none of
them at N = 1) is `nothing` or empty and costs nothing. First the `observer` sees the state at
the start of the step ([`observe!`](@ref): Eⁿ, Bⁿ and the kinetic energy at n − 1/2 that the
previous step left in `K`; `nothing` observes nothing); then the module's step order:

1. a witness bunch (WP-047, [`WitnessBunch`](@ref)) is injected at the start of its step with the
   parameters' momentum and weights; from then on it is pushed with the species and moved and
   retired at the edges as they are, and deposits nothing;
2. per species: store the old position as extras, push the momentum in Eⁿ, Bⁿ gathered at xⁿ on
   each component's stagger (the pusher of the numerics) and move the particle along every
   resolved axis, xⁿ⁺¹ = xⁿ + Δt·u/γ;
3. with a moving window (WP-028), retire the particles that crossed an x edge, before the
   current (their current would land at the opposite edge);
4. the current of the numerics' choice. [`Esirkepov`](@ref)`()`: sort by cell where the
   deposition mode needs it (the extras follow the particles), zero Jx, Jy, Jz and add every
   species' current with the engine's `deposit_flux!` (charge-conserving along the resolved axes,
   `q·w·u/γ` along the others: Jy and Jz at N = 1, Jz at N = 2), fold the guards, then apply the
   numerics' [`BinomialFilter`](@ref) (nothing with zero passes; the scratch is `ρ`).
   `NoCurrent()` (N = 2): J stays as it is;
5. with an open axis 2 (WP-027), retire the particles that crossed the box edge, after the current
   of their last move is deposited;
6. the kinetic energy at n + 1/2, for the next observation;
7. the mesh solve of the numerics, [`solve_fields!`](@ref) (Yee: B half step, E full step with J,
   B half step), which walks the state's field modifiers around the sub-steps
   ([`field_modifiers`](@ref): an open axis' absorbing layer or PML, the PML of a window's back
   edge, the antennas, each the parameters' laser on the state's plane, and the window's x edges);
8. with a moving window, the shift by the cells it moved in this step ([`window_shifts`](@ref)):
   E, B and the particles one column back, the back column dropped, the fresh plasma of the
   injection column loaded with the parameters' densities, profile, drifts, thermal speeds and
   seed.

With `Esirkepov()` and no filter passes, Gauss's law then holds to roundoff after every step, from
continuity (T-212, T-360), off the antenna columns with lasers (T-406), and off the layer rows
with an open boundary (T-451); with passes its filtered form does (T-441). Parameters that do
not fit the state are refused first ([`check_params`](@ref): a laser off its plane, a pulse the
window would pass while it emits, another element type; at N = 1 a step reads no parameter, there
being no source and no refill, but the check is the same); the default observer, the state's
[`HistoryRecorder`](@ref), refuses a `k` outside the steps it was allocated for with an
`ArgumentError`. The step reads and writes the state's evolved arrays ([`evolved`](@ref)) and
nothing else that outlives it but the cache `K`, and `k` enters only through the source's time,
the window's position and the witness bunch's injection step: no step is special, so a run
continues from any step of a saved state ([`restore!`](@ref)).

Three pieces of the step are per N by dispatch, because each gives other bits than its
counterpart: the kinetic energy (`kinetic!`: a deposited density at N = 1, sums over fixed chunks
of particle slots at N = 2), in `initialize!` the initial field, and the move of a particle
(`moved`: on a device the N-tuple form rounds a dual's new offset otherwise than the 1D form).
Besides them N enters through the engine's conventions only: its 1D gather on a bare stagger
(`fields_at`), the names of the extras that hold the old position (`store_old!`), a bare stagger
in the 1D depositions (`rho_stagger`), and which components of the current are per-particle
quantities (`transverse_currents`).
"""
function step!(st::EMState, p::EMParams, k::Integer; observer = st.recorder)
    check_params(st, p)
    (; numerics, grid, fields, dt) = st
    wgs = numerics.workgroupsize
    n = Int32.(grid.n)
    g = Int32(guards(grid))
    dt_dx = map(h -> dt / h, grid.dx)
    window = window_frame(st)
    observe!(observer, st, k)
    inject_witness!(witness_frame(st), p.witness, grid, k, wgs)
    for s in (st.species..., witness_species(st)...)
        launcher(push_kernel!, get_backend(s), wgs)(s, fields, numerics.pusher, numerics.shape, g, s.q / s.m * dt / 2, dt_dx, n;
                                                    ndrange = capacity(s))
    end
    retire_x!(window, st)
    current!(numerics.current, st)
    retire!(open_absorber(st), st)
    kinetic!(st)
    solve_fields!(numerics.solver, fields, grid, dt, st.modifiers, p, k; workgroupsize = wgs)
    advance_window!(window, st, p, k)
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
