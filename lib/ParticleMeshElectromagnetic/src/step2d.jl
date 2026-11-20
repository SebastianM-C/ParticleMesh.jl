# The module at N = 2: what the 2D state has of its own. The step and `initialize!` are step.jl's,
# one method each over N; here are the state's constructor with its field modifiers, the kinetic
# energy as sums over chunks of particle slots, and the initial field by the periodic spectral
# solve (WP-024, WP-023).

"Particle slots per chunk of the 2D kinetic-energy reduction (the chunks fix its association order)."
const KINETIC_CHUNK = 256

# One work item per chunk of `chunk` slots: the kinetic energy m·w·(γ − 1) of its alive
# particles, summed serially in slot order from zero, written to K[first + c − 1].
@kernel function kinetic_chunks_kernel!(K, s, first::Int32, cap::Int32, chunk::Int32)
    c = Int32(@index(Global, Linear))
    T = eltype(K)
    acc = zero(T)
    lo = (c - Int32(1)) * chunk + Int32(1)
    hi = min(c * chunk, cap)
    @inbounds for i in lo:hi
        if s.status[i] == ALIVE
            acc += KineticEnergy()(s, i)
        end
    end
    @inbounds K[first + c - Int32(1)] = acc
end

"""
    EMState(problem::EMProblem{2}, numerics; backend = CPU(), every = 1, record = RECORD_GROUPS)

The state of a 2D problem ([`EMState`](@ref)): the grid of `problem.cells` cells per axis with
[`em_guard_width`](@ref) guards, the time step `problem.dt`, one species per
[`SpeciesDescription`](@ref) at capacity `prod(cells)·prod(ppc)` with the extras of
`flux_extras(T, 2)` (the old position per axis, which the push stores), the workspaces, the fields
of [`field_layout`](@ref)`(Val(2))`, `ρ`, the kinetic-energy chunks `K` (one entry per
[`KINETIC_CHUNK`](@ref) slots of each species; none without species), the default [`HistoryRecorder`](@ref) for
`problem.kmode` and `problem.nsteps` steps (recording every `every` steps the groups of `record`), the field modifiers of the mesh solve
([`field_modifiers`](@ref): the absorber placed on the grid, the absorbing layer's
[`LayerFactors`](@ref) or the PML's [`PMLFields`](@ref); the back edge's [`BackPMLFields`](@ref);
the [`AntennaPlanes`](@ref) of the problem's lasers, where each plane lies on the grid, not the
laser; the window's [`WindowEdges`](@ref)), and in `aux`: the spectral workspace and the potential
`φ` of the initial solve, the moving window's [`WindowFrame`](@ref) (`window`, `nothing` without one; its
species buffers then hold `(1 + numerics.headroom)` times the window's particles) and the
[`WitnessFrame`](@ref) of a witness bunch (`witness`, or `nothing`) and the
[`IonizationFrame`](@ref) of a dopant (`ionization`, or `nothing`; its electrons are part of `K`,
after the species).

With the current [`Esirkepov`](@ref)`()` each workspace also holds the per-cell patches of the
`Reference()` flux deposition (`Workspace(species, grid; flux = (shape, ρ's stagger))`, also
used by `Fast()` on an element type without atomics). Throws an `ArgumentError` unless the
current is `Esirkepov()` or [`NoCurrent`](@ref)`()` and unless `problem.dt` is below the
Courant limit of the numerics' solver ([`check_courant`](@ref)), and unless the numerics'
`absorber` is an [`AbsorbingLayer`](@ref) or a [`PML`](@ref) exactly when the problem's axis 2 is
`:open` (the absorber also checks its fit on the grid), and unless the numerics' `filter` has zero
passes with an open axis 2 (the filter wraps the current across the open edge, and no test checks
the two together yet), and unless the problem's own parameters fit the planes and the window
([`check_params`](@ref)'s conditions: with a window, every pulse over before the window passes its
antenna).
"""
function EMState(problem::EMProblem{2,T}, numerics::EMNumerics; backend = CPU(), every::Integer = 1, record = RECORD_GROUPS) where {T}
    numerics.current isa Union{Esirkepov,NoCurrent} ||
        throw(ArgumentError("a 2D state takes current = Esirkepov() or NoCurrent() in EMNumerics, got $(numerics.current)"))
    n = problem.cells
    wgs = numerics.workgroupsize
    grid = PeriodicGrid(n, problem.length ./ n, em_guard_width(numerics.shape, numerics.solver))
    check_courant(numerics.solver, grid, problem.dt)
    open_y = problem.boundaries[2] === :open
    open_y && !(numerics.absorber isa Union{AbsorbingLayer,PML}) &&
        throw(ArgumentError("axis 2 of the problem is open: EMNumerics needs absorber = AbsorbingLayer(...) or PML(...), got $(numerics.absorber)"))
    !open_y && numerics.absorber !== nothing &&
        throw(ArgumentError("absorber = $(numerics.absorber) needs an open axis 2 in the problem (boundaries = (:periodic, :open))"))
    open_y && numerics.filter.passes > 0 &&
        throw(ArgumentError("filter = $(numerics.filter) with absorber = $(numerics.absorber): the binomial filter is not yet tested with an open boundary (it wraps J across the open edge), so an open axis 2 takes BinomialFilter(0)"))
    layers = open_y ? (place_absorber(numerics.absorber, grid, problem.dt; axis = 2, backend, workgroupsize = numerics.workgroupsize,
                                    solver = numerics.solver),) : ()
    numerics.back_absorber === nothing || numerics.back_absorber isa PML ||
        throw(ArgumentError("back_absorber = $(numerics.back_absorber): the window's back edge takes a PML(...) or nothing"))
    problem.window === nothing && numerics.back_absorber !== nothing &&
        throw(ArgumentError("back_absorber = $(numerics.back_absorber) needs a moving window (boundaries = (:window, ·))"))
    window = problem.window === nothing ? nothing : WindowFrame(problem, grid, backend, numerics.shape, numerics.filter.passes)
    back = numerics.back_absorber === nothing ? nothing : BackPMLFields(numerics.back_absorber, grid, problem.dt; backend, workgroupsize = wgs,
                                                                                       solver = numerics.solver)
    planes = map(l -> AntennaPlane(l.position, grid, problem.dt, numerics.solver), problem.lasers)
    window === nothing || check_window_antennas(window, back, map(Antenna, params(problem).lasers, planes))
    witness = problem.witness === nothing ? nothing : WitnessFrame(problem, grid, backend)
    ionization = problem.ionization === nothing ? nothing : IonizationFrame(problem, grid, backend, numerics, window)
    species = map(problem.species) do d
        cap = prod(n) * prod(d.ppc)
        window === nothing || (cap = ceil(Int, (1 + numerics.headroom) * cap))
        Species(backend, T, cap; q = d.charge, m = d.mass, N = 2, extras = flux_extras(T, 2))
    end
    flux = numerics.current isa Esirkepov ? (numerics.shape, RHO_STAGGER_2D) : nothing
    # a window refills one column of a species behind its count after each shift's sort
    ws = map((s, d) -> Workspace(s, grid; flux, append = sort_append(numerics, window === nothing ? 0 : n[2] * prod(d.ppc)),
                                 bins = sort_bins(numerics, Val(2); window = window !== nothing)), species,
             problem.species)
    nK = sum(s -> cld(capacity(s), KINETIC_CHUNK), (species..., born_species(ionization)...); init = 0)
    recorder = HistoryRecorder(backend, T, grid, problem.length, problem.kmode, problem.nsteps; workgroupsize = wgs, every, record)
    aux = (spectral = SpectralWorkspace(backend, T, grid; workgroupsize = wgs), φ = field(backend, T, grid),
           window = window, witness = witness, ionization = ionization)
    check_gather(numerics.gather, numerics.shape, grid)
    fields = em_fields(backend, T, grid)
    return EMState(numerics, grid, problem.dt, species, ws, fields, gather_scratch(numerics.gather, fields), field(backend, T, grid),
                   KernelAbstractions.zeros(backend, T, nK), recorder, map(species_setup, problem.species),
                   field_modifiers(layers, back, planes, window), aux, Ref(false))
end

"""
    field_modifiers(layers::Tuple, back, planes, window) -> Tuple

The [`FieldModifier`](@ref)s of a 2D or 3D state, in the order the mesh solve walks them
([`solve_fields!`](@ref)), from what the state placed on its grid (each `nothing` or empty where
the problem has none): the open axes' absorbers `layers` (one per open axis, in the axes' order),
the back edge's PML `back`, the antenna `planes` and the moving window's frame. Around each Yee
sub-step that is

    before:  the absorbing layers' half damping
    the Yee update
    after:   the open axes' PML corrections, the back edge's PML correction, the antennas'
             corrections, the absorbing layers' half damping, the window's zero x-guards

so a PML corrects the curl the update has just taken before a source is added to it, the
absorbing layer damps what the sub-step and its source left (symmetric splitting), and the
window's edges are set last, over whatever the others wrote into the x-guards. The order is the
tuple's: the PMLs first, the absorbing layers after the antennas (the open axes have one kind or
the other), the edges last.
"""
field_modifiers(layers::Tuple, back, planes::Tuple, window) =
    (each_first(layers)..., present(back)..., antenna_modifier(planes, window)..., each_late(layers)..., present(edges_of(window))...)

each_first(::Tuple{}) = ()
each_first(ls::Tuple) = (modifier_first(first(ls))..., each_first(Base.tail(ls))...)
each_late(::Tuple{}) = ()
each_late(ls::Tuple) = (modifier_late(first(ls))..., each_late(Base.tail(ls))...)

present(::Nothing) = ()
present(m::FieldModifier) = (m,)
modifier_first(::Any) = ()
modifier_late(::Any) = ()
antenna_modifier(::Tuple{}, window) = ()
antenna_modifier(planes::Tuple, window) = (AntennaPlanes(planes, WindowClock(window)),)
edges_of(::Nothing) = nothing

"The kinetic energy of every chunk of particle slots at the current momenta, into `st.K` (deterministic, no deposition); N ≥ 2."
function kinetic!(st::EMState)
    wgs = st.numerics.workgroupsize
    first = 1
    for s in deposited_species(st)
        nc = cld(capacity(s), KINETIC_CHUNK)
        launcher(kinetic_chunks_kernel!, get_backend(s), wgs)(st.K, s, Int32(first), Int32(capacity(s)), Int32(KINETIC_CHUNK);
                                                             ndrange = nc)
        first += nc
    end
    st.fresh[] = true
    return st
end

# What `EMProblem` requires of a moving-window plasma, of the parameters handed to the state of
# the `WindowFrame` `w`.
function check_window_params(w, p::EMParams)
    all(ps -> iszero(ps.amplitude), p.species) ||
        throw(ArgumentError("a moving-window plasma starts neutral against its immobile background, so every species' amplitude must be zero"))
    all(iszero, p.E0) && all(iszero, p.B0) ||
        throw(ArgumentError("a moving-window run takes no background fields (the outside of the window is field-free)"))
    w.window.start > 0 && !all(ps -> all(iszero, ps.drift) && all(iszero, ps.thermal), p.species) &&
        throw(ArgumentError("before a moving window starts (start = $(w.window.start)) its front is a fixed edge that a plasma with a drift or a thermal speed reaches: start the window at 0 or keep the species cold"))
    return nothing
end

# The initial field of a state of N ≥ 2 from the deposited, folded charge in `st.ρ`: the engine's
# periodic solve, `φ` from `inverse_laplacian_periodic!` and `E = G` from `gradient_periodic!` (the
# components of the resolved axes). A moving window's plasma starts neutral against its immobile
# background, so its E⁰ is zero and no solve is made.
function initial_field!(st::EMState{N}) where {N}
    window_frame(st) === nothing || return st
    initial_field!(st.fields, st.aux.φ, st.ρ, st.grid, st.aux.spectral; workgroupsize = st.numerics.workgroupsize)
    return st
end

"""
    initial_field!(fields, φ, ρ, grid::PeriodicGrid{N}, spectral; workgroupsize = nothing) -> fields

The electric field of the folded charge `ρ` on a periodic grid of N ≥ 2 axes, into the components
of the resolved axes of `fields` ([`field_layout`](@ref)`(Val(N))`; Ex and Ey at N = 2, Ex, Ey and
Ez at N = 3) with their guards filled: `φ` (scratch) by the engine's `inverse_laplacian_periodic!`
with the `SpectralWorkspace` `spectral`, then `E = G` by `gradient_periodic!`, so that the
backward-difference divergence of E is ρ − mean(ρ) at every node to roundoff, the mean being the
immobile neutralizing background.
"""
function initial_field!(fields::NamedTuple, φ, ρ, grid::PeriodicGrid{N}, spectral; workgroupsize = nothing) where {N}
    inverse_laplacian_periodic!(φ, ρ, grid, spectral; workgroupsize)
    gradient_periodic!(resolved_components((fields.Ex, fields.Ey, fields.Ez), Val(N)), φ, grid; workgroupsize)
    return fields
end

"The components of the resolved axes of a three-component field, as an `NTuple{N}`."
@inline resolved_components(F::NTuple{3,Any}, ::Val{N}) where {N} = ntuple(d -> F[d], Val(N))
