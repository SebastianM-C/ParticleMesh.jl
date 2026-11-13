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
    EMState(problem::EMProblem{2}, numerics; backend = CPU())

The state of a 2D problem ([`EMState`](@ref)): the grid of `problem.cells` cells per axis with
[`em_guard_width`](@ref) guards, the time step `problem.dt`, one species per
[`SpeciesDescription`](@ref) at capacity `prod(cells)·prod(ppc)` with the extras of
`flux_extras(T, 2)` (the old position per axis, which the push stores), the workspaces, the fields
of [`field_layout`](@ref)`(Val(2))`, `ρ`, the kinetic-energy chunks `K` (one entry per
[`KINETIC_CHUNK`](@ref) slots of each species; none without species), the default [`HistoryRecorder`](@ref) for
`problem.kmode` and `problem.nsteps` steps, the field modifiers of the mesh solve
([`field_modifiers`](@ref): the absorber placed on the grid, the absorbing layer's
[`LayerFactors`](@ref) or the PML's [`PMLFields`](@ref); the back edge's [`BackPMLFields`](@ref);
the [`AntennaPlanes`](@ref) of the problem's lasers, where each plane lies on the grid, not the
laser; the window's [`WindowEdges`](@ref)), and in `aux`: the spectral workspace and the potential
`φ` of the initial solve, the moving window's [`WindowFrame`](@ref) (`window`, `nothing` without one; its
species buffers then hold `(1 + numerics.headroom)` times the window's particles) and the
[`WitnessFrame`](@ref) of a witness bunch (`witness`, or `nothing`).

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
function EMState(problem::EMProblem{2,T}, numerics::EMNumerics; backend = CPU()) where {T}
    numerics.current isa Union{Esirkepov,NoCurrent} ||
        throw(ArgumentError("a 2D state takes current = Esirkepov() or NoCurrent() in EMNumerics, got $(numerics.current)"))
    n = problem.cells
    wgs = numerics.workgroupsize
    grid = PeriodicGrid(n, problem.length ./ n, em_guard_width(numerics.shape))
    check_courant(numerics.solver, grid, problem.dt)
    open_y = problem.boundaries[2] === :open
    open_y && !(numerics.absorber isa Union{AbsorbingLayer,PML}) &&
        throw(ArgumentError("axis 2 of the problem is open: EMNumerics needs absorber = AbsorbingLayer(...) or PML(...), got $(numerics.absorber)"))
    !open_y && numerics.absorber !== nothing &&
        throw(ArgumentError("absorber = $(numerics.absorber) needs an open axis 2 in the problem (boundaries = (:periodic, :open))"))
    open_y && numerics.filter.passes > 0 &&
        throw(ArgumentError("filter = $(numerics.filter) with absorber = $(numerics.absorber): the binomial filter is not yet tested with an open boundary (it wraps J across the open edge), so an open axis 2 takes BinomialFilter(0)"))
    layer = open_y ? place_absorber(numerics.absorber, grid, problem.dt; backend, workgroupsize = numerics.workgroupsize) : nothing
    numerics.back_absorber === nothing || numerics.back_absorber isa PML ||
        throw(ArgumentError("back_absorber = $(numerics.back_absorber): the window's back edge takes a PML(...) or nothing"))
    problem.window === nothing && numerics.back_absorber !== nothing &&
        throw(ArgumentError("back_absorber = $(numerics.back_absorber) needs a moving window (boundaries = (:window, ·))"))
    window = problem.window === nothing ? nothing : WindowFrame(problem, grid, backend, numerics.shape, numerics.filter.passes)
    back = numerics.back_absorber === nothing ? nothing : BackPMLFields(numerics.back_absorber, grid, problem.dt; backend, workgroupsize = wgs)
    planes = map(l -> AntennaPlane(l.position, grid, problem.dt), problem.lasers)
    window === nothing || check_window_antennas(window, back, map(Antenna, params(problem).lasers, planes))
    witness = problem.witness === nothing ? nothing : WitnessFrame(problem, grid, backend)
    species = map(problem.species) do d
        cap = prod(n) * prod(d.ppc)
        window === nothing || (cap = ceil(Int, (1 + numerics.headroom) * cap))
        Species(backend, T, cap; q = d.charge, m = d.mass, N = 2, extras = flux_extras(T, 2))
    end
    flux = numerics.current isa Esirkepov ? (numerics.shape, RHO_STAGGER_2D) : nothing
    ws = map(s -> Workspace(s, grid; flux), species)
    nK = sum(s -> cld(capacity(s), KINETIC_CHUNK), species; init = 0)
    recorder = HistoryRecorder(backend, T, grid, problem.length, problem.kmode, problem.nsteps; workgroupsize = wgs)
    aux = (spectral = SpectralWorkspace(backend, T, grid; workgroupsize = wgs), φ = field(backend, T, grid),
           window = window, witness = witness)
    return EMState(numerics, grid, problem.dt, species, ws, em_fields(backend, T, grid), field(backend, T, grid),
                   KernelAbstractions.zeros(backend, T, nK), recorder, map(species_setup, problem.species),
                   field_modifiers(layer, back, planes, window), aux)
end

"""
    field_modifiers(layer, back, planes, window) -> Tuple

The [`FieldModifier`](@ref)s of a 2D state, in the order the mesh solve walks them
([`solve_fields!`](@ref)), from what the state placed on its grid (each `nothing` or empty where
the problem has none): the open axis' absorber `layer`, the back edge's PML `back`, the antenna
`planes` and the moving window's frame. Around each Yee sub-step that is

    before:  the absorbing layer's half damping
    the Yee update
    after:   the open axis' PML correction, the back edge's PML correction, the antennas'
             corrections, the absorbing layer's half damping, the window's zero x-guards

so a PML corrects the curl the update has just taken before a source is added to it, the
absorbing layer damps what the sub-step and its source left (symmetric splitting), and the
window's edges are set last, over whatever the others wrote into the x-guards. The order is the
tuple's: a PML first, an absorbing layer after the antennas (an open axis has one or the other),
the edges last.
"""
field_modifiers(layer, back, planes::Tuple, window) =
    (modifier_first(layer)..., present(back)..., antenna_modifier(planes, window)..., modifier_late(layer)..., present(edges_of(window))...)

present(::Nothing) = ()
present(m::FieldModifier) = (m,)
modifier_first(::Any) = ()
modifier_late(::Any) = ()
antenna_modifier(::Tuple{}, window) = ()
antenna_modifier(planes::Tuple, window) = (AntennaPlanes(planes, WindowClock(window)),)
edges_of(::Nothing) = nothing

"The kinetic energy of every chunk of particle slots at the current momenta, into `st.K` (deterministic, no deposition)."
function kinetic!(st::EMState{2})
    wgs = st.numerics.workgroupsize
    first = 1
    for s in st.species
        nc = cld(capacity(s), KINETIC_CHUNK)
        launcher(kinetic_chunks_kernel!, get_backend(s), wgs)(st.K, s, Int32(first), Int32(capacity(s)), Int32(KINETIC_CHUNK);
                                                             ndrange = nc)
        first += nc
    end
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

# The initial field of a 2D state from the deposited, folded charge in `st.ρ`: the engine's periodic
# solve, `φ` from `inverse_laplacian_periodic!` and `E = G` from `gradient_periodic!`. A moving
# window's plasma starts neutral against its immobile background, so its E⁰ is zero and no solve
# is made.
function initial_field!(st::EMState{2})
    window_frame(st) === nothing || return st
    wgs = st.numerics.workgroupsize
    inverse_laplacian_periodic!(st.aux.φ, st.ρ, st.grid, st.aux.spectral; workgroupsize = wgs)
    gradient_periodic!((st.fields.Ex, st.fields.Ey), st.aux.φ, st.grid; workgroupsize = wgs)
    return st
end
