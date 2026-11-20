# The module at N = 3 (WP-042, WP-072, WP-074, WP-075): the state of a 3D problem. The step,
# `initialize!`, the kinetic energy, the initial field, the moving window and the field ionization
# are the methods over N of step.jl, step2d.jl, window.jl and ionization.jl; a 3D problem has no
# witness bunch, so its field modifiers are the absorbers of its open transverse axes, the PML of a
# moving window's back edge, the antennas of its lasers (WP-073) and the window's x edges, and
# `aux` has the window's frame and the dopant's and no witness.

"""
    EMState(problem::EMProblem{3}, numerics; backend = CPU(), every = 1, record = RECORD_GROUPS)

The state of a 3D problem ([`EMState`](@ref)): the grid of `problem.cells` cells per axis with
[`em_guard_width`](@ref) guards, the time step `problem.dt`, one species per
[`SpeciesDescription`](@ref) at capacity `prod(cells)·prod(ppc)` with the extras of
`flux_extras(T, 3)` (the old position per axis, which the push stores; with a moving window
`(1 + numerics.headroom)` times that capacity), the workspaces, the
fields of [`field_layout`](@ref)`(Val(3))`, `ρ`, the kinetic-energy chunks `K` (one entry per
[`KINETIC_CHUNK`](@ref) slots of each species; none without species), the default
[`HistoryRecorder`](@ref) for `problem.kmode` and `problem.nsteps` steps (recording every `every`
steps the groups of `record`), as field modifiers the numerics' absorber placed on each open
transverse axis (a [`PMLFields`](@ref) or a [`LayerFactors`](@ref) per open axis, axis 2 before
axis 3; none for a periodic box), the back edge's [`BackPMLFields`](@ref) of the numerics'
`back_absorber`, the [`AntennaPlanes`](@ref) of the problem's lasers and the window's
[`WindowEdges`](@ref), in the order of [`field_modifiers`](@ref), and in `aux` the spectral
workspace and the potential `φ` of the initial solve, the moving window's [`WindowFrame`](@ref)
(`window`, `nothing` without one; WP-074), the [`IonizationFrame`](@ref) of a dopant
(`ionization`, or `nothing`; WP-075: its electrons are part of `K`, after the species) and
`witness`, `nothing`.

With the current [`Esirkepov`](@ref)`()` each workspace also holds the patches of the
`Reference()` flux deposition (`Workspace(species, grid; flux = (shape, ρ's stagger))`, one
component at a time at N = 3, also used by `Fast()` on an element type without atomics). Throws an
`ArgumentError` unless the current is `Esirkepov()` or [`NoCurrent`](@ref)`()`, unless the
numerics' `absorber` is an [`AbsorbingLayer`](@ref) or a [`PML`](@ref) exactly when a transverse
axis of the problem is `:open` (each placed layer also checks its fit on its axis), a
`back_absorber` that is `nothing` or a [`PML`](@ref), and only with a moving window, a `filter` of zero
passes and `resort = false` (neither the binomial filter nor the resort is tested at N = 3 yet),
`gather = nothing` or `GalerkinGather()` (a [`CompensatedGather`](@ref), alone or in a Galerkin
gather, is not tested at N = 3), unless `problem.dt` is below the Courant limit of the numerics'
solver ([`check_courant`](@ref)), and unless with a window every pulse is over before the window
passes its antenna or the back edge's PML reaches it (as at N = 2, [`check_params`](@ref)).
"""
function EMState(problem::EMProblem{3,T}, numerics::EMNumerics; backend = CPU(), every::Integer = 1, record = RECORD_GROUPS) where {T}
    numerics.current isa Union{Esirkepov,NoCurrent} ||
        throw(ArgumentError("a 3D state takes current = Esirkepov() or NoCurrent() in EMNumerics, got $(numerics.current)"))
    open = filter(d -> problem.boundaries[d] === :open, (2, 3))
    !isempty(open) && !(numerics.absorber isa Union{AbsorbingLayer,PML}) &&
        throw(ArgumentError("axes $open of the problem are open: EMNumerics needs absorber = AbsorbingLayer(...) or PML(...), got $(numerics.absorber)"))
    isempty(open) && numerics.absorber !== nothing &&
        throw(ArgumentError("absorber = $(numerics.absorber) needs an open transverse axis in the problem (boundaries = (:periodic, :open, ·) or (:periodic, ·, :open))"))
    numerics.back_absorber === nothing || numerics.back_absorber isa PML ||
        throw(ArgumentError("back_absorber = $(numerics.back_absorber): the window's back edge takes a PML(...) or nothing"))
    problem.window === nothing && numerics.back_absorber !== nothing &&
        throw(ArgumentError("back_absorber = $(numerics.back_absorber) needs a moving window (boundaries = (:window, ·, ·))"))
    numerics.filter.passes == 0 ||
        throw(ArgumentError("filter = $(numerics.filter): the binomial filter is not yet tested at N = 3, so a 3D state takes BinomialFilter(0)"))
    numerics.resort &&
        throw(ArgumentError("resort = true: the sort from the previous order is not yet tested at N = 3, so a 3D state takes resort = false"))
    (numerics.gather isa CompensatedGather || numerics.gather isa GalerkinGather{CompensatedGather}) &&
        throw(ArgumentError("gather = $(numerics.gather): the compensated gather is not tested at N = 3, so a 3D state reads the grid's fields (gather = nothing or GalerkinGather())"))
    n = problem.cells
    wgs = numerics.workgroupsize
    grid = PeriodicGrid(n, problem.length ./ n, em_guard_width(numerics.shape))
    check_courant(numerics.solver, grid, problem.dt)
    layers = map(d -> place_absorber(numerics.absorber, grid, problem.dt; axis = d, backend, workgroupsize = wgs), open)
    window = problem.window === nothing ? nothing : WindowFrame(problem, grid, backend, numerics.shape, numerics.filter.passes)
    back = numerics.back_absorber === nothing ? nothing : BackPMLFields(numerics.back_absorber, grid, problem.dt; backend, workgroupsize = wgs)
    planes = map(l -> AntennaPlane(l.position, grid, problem.dt), problem.lasers)
    window === nothing || check_window_antennas(window, back, map(Antenna, params(problem).lasers, planes))
    ionization = problem.ionization === nothing ? nothing : IonizationFrame(problem, grid, backend, numerics, window)
    species = map(problem.species) do d
        cap = prod(n) * prod(d.ppc)
        window === nothing || (cap = ceil(Int, (1 + numerics.headroom) * cap))
        Species(backend, T, cap; q = d.charge, m = d.mass, N = 3, extras = flux_extras(T, 3))
    end
    flux = numerics.current isa Esirkepov ? (numerics.shape, rho_stagger(Val(3))) : nothing
    # a window refills one column (n₂·n₃ cells) of a species behind its count after each shift's sort
    ws = map((s, d) -> Workspace(s, grid; flux, append = sort_append(numerics, window === nothing ? 0 : n[2] * n[3] * prod(d.ppc)),
                                 bins = sort_bins(numerics, Val(3); window = window !== nothing)), species, problem.species)
    nK = sum(s -> cld(capacity(s), KINETIC_CHUNK), (species..., born_species(ionization)...); init = 0)
    recorder = HistoryRecorder(backend, T, grid, problem.length, problem.kmode, problem.nsteps; workgroupsize = wgs, every, record)
    aux = (spectral = SpectralWorkspace(backend, T, grid; workgroupsize = wgs), φ = field(backend, T, grid),
           window = window, witness = nothing, ionization = ionization)
    return EMState(numerics, grid, problem.dt, species, ws, em_fields(backend, T, grid), nothing, field(backend, T, grid),
                   KernelAbstractions.zeros(backend, T, nK), recorder, map(species_setup, problem.species),
                   field_modifiers(layers, back, planes, window), aux, Ref(false))
end
