# The module at N = 3 (WP-042): the state of a periodic 3D problem. The step, `initialize!`, the
# kinetic energy and the initial field are the methods over N of step.jl and step2d.jl; a 3D
# problem has no laser, open axis, window, witness bunch or dopant, so its state has no field
# modifier and none of their frames in `aux`.

"""
    EMState(problem::EMProblem{3}, numerics; backend = CPU(), every = 1, record = RECORD_GROUPS)

The state of a 3D problem ([`EMState`](@ref)): the grid of `problem.cells` cells per axis with
[`em_guard_width`](@ref) guards, the time step `problem.dt`, one species per
[`SpeciesDescription`](@ref) at capacity `prod(cells)·prod(ppc)` with the extras of
`flux_extras(T, 3)` (the old position per axis, which the push stores), the workspaces, the
fields of [`field_layout`](@ref)`(Val(3))`, `ρ`, the kinetic-energy chunks `K` (one entry per
[`KINETIC_CHUNK`](@ref) slots of each species; none without species), the default
[`HistoryRecorder`](@ref) for `problem.kmode` and `problem.nsteps` steps (recording every `every`
steps the groups of `record`), no field modifiers, and in `aux` the spectral workspace and the
potential `φ` of the initial solve, with `window`, `witness` and `ionization` `nothing`.

With the current [`Esirkepov`](@ref)`()` each workspace also holds the patches of the
`Reference()` flux deposition (`Workspace(species, grid; flux = (shape, ρ's stagger))`, one
component at a time at N = 3, also used by `Fast()` on an element type without atomics). Throws an
`ArgumentError` unless the current is `Esirkepov()` or [`NoCurrent`](@ref)`()`, unless the
numerics have no `absorber` and no `back_absorber` (the box is periodic), a `filter` of zero
passes and `resort = false` (neither the binomial filter nor the resort is tested at N = 3 yet),
`gather = nothing` or `GalerkinGather()` (a [`CompensatedGather`](@ref), alone or in a Galerkin
gather, sharpens along a laser axis), and unless
`problem.dt` is below the Courant limit of the numerics' solver ([`check_courant`](@ref)).
"""
function EMState(problem::EMProblem{3,T}, numerics::EMNumerics; backend = CPU(), every::Integer = 1, record = RECORD_GROUPS) where {T}
    numerics.current isa Union{Esirkepov,NoCurrent} ||
        throw(ArgumentError("a 3D state takes current = Esirkepov() or NoCurrent() in EMNumerics, got $(numerics.current)"))
    numerics.absorber === nothing && numerics.back_absorber === nothing ||
        throw(ArgumentError("the 3D box is periodic: EMNumerics takes no absorber and no back_absorber at N = 3"))
    numerics.filter.passes == 0 ||
        throw(ArgumentError("filter = $(numerics.filter): the binomial filter is not yet tested at N = 3, so a 3D state takes BinomialFilter(0)"))
    numerics.resort &&
        throw(ArgumentError("resort = true: the sort from the previous order is not yet tested at N = 3, so a 3D state takes resort = false"))
    (numerics.gather isa CompensatedGather || numerics.gather isa GalerkinGather{CompensatedGather}) &&
        throw(ArgumentError("gather = $(numerics.gather): the compensated gather sharpens along a laser axis, which the periodic 3D box has not, so a 3D state reads the grid's fields (gather = nothing or GalerkinGather())"))
    n = problem.cells
    wgs = numerics.workgroupsize
    grid = PeriodicGrid(n, problem.length ./ n, em_guard_width(numerics.shape))
    check_courant(numerics.solver, grid, problem.dt)
    species = map(d -> Species(backend, T, prod(n) * prod(d.ppc); q = d.charge, m = d.mass, N = 3, extras = flux_extras(T, 3)),
                  problem.species)
    flux = numerics.current isa Esirkepov ? (numerics.shape, rho_stagger(Val(3))) : nothing
    ws = map(s -> Workspace(s, grid; flux, bins = sort_bins(numerics, Val(3))), species)
    nK = sum(s -> cld(capacity(s), KINETIC_CHUNK), species; init = 0)
    recorder = HistoryRecorder(backend, T, grid, problem.length, problem.kmode, problem.nsteps; workgroupsize = wgs, every, record)
    aux = (spectral = SpectralWorkspace(backend, T, grid; workgroupsize = wgs), φ = field(backend, T, grid),
           window = nothing, witness = nothing, ionization = nothing)
    return EMState(numerics, grid, problem.dt, species, ws, em_fields(backend, T, grid), nothing, field(backend, T, grid),
                   KernelAbstractions.zeros(backend, T, nK), recorder, map(species_setup, problem.species), (), aux, Ref(false))
end
