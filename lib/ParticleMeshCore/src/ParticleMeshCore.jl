"""
    ParticleMeshCore

The engine core of ParticleMesh.jl: a periodic grid with guard cells, fixed-capacity
structure-of-arrays species with module-declared extras, B-spline shapes of order 1 and 2,
gather and the deposition of a module-supplied quantity in a deterministic `Reference` mode and
an atomic `Fast` mode, the periodic fill/fold pair, a sort by cell, counter-based
Philox4x32-10 draws, quiet-start loading and the zero-mean periodic prefix sum
`cumsum_periodic!`.

The engine holds no physics (A9): a physics module composes its own step from these pieces.
All kernels are KernelAbstractions kernels, generic in the element type `T`, with static
shapes and no host-side branching on device data (`docs/plan/architecture-draft.md` §0.4);
every launch takes `workgroupsize = nothing` (the backend default), which selects the launch
shape only. Only `N = 1` is implemented and tested. The contracts are those of
`docs/plan/architecture-draft.md` §0.7.
"""
module ParticleMeshCore

using KernelAbstractions
using KernelAbstractions: @kernel, @index, get_backend
import Atomix
import Adapt
import AcceleratedKernels as AK

export PeriodicGrid, ncells, cellsize, guards, field, interior, interior_view
export Species, capacity, alive_count
export Shape, Linear, Quadratic, Node, Center, order, guard_width, shape_weights, advance
export fill_guards!, fold_guards!
export Reference, Fast, Atomic, Sorted, accumulation, effective_mode
export Workspace, sort_by_cell!, deposit!, gather!, ChargeWeight
export cumsum_periodic!
export philox4x32, uniform_open, box_muller
export load_quiet!

include("launch.jl")
include("grid.jl")
include("species.jl")
include("shapes.jl")
include("guards.jl")
include("philox.jl")
include("loading.jl")
include("sort.jl")
include("transfers.jl")
include("mesh_ops.jl")

end
