"""
    ParticleMeshCore

The engine core of ParticleMesh.jl, as built by the spike WP-010: a periodic grid with guard
cells, fixed-capacity structure-of-arrays species, B-spline shapes of order 1 and 2, gather and
scalar deposition in a deterministic `Reference` mode and an atomic `Fast` mode, the periodic
fill/fold pair, a sort by cell, counter-based Philox4x32-10 draws and quiet-start loading.

All kernels are KernelAbstractions kernels, generic in the element type `T`, with static
shapes and no host-side branching on device data (`docs/plan/architecture-draft.md` §0.4).
Only `N = 1` is implemented and tested; names and signatures are the spike's, not contracts.

The electrostatic physics lives in the submodule [`Electrostatic`](@ref), so that the
engine/module split can later be refactored from working code.
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
export Workspace, sort_by_cell!, deposit!, gather!
export philox4x32, uniform_open, box_muller
export load_quiet!

include("grid.jl")
include("species.jl")
include("shapes.jl")
include("guards.jl")
include("philox.jl")
include("loading.jl")
include("sort.jl")
include("transfers.jl")

include("Electrostatic/Electrostatic.jl")

end
