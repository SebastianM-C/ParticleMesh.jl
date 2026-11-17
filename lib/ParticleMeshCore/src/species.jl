"""
    Species{N,T}

A fixed-capacity structure of arrays for one particle species:

- `cell::NTuple{N}` of `Int32` arrays: the cell index per resolved axis (`1:n`);
- `offset::NTuple{N}` of `T` arrays: the in-cell offset in `[0, 1)`, in cell units;
- `u::NTuple{3}` of `T` arrays: three components of whatever the module says they hold
  (velocities in units of c in the electrostatic reference, momenta in the electromagnetic
  module); the engine only moves them;
- `w`: the weight (physical density per macro-particle in units of n_c, A3);
- `status::UInt8` array: `0x01` alive, `0x00` free;
- `count`: a one-element `Int32` device array with the number of alive particles after the last
  sort (a high-water mark otherwise);
- `q`, `m`: per-species charge and mass in units of e and mₑ;
- `extras::NamedTuple`: module-declared per-particle arrays at capacity (for example the old
  cell and offset that a charge-conserving deposition needs), each of its own element type,
  permuted by `sort_by_cell!` together with the core arrays. Empty by default.

No `push!` or `resize!`: the capacity is fixed at construction (§0.4 rule 1).

The per-particle arrays are read as properties (`s.u[1]`) and held in a [`SpeciesArrays`](@ref):
a species built by `Species(backend, T, capacity; ...)` owns them, and `sort_by_cell!` may swap
them for a workspace's spare set (WP-041), so an array read from it is the species' until its
next sort; `count` is never swapped.
"""
struct Species{N,T,VI<:AbstractVector{Int32},VT<:AbstractVector{T},VS<:AbstractVector{UInt8},VC<:AbstractVector{Int32},X<:NamedTuple,D}
    data::D
    count::VC
    q::T
    m::T
    owns::Bool
end

"""
    SpeciesArrays

The per-particle arrays of a [`Species`](@ref): `cell`, `offset`, `u`, `w`, `status` and
`extras`, as `Species` lists them. A species reads them through its properties (`s.u[1]`), so no
caller names this type; it exists so that `sort_by_cell!` can swap a species' arrays with a
workspace's spare set as a whole (WP-041).
"""
struct SpeciesArrays{N,VI,VT,VS,X<:NamedTuple}
    cell::NTuple{N,VI}
    offset::NTuple{N,VT}
    u::NTuple{3,VT}
    w::VT
    status::VS
    extras::X
end

Adapt.@adapt_structure SpeciesArrays

# A species holds its arrays in a `Base.RefValue` on the host, which `sort_by_cell!` may point
# at another set of arrays of the same types when the species owns them (`owns`: built by
# `Species(backend, T, capacity; ...)`; not a species over arrays that belong to someone else,
# `Species(cell, offset, u, w, status, count, q, m, extras)`), and as they are on a device, where
# `Adapt` resolves the reference before a kernel launches. Both kinds have one host type.
function Species(data::Union{SpeciesArrays{N,VI,VT,VS,X},Base.RefValue{SpeciesArrays{N,VI,VT,VS,X}}}, count::VC, q, m,
                 owns::Bool) where {N,VI,VT,VS,X,VC}
    T = eltype(VT)
    return Species{N,T,VI,VT,VS,VC,X,typeof(data)}(data, count, T(q), T(m), owns)
end

"""
    Species(cell, offset, u, w, status, count, q, m, extras)

A species over the given arrays, which it does not own: a sort writes its permutation back into
these arrays (the copy-back form of [`sort_by_cell!`](@ref)) and never swaps them for others. For
a view of arrays that belong elsewhere (an integrator's state, a saved state).
"""
Species(cell::NTuple{N}, offset::NTuple{N}, u::NTuple{3}, w, status, count, q, m, extras::NamedTuple) where {N} =
    Species(Ref(SpeciesArrays(cell, offset, u, w, status, extras)), count, q, m, false)

@inline species_data(s::Species) = _data(getfield(s, :data))
@inline _data(r::Base.RefValue) = r[]
@inline _data(a::SpeciesArrays) = a

"Whether `sort_by_cell!` may swap the species' arrays for a workspace's spare set: a species that owns them."
owns_arrays(s::Species) = getfield(s, :owns)

# `s.cell`, `s.offset`, `s.u`, `s.w`, `s.status`, `s.extras` read the species' current arrays; on a
# device the symbol is a constant and the lookup folds away.
@inline function Base.getproperty(s::Species, f::Symbol)
    (f === :count || f === :q || f === :m || f === :data || f === :owns) && return getfield(s, f)
    return getfield(species_data(s), f)
end
Base.propertynames(::Species) = (:cell, :offset, :u, :w, :status, :count, :q, :m, :extras)

# The device form holds the arrays themselves (isbits for a kernel argument).
Adapt.adapt_structure(to, s::Species) = Species(Adapt.adapt(to, species_data(s)), Adapt.adapt(to, s.count), s.q, s.m, s.owns)

"""
    Species(backend, T, capacity; q, m, N = 1, extras = NamedTuple())

An empty species (every slot free) of the given capacity. `extras` names the module's
per-particle arrays and their element types, `(old_cell = Int32, old_offset = T)`; each is
allocated zeroed at capacity on `backend`. The species owns its arrays: a workspace built with
`swap` (the default) sorts it into its spare arrays and swaps the two sets, so an array read
from the species (`s.u[1]`) is the species' until its next sort, and is the workspace's scratch
after it.
"""
function Species(backend, ::Type{T}, cap::Integer; q, m, N::Integer = 1, extras::NamedTuple = NamedTuple()) where {T}
    zT() = KernelAbstractions.zeros(backend, T, cap)
    cell = ntuple(_ -> KernelAbstractions.ones(backend, Int32, cap), N)
    offset = ntuple(_ -> zT(), N)
    u = ntuple(_ -> zT(), 3)
    status = KernelAbstractions.zeros(backend, UInt8, cap)
    count = KernelAbstractions.zeros(backend, Int32, 1)
    x = map(E -> KernelAbstractions.zeros(backend, E, cap), extras)
    return Species(Ref(SpeciesArrays(cell, offset, u, zT(), status, x)), count, q, m, true)
end

capacity(s::Species) = length(s.w)
Base.eltype(::Species{N,T}) where {N,T} = T
KernelAbstractions.get_backend(s::Species) = get_backend(s.w)

"Host copy of the alive count (a device read; never call it inside a step)."
alive_count(s::Species) = Int(Array(s.count)[1])

const ALIVE = 0x01
const FREE = 0x00
