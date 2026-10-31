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
"""
struct Species{N,T,VI<:AbstractVector{Int32},VT<:AbstractVector{T},VS<:AbstractVector{UInt8},VC<:AbstractVector{Int32},X<:NamedTuple}
    cell::NTuple{N,VI}
    offset::NTuple{N,VT}
    u::NTuple{3,VT}
    w::VT
    status::VS
    count::VC
    q::T
    m::T
    extras::X
end

Adapt.@adapt_structure Species

"""
    Species(backend, T, capacity; q, m, N = 1, extras = NamedTuple())

An empty species (every slot free) of the given capacity. `extras` names the module's
per-particle arrays and their element types, `(old_cell = Int32, old_offset = T)`; each is
allocated zeroed at capacity on `backend`.
"""
function Species(backend, ::Type{T}, cap::Integer; q, m, N::Integer = 1, extras::NamedTuple = NamedTuple()) where {T}
    zT() = KernelAbstractions.zeros(backend, T, cap)
    cell = ntuple(_ -> KernelAbstractions.ones(backend, Int32, cap), N)
    offset = ntuple(_ -> zT(), N)
    u = ntuple(_ -> zT(), 3)
    status = KernelAbstractions.zeros(backend, UInt8, cap)
    count = KernelAbstractions.zeros(backend, Int32, 1)
    x = map(E -> KernelAbstractions.zeros(backend, E, cap), extras)
    return Species{N,T,typeof(cell[1]),typeof(offset[1]),typeof(status),typeof(count),typeof(x)}(
        cell, offset, u, zT(), status, count, T(q), T(m), x)
end

capacity(s::Species) = length(s.w)
Base.eltype(::Species{N,T}) where {N,T} = T
KernelAbstractions.get_backend(s::Species) = get_backend(s.w)

"Host copy of the alive count (a device read; never call it inside a step)."
alive_count(s::Species) = Int(Array(s.count)[1])

const ALIVE = 0x01
const FREE = 0x00
