"""
    Species{N,T}

A fixed-capacity structure of arrays for one particle species:

- `cell::NTuple{N}` of `Int32` arrays: the cell index per resolved axis (`1:n`);
- `offset::NTuple{N}` of `T` arrays: the in-cell offset in `[0, 1)`, in cell units;
- `u::NTuple{3}` of `T` arrays: three velocity components (the module decides whether they are
  velocities or momenta; the electrostatic module stores velocities in units of c);
- `w`: the weight (physical density per macro-particle in units of n_c, A3);
- `status::UInt8` array: `0x01` alive, `0x00` free;
- `count`: a one-element `Int32` device array with the number of alive particles after the last
  sort (a high-water mark otherwise);
- `q`, `m`: per-species charge and mass in units of e and mₑ.

No `push!` or `resize!`: the capacity is fixed at construction (§0.4 rule 1).
"""
struct Species{N,T,VI<:AbstractVector{Int32},VT<:AbstractVector{T},VS<:AbstractVector{UInt8},VC<:AbstractVector{Int32}}
    cell::NTuple{N,VI}
    offset::NTuple{N,VT}
    u::NTuple{3,VT}
    w::VT
    status::VS
    count::VC
    q::T
    m::T
end

Adapt.@adapt_structure Species

"""
    Species(backend, T, capacity; q, m, N = 1)

An empty species (every slot free) of the given capacity.
"""
function Species(backend, ::Type{T}, cap::Integer; q, m, N::Integer = 1) where {T}
    zT() = KernelAbstractions.zeros(backend, T, cap)
    cell = ntuple(_ -> KernelAbstractions.ones(backend, Int32, cap), N)
    offset = ntuple(_ -> zT(), N)
    u = ntuple(_ -> zT(), 3)
    status = KernelAbstractions.zeros(backend, UInt8, cap)
    count = KernelAbstractions.zeros(backend, Int32, 1)
    return Species{N,T,typeof(cell[1]),typeof(offset[1]),typeof(status),typeof(count)}(
        cell, offset, u, zT(), status, count, T(q), T(m))
end

capacity(s::Species) = length(s.w)
Base.eltype(::Species{N,T}) where {N,T} = T
KernelAbstractions.get_backend(s::Species) = get_backend(s.w)

"Host copy of the alive count (a device read; never call it inside a step)."
alive_count(s::Species) = Int(Array(s.count)[1])

const ALIVE = 0x01
const FREE = 0x00
