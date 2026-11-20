"""
    ParticleMeshElectromagneticCheckpointingExt

A storage backend for Checkpointing.jl's schemes over loops that step `EMState`s (WP-080):
`EvolvedStorage`, which `checkpoint_storage` returns. It implements Checkpointing.jl's storage
interface (`save!`, `load!`, `similar`, `length`, as its own `ArrayStorage` and `HDF5Storage` do):
a checkpoint is a copy of the `evolved` arrays of every state the loop body captures, allocated at
the first store into a slot and copied into after that, and a restore is `restore!`.
"""
module ParticleMeshElectromagneticCheckpointingExt

using ParticleMeshElectromagnetic
using ParticleMeshElectromagnetic: EMState, evolved, restore!, copy_arrays!
import Checkpointing
using Checkpointing: AbstractStorage

"""
    EvolvedStorage{FT,H}(checkpoints)

Checkpoints of the `evolved` arrays of the states a loop body of type `FT` captures, `checkpoints`
slots, on the states' backend (`H = false`) or in host memory (`H = true`). Each slot is `nothing`
until the first store into it, then a tuple (one entry per captured state, in field order) of
tuples of arrays in `evolved`'s order.
"""
struct EvolvedStorage{FT,H} <: AbstractStorage
    slots::Vector{Any}
end
EvolvedStorage{FT,H}(checkpoints::Integer) where {FT,H} = EvolvedStorage{FT,H}(Vector{Any}(nothing, checkpoints))

Base.similar(s::EvolvedStorage{<:Any,H}, ::Type{FT}) where {FT,H} = EvolvedStorage{FT,H}(length(s.slots))
Base.ndims(::Type{<:EvolvedStorage}) = 1
Base.size(s::EvolvedStorage) = size(s.slots)
Base.length(s::EvolvedStorage) = length(s.slots)

ParticleMeshElectromagnetic.checkpoint_storage(; host::Bool = false) = EvolvedStorage{FT,host} where {FT}

# The paths (field indices) from a value of type `T` to the `EMState`s it holds, through fields of
# structs and tuples, a few levels deep (a closure holds its captured variables as fields; a
# `Ref` its value). Arrays, modules and types are not entered, nor a state's own fields.
const STATE_DEPTH = 4
state_paths(::Type{<:EMState}, path, depth) = (path,)
function state_paths(::Type{T}, path, depth) where {T}
    (depth >= STATE_DEPTH || !isconcretetype(T) || !isstructtype(T) || T <: AbstractArray || T <: Module || T <: Type ||
     T <: AbstractString || T <: Symbol) && return ()
    return Tuple(Iterators.flatten(state_paths(fieldtype(T, i), (path..., i), depth + 1) for i in 1:fieldcount(T)))
end

"The `EMState`s that `body` holds, in field order (a tuple; empty when it holds none)."
@generated function body_states(body)
    paths = state_paths(body, (), 0)
    gets = [foldl((ex, i) -> :(getfield($ex, $i)), p; init = :body) for p in paths]
    return :(($(gets...),))
end

function states_of(body)
    sts = body_states(body)
    isempty(sts) &&
        throw(ArgumentError("checkpoint_storage: the loop body $(typeof(body)) holds no EMState; its checkpoints are the evolved arrays of the states it steps"))
    return sts
end

save_copy(a, ::Val{false}) = copy(a)
save_copy(a, ::Val{true}) = Array(a)

function Checkpointing.save!(s::EvolvedStorage{FT,H}, body::FT, i::Integer) where {FT,H}
    checkbounds(s.slots, i)
    sts = states_of(body)
    slot = s.slots[i]
    if slot === nothing
        s.slots[i] = map(st -> map(a -> save_copy(a, Val(H)), evolved(st)), sts)
    else
        foreach((dst, st) -> copy_arrays!(dst, evolved(st)), slot, sts)
    end
    return s
end

function Checkpointing.load!(body::FT, s::EvolvedStorage{FT}, i::Integer) where {FT}
    checkbounds(s.slots, i)
    slot = s.slots[i]
    slot === nothing && throw(ArgumentError("checkpoint_storage: checkpoint $i was restored before it was stored"))
    foreach(restore!, states_of(body), slot)
    return body
end

end
