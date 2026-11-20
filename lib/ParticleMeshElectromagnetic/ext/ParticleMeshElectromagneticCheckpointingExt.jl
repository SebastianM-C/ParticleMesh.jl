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

# The `EMState`s a loop body holds, found by walking its fields (a closure holds its captured
# variables as fields, a `Ref` or `Core.Box` its value), the entries of the arrays whose element
# type can be a state, and tuples, depth first in field order, each state once. A state's own
# fields are not entered, nor arrays of other element types (the device arrays), modules or types.
# A walk deeper than `STATE_DEPTH` levels is refused rather than cut short, so that no state the
# body holds goes unsaved.
const STATE_DEPTH = 32
function find_states!(out, seen, x, depth)
    if x isa EMState
        any(y -> y === x, out) || push!(out, x)
        return out
    end
    (isbits(x) || x isa Union{Module,Type,AbstractString,Symbol}) && return out
    if ismutable(x)
        x in seen && return out
        push!(seen, x)
    end
    depth >= STATE_DEPTH &&
        throw(ArgumentError("checkpoint_storage: the loop body nests its values more than $STATE_DEPTH levels deep, below which it could hold an EMState that would not be saved"))
    if x isa AbstractArray
        typeintersect(eltype(x), EMState) === Union{} && return out
        for k in eachindex(x)
            isassigned(x, k) && find_states!(out, seen, x[k], depth + 1)
        end
        return out
    end
    for k in 1:nfields(x)
        isdefined(x, k) && find_states!(out, seen, getfield(x, k), depth + 1)
    end
    return out
end

"The `EMState`s that `body` holds, each once, in the order they are first reached (a tuple; empty when it holds none)."
body_states(body) = Tuple(find_states!(EMState[], Base.IdSet{Any}(), body, 0))

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
