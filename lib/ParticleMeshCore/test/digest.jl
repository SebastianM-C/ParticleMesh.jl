bits(a::AbstractArray{Float32}) = reinterpret(UInt32, a)
bits(a::AbstractArray{Float64}) = reinterpret(UInt64, a)
bits(a::AbstractArray{<:Integer}) = a

"""
    fnv_digest(arrays...)

64-bit FNV-1a over the bytes of the arrays in order. Unlike `hash`, it is fixed by its
definition, so a recorded value stays valid across Julia versions (T-260).
"""
function fnv_digest(arrays::AbstractArray...)
    h = 0xcbf29ce484222325
    for a in arrays, b in reinterpret(UInt8, vec(collect(a)))
        h = (h ⊻ UInt64(b)) * 0x00000100000001b3
    end
    return h
end
