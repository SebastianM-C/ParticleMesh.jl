bits(a::AbstractArray{Float32}) = reinterpret(UInt32, a)
bits(a::AbstractArray{Float64}) = reinterpret(UInt64, a)
bits(a::AbstractArray{<:Integer}) = a

"Bitwise digest of an electrostatic state: every SoA array of every species and the histories."
state_digest(st) = hash((map(s -> map(bits, (s.cell[1], s.offset[1], s.u[1], s.u[2], s.u[3], s.w, s.status, s.count)), st.species),
                         bits(st.ρ), bits(st.Ehalf), bits(st.W), bits(st.Ere), bits(st.Eim)))

"""
    fnv_digest(arrays...)

64-bit FNV-1a over the bytes of the arrays in order. Unlike `hash`, it is fixed by its
definition, so a recorded value stays valid across Julia versions (T-170).
"""
function fnv_digest(arrays::AbstractArray...)
    h = 0xcbf29ce484222325
    for a in arrays, b in reinterpret(UInt8, vec(collect(a)))
        h = (h ⊻ UInt64(b)) * 0x00000100000001b3
    end
    return h
end

"FNV digest of an electrostatic state: every species array, then ρ, Ehalf, Enode and the histories."
fnv_state_digest(st) = fnv_digest((a for s in st.species for a in (s.cell[1], s.offset[1], s.u[1], s.u[2], s.u[3], s.w, s.status, s.count))...,
                                  st.ρ, st.Ehalf, st.Enode, st.W, st.Ere, st.Eim)
