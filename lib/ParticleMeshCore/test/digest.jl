bits(a::AbstractArray{Float32}) = reinterpret(UInt32, a)
bits(a::AbstractArray{Float64}) = reinterpret(UInt64, a)
bits(a::AbstractArray{<:Integer}) = a

"Bitwise digest of an electrostatic state: every SoA array of every species and the histories."
state_digest(st) = hash((map(s -> map(bits, (s.cell[1], s.offset[1], s.u[1], s.u[2], s.u[3], s.w, s.status, s.count)), st.species),
                         bits(st.ρ), bits(st.Ehalf), bits(st.W), bits(st.Ere), bits(st.Eim)))
