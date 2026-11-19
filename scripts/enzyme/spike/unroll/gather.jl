# Spike-only: the gather's stencil loop fully unrolled (P+1)^N terms, so that Enzyme's tape has a static size.
@eval ParticleMeshCore begin
    @inline function gather_at(a, c::NTuple{N,Int32}, δ::NTuple{N,T}, shape::Shape{P}, stagger::NTuple{N,Stagger},
                               g::Int32) where {N,T,P}
        fw = shape_weights(shape, stagger, δ)
        terms = ntuple(Val((P + 1)^N)) do L
            Base.@_inline_meta
            k = Tuple(CartesianIndices(ntuple(_ -> Base.OneTo(P + 1), Val(N)))[L])
            idx = ntuple(d -> c[d] + fw[d][1] + g + Int32(k[d] - 1), Val(N))
            @inbounds tensor_weight(fw, k) * a[idx...]
        end
        return foldl(+, terms; init = zero(T))
    end
end
