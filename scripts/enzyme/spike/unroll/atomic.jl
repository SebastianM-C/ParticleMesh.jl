# Spike-only: the flux kernel's atomic block add unrolled.
@eval ParticleMeshCore begin
    @inline function atomic_add_block!(J, v::NTuple{K,Any}, b1::Int, b2::Int, ::Val{M1}) where {K,M1}
        ntuple(Val(K)) do k
            Base.@_inline_meta
            @inbounds Atomix.@atomic :monotonic J[CartesianIndex(b1 + (k - 1) % M1, b2 + (k - 1) ÷ M1)] += v[k]
            nothing
        end
        return nothing
    end
end
