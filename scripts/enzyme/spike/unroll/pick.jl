# Spike-only: the integer clamp of `pick` by ifelse, which lowers to plain selects (SPIRVIntrinsics maps `clamp` to an OpenCL builtin Enzyme has no rule for).
@eval ParticleMeshCore begin
    @inline function pick(w::NTuple{M,T}, d::Int32) where {M,T}
        # select without a dynamic tuple index
        return foldl(+, ntuple(m -> ifelse(d == Int32(m - 1), w[m], zero(T)), Val(M)); init = zero(T))
    end
end
