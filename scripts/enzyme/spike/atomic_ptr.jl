# Spike-only: the 2D flux kernel's atomic adds through the array's pointer (UnsafeAtomics, which
# Atomix calls underneath) instead of `Atomix.@atomic J[...]`. Atomix's `modify!` holds the array in
# `GC.@preserve`, which needs the by-value device array in memory; Enzyme's augmented forward then
# moves that stack slot to the device heap (one `malloc` of the 40-byte descriptor per array, primal
# and shadow, per work item and step, seen on CUDA). A device array needs no GC root inside a kernel.
@eval ParticleMeshCore begin
    @inline function atomic_add_block!(J, v::NTuple{K,Any}, b1::Int, b2::Int, ::Val{M1}) where {K,M1}
        n1 = size(J, 1)
        ntuple(Val(K)) do k
            @inline
            i = b1 + (k - 1) % M1 + (b2 + (k - 1) ÷ M1 - 1) * n1
            Atomix.Internal.UnsafeAtomics.modify!(pointer(J, i), +, convert(eltype(J), v[k]), Atomix.Internal.UnsafeAtomics.monotonic,
                                         Atomix.Internal.UnsafeAtomics.device)
            nothing
        end
        return nothing
    end
end
