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
# Spike-only: prefix sums written out by a generated function (the same left-to-right bits), no inner loop.
@eval ParticleMeshCore begin
    @generated function prefix_sums(t::NTuple{M,T}) where {M,T}
        a = [Symbol(:a, k) for k in 1:M]
        body = [:($(a[1]) = t[1])]
        for k in 2:M
            push!(body, :($(a[k]) = $(a[k - 1]) + t[$k]))
        end
        return quote
            Base.@_inline_meta
            $(body...)
            return ($(a...),)
        end
    end
end
# Spike-only: the integer clamp of `pick` by ifelse, which lowers to plain selects (SPIRVIntrinsics maps `clamp` to an OpenCL builtin Enzyme has no rule for).
@eval ParticleMeshCore begin
    @inline function pick(w::NTuple{M,T}, d::Int32) where {M,T}
        # select without a dynamic tuple index
        return foldl(+, ntuple(m -> ifelse(d == Int32(m - 1), w[m], zero(T)), Val(M)); init = zero(T))
    end
end
# Spike-only: γ through the LLVM sqrt intrinsic; on PoCL `sqrt` lowers to an OpenCL builtin Enzyme has no derivative for.
@eval ParticleMeshElectromagnetic begin
    @inline lorentz_factor(u1::T, u2::T, u3::T) where {T<:Union{Float32,Float64}} = Base.Math.sqrt_llvm(one(T) + u1 * u1 + u2 * u2 + u3 * u3)
end
