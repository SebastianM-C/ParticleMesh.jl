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
