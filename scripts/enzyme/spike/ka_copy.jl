# WP-080 spike patch (from WP-078's unrolled_species.jl): `binomial!` with its device-to-device
# `copyto!` (compiled even with no filter passes) replaced by a KernelAbstractions copy kernel;
# WP-078 traced Enzyme compile-time segfaults of the 2D reverse pass to such copies. Same bits.
using ParticleMeshElectromagnetic, KernelAbstractions
@eval ParticleMeshElectromagnetic begin
    @kernel function ka_copy_kernel!(dst, @Const(src))
        i = @index(Global, Linear)
        @inbounds dst[i] = src[i]
    end
    ka_copy!(dst, src) = (ka_copy_kernel!(get_backend(dst))(dst, src; ndrange = length(dst)); dst)
    function binomial!(a, scratch, grid::PeriodicGrid{N}, passes::Integer; workgroupsize = nothing, edges = nothing) where {N}
        passes == 0 && return a
        size(scratch) == size(a) || throw(DimensionMismatch("scratch of size $(size(scratch)) for a field of size $(size(a))"))
        src, dst = a, scratch
        launch = launcher(binomial_pass_kernel!, get_backend(a), workgroupsize)
        for _ in 1:passes, d in 1:N
            fill_guards!(src, grid; workgroupsize)
            zero_x_edges!(edges, src; workgroupsize)
            launch(dst, src, CartesianIndex(ntuple(k -> k == d ? 1 : 0, Val(N))), guards(grid); ndrange = grid.n)
            src, dst = dst, src
        end
        src === a || ka_copy!(a, src)
        fill_guards!(a, grid; workgroupsize)
        zero_x_edges!(edges, a; workgroupsize)
        return a
    end
end
