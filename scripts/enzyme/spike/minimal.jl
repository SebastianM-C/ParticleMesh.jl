# WP-050 spike: Enzyme reverse mode through minimal KernelAbstractions kernels (a square, sqrt,
# an atomic scatter, a scalar argument), on the CPU (PoCL) or with PM_BACKEND=cuda on the GPU.
using KernelAbstractions, Enzyme
import ParticleMeshCore
const Atomix = ParticleMeshCore.Atomix
const ON_GPU = get(ENV, "PM_BACKEND", "cpu") == "cuda"
if ON_GPU
    using CUDACore
    CUDACore.allowscalar(false)
end
const BE = ON_GPU ? CUDACore.CUDABackend() : CPU()
dev(a) = (b = KernelAbstractions.allocate(BE, eltype(a), size(a)); copyto!(b, a); b)

@kernel function sq!(y, @Const(x))
    i = @index(Global)
    @inbounds y[i] = x[i]^2
end
@kernel function root!(y, @Const(x))
    i = @index(Global)
    @inbounds y[i] = sqrt(1 + x[i]^2)
end
@kernel function scatter!(y, @Const(x), @Const(idx))
    i = @index(Global)
    @inbounds Atomix.@atomic :monotonic y[idx[i]] += 2 * x[i]
end
@kernel function scaled!(y, @Const(x), a)
    i = @index(Global)
    @inbounds y[i] = a * x[i]
end
# The differentiated function launches and returns nothing: no reduction and no synchronize
# inside it (Enzyme cannot differentiate a CUDA stream synchronize). The adjoint of y is seeded
# with ones, so the gradients are those of sum(y).
function run!(k, y, x, extra...)
    k(BE)(y, x, extra...; ndrange = length(x))
    return nothing
end

x = [1.0, 2.0, 3.0, 4.0]; idx = [1, 2, 1, 3]
cases = [
    ("square", sq!, 4, () -> (), 2 .* x),
    ("sqrt", root!, 4, () -> (), x ./ sqrt.(1 .+ x .^ 2)),
    ("atomic scatter", scatter!, 3, () -> (Const(dev(idx)),), fill(2.0, 4)),
]
for (name, k, ny, extras, expected) in cases
    y = dev(zeros(ny)); dy = dev(ones(ny)); xd = dev(x); dx = dev(zeros(4))
    try
        Enzyme.autodiff(Reverse, run!, Const, Const(k), Duplicated(y, dy), Duplicated(xd, dx), extras()...)
        KernelAbstractions.synchronize(BE)
        println("[$name] OK: dx = ", Array(dx), ", expected ", expected, Array(dx) ≈ expected ? " (agree)" : " (DISAGREE)")
    catch e
        println("[$name] FAILED: ", replace(first(sprint(showerror, e), 3000), "\n" => " | "))
    end
end
let y = dev(zeros(4)), dy = dev(ones(4)), xd = dev(x), dx = dev(zeros(4))
    try
        r = Enzyme.autodiff(Reverse, run!, Const, Const(scaled!), Duplicated(y, dy), Duplicated(xd, dx), Active(1.5))
        KernelAbstractions.synchronize(BE)
        println("[active scalar argument] OK: da = ", r[1][4], ", expected ", sum(x), "; dx = ", Array(dx), ", expected 1.5 each")
    catch e
        println("[active scalar argument] FAILED: ", replace(first(sprint(showerror, e), 600), "\n" => " | "))
    end
end
