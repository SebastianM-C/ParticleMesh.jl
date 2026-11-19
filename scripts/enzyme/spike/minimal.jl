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
function run(k, y, x, extra...)
    fill!(y, 0)
    k(BE)(y, x, extra...; ndrange = length(x))
    KernelAbstractions.synchronize(BE)
    return sum(abs2, y)
end
scale_run(y, x, a) = run(scaled!, y, x, 2 * a)

x = [1.0, 2.0, 3.0, 4.0]; idx = [1, 2, 1, 3]
cases = [
    ("square", sq!, () -> (dev(zeros(4)),), () -> (), 4 .* x .^ 3),
    ("sqrt", root!, () -> (dev(zeros(4)),), () -> (), 2 .* x),
    ("atomic scatter", scatter!, () -> (dev(zeros(3)),), () -> (Const(dev(idx)),), [8 * (1 + 3), 8 * 2, 8 * (1 + 3), 8 * 4]),
]
for (name, k, ys, extras, expected) in cases
    y = ys()[1]; dy = zero(y); xd = dev(x); dx = zero(xd)
    try
        Enzyme.autodiff(Reverse, run, Active, Const(k), Duplicated(y, dy), Duplicated(xd, dx), extras()...)
        println("[$name] OK: dx = ", Array(dx), ", expected ", expected, Array(dx) ≈ expected ? " (agree)" : " (DISAGREE)")
    catch e
        println("[$name] FAILED: ", replace(first(sprint(showerror, e), 3000), "\n" => " | "))
    end
end
let y = dev(zeros(4)), dy = zero(y), xd = dev(x), dx = zero(xd)
    try
        r = Enzyme.autodiff(Reverse, scale_run, Active, Duplicated(y, dy), Duplicated(xd, dx), Active(1.5))
        println("[active scalar argument] OK: da = ", r[1][3], ", expected ", 8 * 1.5 * sum(abs2, x))
    catch e
        println("[active scalar argument] FAILED: ", replace(first(sprint(showerror, e), 600), "\n" => " | "))
    end
end
