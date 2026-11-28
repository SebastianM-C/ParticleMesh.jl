# The known-count check of the counter study (WP-091): GPUDiagnostics' own `counter_probe`
# (its test/gpu/counter_probe.jl), three launches of 4096 work-items × 32 Float64 FMAs, so an
# ncu collection of the second and third launches must count 131072 DFMA thread instructions
# each. `study.jl probe` runs it under ncu and checks the count.
#
#   julia +1.13 --project=scripts scripts/counters/probe_kernel.jl
using CUDA, KernelAbstractions, Adapt

@kernel function counter_probe!(out, @Const(x), niter)
    i = @index(Global, Linear)
    @inbounds begin
        a = x[i]
        for _ in 1:niter
            a = fma(a, 0.999, 0.001)
        end
        out[i] = a
    end
end

const BE = CUDABackend()
n, niter = 4096, 32
x = Adapt.adapt(BE, fill(0.5, n))
out = similar(x)
for _ in 1:3
    counter_probe!(BE, 256)(out, x, Int32(niter); ndrange = n)
    KernelAbstractions.synchronize(BE)
end
reference = foldl((a, _) -> fma(a, 0.999, 0.001), 1:niter; init = 0.5)
all(v -> isapprox(v, reference; rtol = 1e-12), Array(out)) || error("probe_kernel.jl: wrong result")
println("probe_kernel.jl: ok")
