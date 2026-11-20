# WP-078: checks `active_reduce.jl` on a toy kernel with a float and a struct as Active
# arguments, against their adjoints by hand. GPU only (CUDA):
#
#     julia +1.13 --project=lib/ParticleMeshElectromagnetic/test scripts/enzyme/spike/active_reduce_check.jl
using CUDACore, KernelAbstractions, Enzyme
include(joinpath(@__DIR__, "..", "active_reduce.jl"))

struct Coeffs{T}
    b::T
    c::T
    n::Int32
end
@kernel function toy!(y, @Const(x), a, s)
    i = @index(Global, Linear)
    @inbounds y[i] = a * x[i] + s.b * x[i]^2 + s.c * s.n
end
function f!(y, x, a, s)
    toy!(get_backend(y))(y, x, a, s; ndrange = length(y))
    return nothing
end
n = 1000
x = CuArray(Float32.(range(-1, 1; length = n)))
y = similar(x); dy = similar(x); fill!(dy, 1)
a = 0.5f0; s = Coeffs(0.25f0, 2.0f0, Int32(3))
res = Enzyme.autodiff(Reverse, f!, Const, Duplicated(y, dy), Const(x), Active(a), Active(s))
da, ds = res[1][3], res[1][4]
xh = Array(x)
@show da sum(xh)
@show ds.b sum(xh .^ 2)
@show ds.c 3n
ok = isapprox(da, sum(xh); rtol = 1e-5, atol = 1e-4) && isapprox(ds.b, sum(xh .^ 2); rtol = 1e-5) && isapprox(ds.c, 3n; rtol = 1e-6)
println(ok ? "active_reduce check: pass" : "active_reduce check: FAIL")
println("seen: ", ACTIVE_SEEN)
