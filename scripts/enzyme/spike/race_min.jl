using KernelAbstractions, Enzyme
const BE = CPU()
@kernel function inplace!(x)
    i = @index(Global)
    @inbounds x[i] = x[i]^2
end
@kernel function gath!(y, f, @Const(x), @Const(idx))
    i = @index(Global)
    @inbounds y[i] = f[idx[i]] * x[i]^2
end
@kernel function gath2!(y, f, x, @Const(idx))
    i = @index(Global)
    @inbounds begin
        a = x[i]
        x[i] = 0.5 * a
        y[i] = f[idx[i]] * a * a + f[idx[i] + 1] * a
    end
end
run1!(x) = (inplace!(BE)(x; ndrange = length(x)); nothing)
run2!(y, f, x, idx) = (gath!(BE)(y, f, x, idx; ndrange = length(x)); nothing)
run3!(y, f, x, idx) = (gath2!(BE)(y, f, x, idx; ndrange = length(x)); nothing)
n = 64
x0 = collect(range(1.0, 2.0, length = n)); idx = [1 + (i % 3) for i in 1:n]; f0 = [1.0, 2.0, 3.0, 4.0]
let x = copy(x0), dx = ones(n)
    Enzyme.autodiff(Reverse, run1!, Const, Duplicated(x, dx))
    println("in place: max err ", maximum(abs, dx .- 2 .* x0))
end
let y = zeros(n), dy = ones(n), f = copy(f0), df = zeros(4), x = copy(x0), dx = zeros(n)
    Enzyme.autodiff(Reverse, run2!, Const, Duplicated(y, dy), Duplicated(f, df), Duplicated(x, dx), Const(idx))
    dfe = zeros(4); for i in 1:n; dfe[idx[i]] += x0[i]^2; end
    println("shared read: df err ", maximum(abs, df .- dfe), "  dx err ", maximum(abs, dx .- 2 .* f0[idx] .* x0))
end
let y = zeros(n), dy = ones(n), f = copy(f0), df = zeros(4), x = copy(x0), dx = zeros(n)
    Enzyme.autodiff(Reverse, run3!, Const, Duplicated(y, dy), Duplicated(f, df), Duplicated(x, dx), Const(idx))
    dfe = zeros(4); for i in 1:n; dfe[idx[i]] += x0[i]^2; dfe[idx[i]+1] += x0[i]; end
    dxe = 2 .* f0[idx] .* x0 .+ f0[idx .+ 1]   # x's final value has zero adjoint seed (dx = 0 in)
    println("shared read + overwrite: df err ", maximum(abs, df .- dfe), "  dx err ", maximum(abs, dx .- dxe))
end
