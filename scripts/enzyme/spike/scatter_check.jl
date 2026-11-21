# WP-078: the ionization's births in miniature, under Enzyme's reverse mode on CUDA: thread i of
# the "atoms" writes into slot count + i of the "electrons" when its weight is positive, both
# structs of arrays with scalar fields (MixedDuplicated). A dot-product test against duals.
#
#     PM_ENZYME_PATCH=scripts/enzyme/active_reduce.jl julia +1.13 --project=lib/ParticleMeshElectromagnetic/test scripts/enzyme/spike/scatter_check.jl
using CUDACore, KernelAbstractions, Enzyme, ForwardDiff
const Adapt = KernelAbstractions.Adapt
foreach(f -> isempty(f) || include(abspath(f)), split(get(ENV, "PM_ENZYME_PATCH", ""), ';'))
using Random: Xoshiro

struct Sp{A,T}
    x::A
    w::A
    q::T
end
Adapt.adapt_structure(to, s::Sp) = Sp(Adapt.adapt(to, s.x), Adapt.adapt(to, s.w), s.q)

@kernel function birth!(a, e, @Const(count), cap::Int32)
    i = Int32(@index(Global, Linear))
    @inbounds if a.w[i] > zero(eltype(a.w))
        w = a.w[i] * a.x[i]
        if w > zero(eltype(a.w))
            slot = count[1] + i
            if slot <= cap
                e.x[slot] = a.x[i] * a.q
                e.w[slot] = w
                a.x[i] = zero(eltype(a.w))
            end
        end
    end
end
f!(a, e, count) = (birth!(get_backend(a.x))(a, e, count, Int32(length(e.x)); ndrange = length(a.x)); nothing)

n, m = 1000, 3000
rng = Xoshiro(3)
ax, aw = rand(rng, Float32, n) .- 0.3f0, rand(rng, Float32, n) .- 0.2f0
ex, ew = rand(rng, Float32, m), rand(rng, Float32, m)
cnt = Int32[700]
va = (randn(rng, Float32, n), randn(rng, Float32, n)); ve = (randn(rng, Float32, m), randn(rng, Float32, m))
ra = (randn(rng, Float32, n), randn(rng, Float32, n)); re = (randn(rng, Float32, m), randn(rng, Float32, m))
q = 1.5f0
# reverse: shadows seeded with r, after the pass they hold Jᵀr
a = Sp(CuArray(ax), CuArray(aw), q); e = Sp(CuArray(ex), CuArray(ew), q)
da = Sp(CuArray(ra[1]), CuArray(ra[2]), 0f0); de = Sp(CuArray(re[1]), CuArray(re[2]), 0f0)
Enzyme.autodiff(set_runtime_activity(Reverse), f!, Const, Duplicated(a, da), Duplicated(e, de), Const(CuArray(cnt)))
rev = sum(Array(da.x) .* va[1]) + sum(Array(da.w) .* va[2]) + sum(Array(de.x) .* ve[1]) + sum(Array(de.w) .* ve[2])
# forward: Dual arrays with tangent v, ⟨r, Jv⟩
D(x, v) = CuArray(ForwardDiff.Dual{Nothing}.(x, v))
ad = Sp(D(ax, va[1]), D(aw, va[2]), ForwardDiff.Dual{Nothing}(q, 0f0)); ed = Sp(D(ex, ve[1]), D(ew, ve[2]), ForwardDiff.Dual{Nothing}(q, 0f0))
f!(ad, ed, CuArray(cnt))
P(x) = ForwardDiff.partials.(Array(x), 1)
fwd = sum(ra[1] .* P(ad.x)) + sum(ra[2] .* P(ad.w)) + sum(re[1] .* P(ed.x)) + sum(re[2] .* P(ed.w))
println("scatter check: ⟨Jᵀr, v⟩ = $rev, ⟨r, Jv⟩ = $fwd, relative ", abs(rev - fwd) / abs(fwd))
