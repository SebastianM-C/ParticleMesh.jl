# WP-078, WP-084: what trapped.jl and trapped3d.jl share: J's summand per particle slot as a kernel,
# every array and every floating-point scalar of a state by its path, and the contractions of a
# reverse pass's adjoints with a Dual twin's tangents over them (⟨adjoint, tangent j⟩ in Float64,
# with the non-finite entries counted).
using ForwardDiff: ForwardDiff
using KernelAbstractions

# J's summand per slot, into `out` (the slot's weight × kinetic energy × position and trapping weights,
# `slot_trapped` of trapped2d_case.jl, which the including script loads first)
@kernel function summand_kernel!(out, @Const(status), @Const(c1), @Const(o1), @Const(u1), @Const(u2), @Const(u3), @Const(w), h1, lo, hi, ulo, uhi)
    i = @index(Global, Linear)
    @inbounds out[i] = slot_trapped(status[i], c1[i], o1[i], u1[i], u2[i], u3[i], w[i], h1, lo, hi, ulo, uhi)
end
function summands!(out, s, h1, fade, rise)
    T = eltype(s.w)
    summand_kernel!(get_backend(out))(out, s.status, s.cell[1], s.offset[1], s.u[1], s.u[2], s.u[3], s.w, h1,
                                      T(fade[1]), T(fade[2]), T(rise[1]), T(rise[2]); ndrange = length(out))
    return nothing
end

# every array of a state (any element type), by its path, each once
all_arrays!(out, path, x::AbstractArray) = (any(y -> y === x, values(out)) || (out[path] = x); out)
all_arrays!(out, path, x::Union{Tuple,NamedTuple}) = (foreach(k -> all_arrays!(out, "$path.$k", x[k]), keys(x)); out)
function all_arrays!(out, path, x)
    (isstructtype(typeof(x)) && !(x isa Module) && !(x isa Type)) || return out
    for f in fieldnames(typeof(x))
        isdefined(x, f) && all_arrays!(out, "$path.$f", getfield(x, f))
    end
    return out
end
all_arrays(x) = all_arrays!(Dict{String,Any}(), "st", x)

# every floating-point scalar of a struct (or a Dual), by its path; arrays are `leaves`'
scalars!(out, path, x::Union{AbstractFloat,ForwardDiff.Dual}) = (out[path] = x; out)
scalars!(out, path, x::AbstractArray) = out
scalars!(out, path, x::Union{Tuple,NamedTuple}) = (foreach(k -> scalars!(out, "$path.$k", x[k]), keys(x)); out)
function scalars!(out, path, x)
    (isstructtype(typeof(x)) && !(x isa Module) && !(x isa Type)) || return out
    for f in fieldnames(typeof(x))
        isdefined(x, f) && scalars!(out, "$path.$f", getfield(x, f))
    end
    return out
end
scalars(x) = scalars!(Dict{String,Any}(), "x", x)
partial(x::ForwardDiff.Dual, j) = ForwardDiff.partials(x, j)
partial(x::Real, j) = zero(x)

# An entry with a zero tangent adds nothing, whatever its adjoint: free slots and scratch can hold
# a non-finite adjoint (from garbage values), and NaN·0 would poison the sum. A non-finite adjoint
# against a nonzero tangent is a real failure: it is counted and reported (NONFINITE).
const NONFINITE = Dict{String,Int}()
function tdot(a, t, path)
    (t == 0 || a == 0) && return 0.0
    isfinite(a) || (NONFINITE[path] = get(NONFINITE, path, 0) + 1)
    return a * t
end
# Fused reductions over the adjoint and the Dual array, without temporaries: at λ₀/16 a broadcast
# temporary of a flux patch's size (3.4 GiB) did not fit beside the shadow and the Dual start (WP-086).
function adot(A, D, j, path)
    bad = mapreduce(+, A, D; init = 0) do a, d
        p = ForwardDiff.partials(d, j)
        return Int((!isfinite(a) & (p != 0)) | (!isfinite(p) & (a != 0)))
    end
    bad > 0 && (NONFINITE[path] = get(NONFINITE, path, 0) + bad)
    # in Float64: a Float32 sum of large products can overflow to ±Inf and give Inf − Inf
    # an entry with a zero adjoint adds nothing either: uninitialized scratch of the Dual start can
    # hold NaN partials (the electrons' sort workspace), against a zero adjoint
    v = mapreduce(+, A, D; init = 0.0) do a, d
        p = ForwardDiff.partials(d, j)
        return (p == 0) | (a == 0) ? 0.0 : Float64(a) * Float64(p)
    end
    isfinite(v) || (println("  non-finite contribution of $path: $v (max |adjoint| ", maximum(abs, A), ", max |tangent| ",
                            mapreduce(d -> abs(ForwardDiff.partials(d, j)), max, D), ")"); NONFINITE[path * " (sum)"] = 1)
    return v
end
"⟨adjoint, tangent j⟩ over the scalars of two structs of the same shape (an adjoint and its Dual twin), by path."
function scalar_dot(adj, dual, j)
    A, D = scalars(adj), scalars(dual)
    return sum((tdot(Float64(A[k]), Float64(partial(D[k], j)), k) for k in intersect(keys(A), keys(D))); init = 0.0)
end
"""
⟨adjoint, tangent j⟩ over the floating-point arrays of a state's adjoint and a Dual state of the same
shape, by path, leaving out the adjoint's arrays in `skip` (identity): a shadow's workspaces when
they are the primal's scratch (trapped3d.jl's `shadow=shared`), whose values are no adjoint.
"""
function array_dot(adj, dual, j; skip = Base.IdSet{Any}())
    A, D = leaves(adj), leaves(dual)
    return sum((adot(A[k], D[k], j, k) for k in intersect(keys(A), keys(D)) if eltype(D[k]) <: ForwardDiff.Dual && !(A[k] in skip)); init = 0.0)
end
