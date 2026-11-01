# Shared by runtests.jl and the subprocess workers: the backend, the A8 element-type matrix, the
# tolerances and the comparison helpers.

using ParticleMeshCore
using KernelAbstractions
using ForwardDiff
using ForwardDiff: Dual, value, partials
using TOML

const backend = CPU()
const TOL = TOML.parsefile(joinpath(@__DIR__, "tolerances.toml"))
"Proposed constant `C` of a test, from `tolerances.toml`."
tolC(id, key = "C") = TOL[id][key]

# The required A8 matrix: Float32, Float64 and `Dual{Float32,3}`.
const D3 = Dual{Nothing,Float32,3}
const TYPES = (Float32, Float64, D3)

basetype(::Type{T}) where {T<:AbstractFloat} = T
basetype(::Type{<:Dual{<:Any,V}}) where {V} = V
base_eps(::Type{T}) where {T} = eps(basetype(T))
components(x::Real) = (x,)
components(d::Dual) = (value(d), Tuple(partials(d))...)
"Largest absolute difference over the value and every partial."
absdiff(a, b) = maximum(abs, components(a - b))
absmax(a) = maximum(abs, components(a))
value_of(x::Real) = x
value_of(d::Dual) = value(d)
"A value of `T` with fixed, distinct partials for duals (so derivatives are exercised)."
seeded(::Type{T}, x) where {T<:AbstractFloat} = T(x)
seeded(::Type{Dual{Nothing,V,N}}, x) where {V,N} = Dual{Nothing}(V(x), ntuple(i -> V(i) / 8, N)...)
"A `Dual{Float32,3}` with the given value and partials."
mkdual(v, ps...) = Dual{Nothing}(Float32(v), Float32.(ps)...)
"Bitwise equality of two arrays of the same isbits element type (`==` on `Dual` compares values only)."
same_bits(a::AbstractArray{T}, b::AbstractArray{T}) where {T} =
    size(a) == size(b) && reinterpret(UInt8, vec(collect(a))) == reinterpret(UInt8, vec(collect(b)))
same_bits(a, b) = false
type_label(T) = T === D3 ? "Dual{Float32,3}" : string(T)

# Measured ratios err / (eps(T)·scale), per test id and type, printed at the end for tolerances.toml.
const MEASURED = Dict{String,Dict{String,Float64}}()
function record!(id, T, ratio, be = CPU())
    d = get!(MEASURED, id, Dict{String,Float64}())
    key = be isa CPU ? type_label(T) : "$(type_label(T)) on $(nameof(typeof(be)))"
    d[key] = max(get(d, key, 0.0), Float64(ratio))
    return ratio
end

include("gpu_support.jl")   # the GPU switch and the host/device round trip (WP-013)
