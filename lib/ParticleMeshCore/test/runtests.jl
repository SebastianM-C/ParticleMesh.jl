using ParticleMeshCore
using ParticleMeshCore.Electrostatic
using KernelAbstractions
using ForwardDiff
using ForwardDiff: Dual, value, partials
using Test
using TOML

const backend = CPU()
const TOL = TOML.parsefile(joinpath(@__DIR__, "tolerances.toml"))
"Proposed constant `C` of a test, from `tolerances.toml`."
tolC(id) = TOL[id]["C"]

# The A8 matrix of the spike: Float32, Float64 and `Dual{Float32,3}`.
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

# Measured ratios err / (eps(T)·scale), per test id and type, printed at the end for tolerances.toml.
const MEASURED = Dict{String,Dict{String,Float64}}()
function record!(id, T, ratio)
    d = get!(MEASURED, id, Dict{String,Float64}())
    key = string(T === D3 ? "Dual{Float32,3}" : T)
    d[key] = max(get(d, key, 0.0), Float64(ratio))
    return ratio
end

include("theory.jl")
include("helpers.jl")

@testset "ParticleMeshCore" begin
    try
        include("transfers.jl")
        include("particles.jl")
        include("electrostatic.jl")
        include("physics.jl")
        include("gradient.jl")
        include("determinism.jl")
    finally
        println("\nMeasured err/(eps(T)·scale) per test (compare with tolerances.toml):")
        for id in sort(collect(keys(MEASURED)))
            println("  ", id, ": ", join(("$k = $(round(v, sigdigits = 3))" for (k, v) in sort(collect(MEASURED[id]))), ", "))
        end
    end
end
