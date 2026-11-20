# Spike-only: γ through the LLVM sqrt intrinsic; on PoCL `sqrt` lowers to an OpenCL builtin Enzyme has no derivative for.
@eval ParticleMeshElectromagnetic begin
    @inline lorentz_factor(u1::T, u2::T, u3::T) where {T<:Union{Float32,Float64}} = Base.Math.sqrt_llvm(one(T) + u1 * u1 + u2 * u2 + u3 * u3)
end
