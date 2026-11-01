"""
    ParticleMeshElectromagneticOrdinaryDiffEqCoreExt

The pushers as OrdinaryDiffEq algorithms for single particles in prescribed fields (WP-015).
A [`PusherAlgorithm`](@ref) wraps a pusher type; its step is the module's `push_particle`, the
same `push_momentum` device function the PIC kernel calls, so each pusher has one
implementation. The problem is a plain `ODEProblem` with the true right-hand side
`lorentz_rhs!`, so the same problem goes to `Vern9` for a reference orbit.
"""
module ParticleMeshElectromagneticOrdinaryDiffEqCoreExt

using ParticleMeshElectromagnetic: ParticleMeshElectromagnetic, Pusher, push_particle, lorentz_rhs!
import ParticleMeshElectromagnetic: particle_problem, pusher_algorithm
using SciMLBase: SciMLBase, ODEProblem
import OrdinaryDiffEqCore
using OrdinaryDiffEqCore: OrdinaryDiffEqAlgorithm, OrdinaryDiffEqMutableCache, @cache
import OrdinaryDiffEqCore: alg_cache, perform_step!, initialize!, get_fsalfirstlast, alg_order,
    isfsal, dt_required

"""
    PusherAlgorithm(pusher)

A fixed-step OrdinaryDiffEq algorithm that steps a `particle_problem` with `push_particle`.
The pusher types stay free of OrdinaryDiffEqCore (it is not a hard dependency of the module),
so the algorithm wraps them: `pusher_algorithm(Boris())`.
"""
struct PusherAlgorithm{P<:Pusher} <: OrdinaryDiffEqAlgorithm
    pusher::P
end

pusher_algorithm(p::Pusher) = PusherAlgorithm(p)

alg_order(::PusherAlgorithm) = 2
isfsal(::PusherAlgorithm) = false
dt_required(::PusherAlgorithm) = true
SciMLBase.isautodifferentiable(::PusherAlgorithm) = true
SciMLBase.allows_arbitrary_number_types(::PusherAlgorithm) = true

function particle_problem(x0, u0, tspan; qm, E, B)
    T = mapreduce(typeof, promote_type, (x0..., u0...))
    s0 = T[x0[1], x0[2], x0[3], u0[1], u0[2], u0[3]]
    return ODEProblem(lorentz_rhs!, s0, tspan, (; qm = T(qm), E, B))
end

@cache struct PusherCache{uType} <: OrdinaryDiffEqMutableCache
    u::uType
    uprev::uType
end

get_fsalfirstlast(::PusherCache, u) = (nothing, nothing)

function alg_cache(::PusherAlgorithm, u, rate_prototype, ::Type{uEltypeNoUnits},
                   ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits}, uprev, uprev2, f, t,
                   dt, reltol, p, calck, ::Val{true}, verbose) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    return PusherCache(u, uprev)
end

function alg_cache(::PusherAlgorithm, u, rate_prototype, ::Type{uEltypeNoUnits},
                   ::Type{uBottomEltypeNoUnits}, ::Type{tTypeNoUnits}, uprev, uprev2, f, t,
                   dt, reltol, p, calck, ::Val{false}, verbose) where {uEltypeNoUnits, uBottomEltypeNoUnits, tTypeNoUnits}
    throw(ArgumentError("PusherAlgorithm steps in place: build the problem with particle_problem"))
end

function initialize!(integrator, ::PusherCache)
    integrator.kshortsize = 0
    resize!(integrator.k, 0)
    return nothing
end

function perform_step!(integrator, cache::PusherCache, repeat_step = false)
    (; u, uprev, dt, t, p) = integrator
    T = eltype(u)
    x = (uprev[1], uprev[2], uprev[3])
    v = (uprev[4], uprev[5], uprev[6])
    x′, v′ = push_particle(integrator.alg.pusher, x, v, convert(T, t), convert(T, dt), p.qm, p.E, p.B)
    u[1], u[2], u[3] = x′
    u[4], u[5], u[6] = v′                    # the step never calls the problem's f: stats.nf stays 0
    return nothing
end

end
