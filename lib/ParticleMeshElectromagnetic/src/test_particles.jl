# Single particles in prescribed fields (WP-015): the equation of motion as an ODE right-hand
# side, and the names the OrdinaryDiffEq extension gives methods to. The PIC step does not go
# through this interface (`docs/plan/sciml-reuse.md` §1); it calls `push_momentum` directly.

"""
    lorentz_rhs!(ds, s, p, t)

The equation of motion of one particle in A1 units as an in-place ODE right-hand side, for the
state `s = [x; u]` (six entries, u = γv per unit mass): `dx/dt = u/γ`,
`du/dt = (q/m)(E(x, t) + u/γ × B(x, t))`, with `p = (; qm, E, B)`, `E` and `B` callables
`(x::NTuple{3}, t) -> NTuple{3}`. A high-order solver such as `Vern9` integrates this; the
pusher algorithms ignore it and step with `push_particle` instead.
"""
function lorentz_rhs!(ds, s, p, t)
    x = (s[1], s[2], s[3])
    u = (s[4], s[5], s[6])
    ig = inv(lorentz_factor(u))
    v = (ig * u[1], ig * u[2], ig * u[3])
    E = p.E(x, t)
    c = cross3(v, p.B(x, t))
    ds[1], ds[2], ds[3] = v
    ds[4] = p.qm * (E[1] + c[1])
    ds[5] = p.qm * (E[2] + c[2])
    ds[6] = p.qm * (E[3] + c[3])
    return nothing
end

"""
    particle_problem(x0, u0, tspan; qm, E, B) -> ODEProblem

One particle in prescribed fields as a `SciMLBase.ODEProblem` with [`lorentz_rhs!`](@ref): the
state `[x0; u0]` (a `Vector` of the element type of `x0` and `u0`), `p = (; qm, E, B)`. Any
OrdinaryDiffEq solver integrates it, and so does a pusher through
[`pusher_algorithm`](@ref). Needs SciMLBase and OrdinaryDiffEqCore loaded (a package extension).
"""
function particle_problem end

"""
    pusher_algorithm(pusher::Pusher) -> OrdinaryDiffEqAlgorithm

The pusher as a fixed-step OrdinaryDiffEq algorithm for a [`particle_problem`](@ref): each step
is [`push_particle`](@ref) with the problem's `p.qm`, `p.E`, `p.B`, so a solve gives the same
bits as stepping `push_particle` by hand from `t₀` with the same `dt` (T-237). The saved states
are synchronized, x and u at the same time, so no half-step start is needed; `dt` is required,
and `saveat` must name step times (the algorithm has no interpolant: between steps the
solution is the left-endpoint value). Needs SciMLBase and OrdinaryDiffEqCore loaded.
"""
function pusher_algorithm end
