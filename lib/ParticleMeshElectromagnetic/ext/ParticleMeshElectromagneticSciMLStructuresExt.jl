"""
    ParticleMeshElectromagneticSciMLStructuresExt

The SciMLStructures interface of the electromagnetic module's parameters (WP-060): an
`EMProblem`, the `p` of the harness's `DiscreteProblem`, and an `EMParams` are SciML structures
whose `Tunable` portion is [`tunables`](@ref), so the SciML tools that ask a problem for its
parameters as a vector (`SciMLStructures.canonicalize`, `replace`) get the differentiable ones
of A3 and can write them back.

- `canonicalize(Tunable(), p)` is `(tunables(p), v -> with_tunables(p, v), false)`: a new vector,
  no alias of `p`, which is immutable.
- `replace(Tunable(), p, v)` is `with_tunables(p, v)`, in the element type of `v`. There is no
  `replace!`.
- No other portion: the configuration is not a set of numbers a tool may vary.

A parameter of the initial load reaches the run only through a state loaded from it, so the
gradient through a solve is that of
`x -> J(solve(remake_reload(prob; p = replace(Tunable(), prob.p, x)), FunctionMap()))`
(`remake_reload` of the SciML harness), not of a plain `remake(prob; p)`, which keeps `u0`.
"""
module ParticleMeshElectromagneticSciMLStructuresExt

using ParticleMeshElectromagnetic: EMProblem, EMParams, tunables, with_tunables
import SciMLStructures
using SciMLStructures: Tunable, AbstractPortion

const Parameters = Union{EMProblem,EMParams}

SciMLStructures.isscimlstructure(::Parameters) = true
SciMLStructures.ismutablescimlstructure(::Parameters) = false
SciMLStructures.hasportion(::Tunable, ::Parameters) = true
SciMLStructures.hasportion(::AbstractPortion, ::Parameters) = false
SciMLStructures.canonicalize(::Tunable, p::Parameters) = (tunables(p), v -> with_tunables(p, v), false)
SciMLStructures.replace(::Tunable, p::Parameters, v::AbstractVector) = with_tunables(p, v)

end
