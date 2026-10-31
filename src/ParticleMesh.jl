"""
    ParticleMesh

Differentiable particle–mesh simulation engine with pluggable physics modules.
The engine (particles, grids, particle↔mesh transfers, time loop, differentiation,
backends) lives in `lib/ParticleMeshCore`; physics modules plug in through the interaction
interfaces described in `docs/plan/architecture-draft.md` §0.

This umbrella package currently re-exports `ParticleMeshCore`. No physics module exists yet:
the electromagnetic module is the first (A9), and the electrostatic reference of the WP-010
spike is test-support code, `lib/ParticleMeshCore/test/es_reference.jl`. See
`docs/plan/overview.md`.
"""
module ParticleMesh

using ParticleMeshCore
export ParticleMeshCore
for name in names(ParticleMeshCore)
    name === :ParticleMeshCore || @eval export $name
end

end
