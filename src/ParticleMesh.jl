"""
    ParticleMesh

Differentiable particle–mesh simulation engine with pluggable physics modules.
The engine (particles, grids, particle↔mesh transfers, time loop, differentiation,
backends) lives in `lib/ParticleMeshCore`; physics modules plug in through the interaction
interfaces described in `docs/plan/architecture-draft.md` §0.

This umbrella package currently re-exports `ParticleMeshCore`. The only physics module (A9),
`ParticleMeshElectromagnetic` in `lib/`, is a workspace member of its own and is loaded
directly; it runs the electrostatic cases with no transverse fields. See
`docs/plan/overview.md`.
"""
module ParticleMesh

using ParticleMeshCore
export ParticleMeshCore
for name in names(ParticleMeshCore)
    name === :ParticleMeshCore || @eval export $name
end

end
