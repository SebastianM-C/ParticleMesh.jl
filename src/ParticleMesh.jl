"""
    ParticleMesh

Differentiable particle–mesh simulation engine with pluggable physics modules.
The engine (particles, grids, particle↔mesh transfers, time loop, differentiation,
backends) lives in `lib/ParticleMeshCore` once it exists; physics modules plug in
through the interaction interfaces described in `docs/plan/architecture-draft.md` §0.

This umbrella package currently only re-exports what exists. See `docs/plan/overview.md`.
"""
module ParticleMesh

end
