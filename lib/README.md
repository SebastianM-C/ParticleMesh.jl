# Subpackages

Present:

- `ParticleMeshCore`: the engine (particles, grids, transfers including the charge-conserving `deposit_flux!`, sort, loading, mesh operators, KernelAbstractions kernels; no physics).
- `ParticleMeshElectromagnetic`: self-consistent electromagnetic PIC in 1D, 2D and 3D, the only physics module (A9), with its problem description kept apart from the numerics (A10); the laser antennas, the moving window, the open boundaries, the witness bunch and the field ionization are part of it, and the electrostatic cases run as its edge cases.
- `RunManifests`: the provenance of a run (F3), the `manifest.toml` next to its outputs with the commit, the hashes of its inputs and whatever the run adds.

Planned, in the order they are expected to appear: `Wakefields` (the laser-wakefield user layer on top of `ParticleMeshElectromagnetic`: the cases, the SI conversion of A1, the problem builders, the fixed-energy coupling, the objectives and the LWFA read-outs; WP-092), `ParticleMeshMTK` (symbolic front-end), `ParticleMeshReactant` (Reactant path and the stress-test suite).
