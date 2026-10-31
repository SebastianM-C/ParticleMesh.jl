# WP-009: the cost of the charge-conserving flux deposition

| | |
|---|---|
| Work package | WP-009 (goals A2, A3, A4, A7, A8) |
| Commit of the runs | `0e3d0bc60f78d8f303dbcb6f3ce8b59e672fbb84` (clean tree) |
| Runs | `flux-timing-t1-20261031T220715`, `flux-timing-t16-20261031T220734` |
| Input hashes | SHA-256 of every input file in each run's manifest: `wp-009-flux/<run>.manifest.json` |
| Environment | Julia 1.13.3, KernelAbstractions 0.10.4 CPU backend; Intel Core i9-14900K, 1 and 16 threads |
| Figures | `wp-009-flux/flux_timing_t{1,16}.png`, each with its JSON sidecar |
| Status | measured, not independently reviewed; no thresholds (WP-009's budget is a report) |

## Setup

`scripts/es1d/flux_timing.jl`: 64 cells, quadratic shape at `Center()`, a quiet-start species with every particle then moved by up to 0.9 cells, sorted once. The script times `deposit!` (the default q·w) and `deposit_flux!` (Jx, Jy, Jz) separately, without the sort, as the median of 21 calls after a warm-up call.

## Results

ns per particle (sidecars `flux_timing_t1.json`, `flux_timing_t16.json`):

| particles | threads | T | Fast deposit! | Fast deposit_flux! | Reference deposit! | Reference deposit_flux! |
|---|---|---|---|---|---|---|
| 16 384 | 1 | Float32 | 13.0 | 66.8 | 3.0 | 47.7 |
| 262 144 | 1 | Float32 | 12.5 | 59.1 | 2.9 | 45.6 |
| 262 144 | 1 | Float64 | 11.9 | 59.1 | 2.7 | 41.9 |
| 262 144 | 16 | Float32 | 22.1 | 98.5 | 1.4 | 23.6 |
| 262 144 | 16 | Float64 | 16.4 | 54.2 | 1.4 | 21.7 |

Two observations, both from one run per configuration:

- On one thread the flux costs about 16 times `deposit!` in Reference mode and about 5 times in Fast mode. It writes three fields, but the larger factor is in Reference mode. There each node work item computes a particle's whole flux stencil (old and new weights, the running sum, three outputs) once for every node it touches, and keeps one entry. Most of that work is repeated, and the kernel could compute only the entry it needs. That is left as a follow-up; the contract and the bits do not depend on it.
- With 16 threads the Reference flux scales by about 2× and the atomic flux gets slower, as the spike saw for `deposit!`: threads contend on 64 nodes.

Scratch per particle: 33 B for Float32 and 37 B for Float64, the same as without extras, because the two old-position extras (`Int32` and `T`) reuse the core permutation buffers. The extras themselves add 8 B (Float32) or 12 B (Float64) to every particle.

## Open

- A Reference kernel that computes only the needed stencil entry, or a per-particle pass followed by a node-wise reduction of stored contributions.
- No GPU numbers (WP-013, or its follow-up issue for `deposit_flux!`).
