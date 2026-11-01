# WP-020: the Esirkepov flux deposition at N = 2

| | |
|---|---|
| Work package | WP-020 (goals A2, A4, A7, A8, A9) |
| Commits | base `0d6ecd7`; phase 1 (tests, 2D stencil, Fast path, prototype script) `1bf5276`; phase 2 (the Reference path) `52bcfa1`; contract text `4c58406`; folded prototype checks `9e6d587` |
| Runs | prototypes (on `9e6d587`): `engine2d-flux-prototypes-t1-20261101T183639`, `engine2d-flux-prototypes-t32-20261101T183848`; engine (on `4c58406`; the engine sources are unchanged since, but `scripts/engine2d/flux_timing.jl` is not: these two runs record its hash `d97cbea…`, the script on `main` and in the prototype runs hashes `bf3182a…`, so the ported runs predate the final script, and what changed in it is not checked here): `engine2d-flux-ported-t1-20261101T182649`, `engine2d-flux-ported-t32-20261101T182740` |
| Input hashes | SHA-256 of every input file in each run's manifest: `wp-020-flux-2d/<run>.manifest.toml`; all four say `dirty = false` |
| Environment | Julia 1.13.3, KernelAbstractions 0.10.4 CPU backend (POCL), AcceleratedKernels 0.5.11; Intel Core i9-14900K, 1 and 32 threads |
| Figures | `wp-020-flux-2d/flux_prototypes_t{1,32}.png`, `flux_ported_t{1,32}.png`, each with its JSON sidecar; `flux_scratch.json` |
| Status | measured, not independently reviewed; no number here is to leave the repository before that review. One run per thread count; every figure is the median of 7 samples with the spread (min, max) in the sidecars. |

## 1. The scheme and the sources read

`deposit_flux!` at N = 2 (`lib/ParticleMeshCore/src/flux2d.jl`, `flux_stencil`) takes per axis `d` the old and new 1D shape weights `S0_d`, `S1_d` on the union of the two stencils, `ΔS_d = S1_d − S0_d`, `A_d = (S0_d + S1_d)/2` and the running sum `P_d` of `ΔS_d` in increasing node order, and deposits

    Jx[k1, k2] = −(q·w·Δx₁/Δt) · P_1[k1] · A_2[k2]
    Jy[k1, k2] = −(q·w·Δx₂/Δt) · A_1[k1] · P_2[k2]
    Jz[k1, k2] = quantity_z · (A_1[k1]·A_2[k2] + ΔS_1[k1]·ΔS_2[k2]/12).

Expanded, `Jx` is `−(q·w·Δx₁/Δt) Σ_{m ≤ k1} ΔS_1[m]·(S0_2[k2] + ΔS_2[k2]/2)` and the weight of `Jz` is `S0_1·S0_2 + (ΔS_1·S0_2 + S0_1·ΔS_2)/2 + ΔS_1·ΔS_2/3`, Esirkepov's out-of-plane weight with its ⅓ term. It is not the average of the old and new products (T-301 checks the difference in one hand-evaluated case).

Read in this package:

- **Smilei v5.1** (the tag of `scripts/crosscode/containers/smilei.Containerfile`), `src/Projector/Projector2D2Order.cpp`, `Projector2D2Order::currents`: `tmpJx[j] -= crx_p·DSx[i−1]·(Sy0[j] + 0.5·DSy[j])` accumulated over `i`, `Jy` likewise with `cry_p·DSy[j−1]·(Sx0[i] + 0.5·DSx[i])` accumulated over `j`, and `Jz += crz_p·(Sy0[j]·(0.5·Sx1[i] + Sx0[i]) + Sy1[j]·(0.5·Sx0[i] + Sx1[i]))` with `crz_p = charge_weight·one_third·v_z`. That is `(1/3)(S0S0 + S1S1) + (1/6)(S1S0 + S0S1)`, the same weight as above. `crx_p = charge_weight·dx/dt`, the same prefactor.
- **PIConGPU 0.8.0** (commit `6bb3470`, as `scripts/crosscode/containers/picongpu.Containerfile`), `include/picongpu/fields/currentDeposition/Esirkepov/Esirkepov2D.hpp`: `cptCurrent1D` accumulates `DS(i)·(−currentSurfaceDensity·(s0j + 0.5·dsj))` over `i`, and `cptCurrentZ` uses `W = s0i·s0j + 0.5·(dsi·s0j + s0i·dsj) + (1/3)·dsi·dsj` times `charge/cellVolume·v_z`.
- **Esirkepov, Comput. Phys. Commun. 135 (2001) 144**: **not read** (not reachable from this session). The statement that the two codes implement the paper's scheme rests on the codes' own naming, so it is unverified here.

Both codes bin by the nearest node at order 2 (Smilei `round(x/dx)`, a 5-point stencil). The engine bins by cell (contract 9), so its stencil at a `Node()` stagger is 6 points wide; the deposited current is the same up to roundoff (section 3).

## 2. The Reference mechanism (prototypes)

`scripts/engine2d/flux_timing.jl prototypes` times prototype Reference kernels on the engine's per-particle `flux_stencil`, so a particle's contributions are the same bits in every mechanism. The case is 16 × 12 and 64 × 48 cells, 64 particles per cell at random offsets, every particle moved by up to half a cell per axis and sorted by the engine. The kernels:

- `recompute`: one work item per node recomputes the stencil of every particle of every cell that reaches it (the N = 1 design of #93);
- `scratch`: one work item per particle writes its `3·M₁·M₂` contributions into a per-particle buffer, then one work item per node reduces them in `recompute`'s order;
- `patch_*`: one work item per cell sums its particles in slot order into a patch, then one work item per node adds the patches of the cells that reach it in increasing linear cell order. `patch_global` accumulates in global memory, `patch_tuple` in a register tuple written once; `_cm` stores the patch cell-major (`[cell, k]` instead of `[k, cell]`);
- `*_nn`: `recompute` and `patch_tuple` with a host-built binning by the nearest node (only order 2 at (Node, Node), where it shrinks the flux stencil from 6 to 5 per axis);

next to the engine's 2D Fast flux and the engine's 2D `deposit!` (Reference) on the same particles.

Checks (sidecars, every case at both thread counts): `scratch` is bitwise equal to `recompute`, as the order says; the four patch variants are bitwise equal to each other and differ from `recompute` in the last bits (relative difference at most 1.0e-6, Float32); Fast agrees within 1.2e-6; the nearest-node variants agree within 1.4e-6 after the fold (a bin that wraps the seam deposits into the other side's guards, so the comparison is on folded fields).

ns per particle, 64 × 48 cells, 1 thread (`flux_prototypes_t1.json`; 16 × 12 is in the sidecar and the figure):

| case | recompute | scratch | patch global | patch tuple | patch tuple cm | recompute nn | patch nn | Fast | `deposit!` Ref |
|---|---|---|---|---|---|---|---|---|---|
| Float32, order 1, (Node, Node) | 280 | 38.7 | 18.1 | 18.9 | 15.2 | – | – | 189 | 5.01 |
| Float32, order 1, (Center, Center) | 657 | 74.1 | 30.6 | 30.6 | 32 | – | – | 303 | 60.5 |
| **Float32, order 2, (Node, Node)** | **1360** | **110** | **44.3** | **39.8** | **42.5** | **776** | **35.4** | **444** | **107** |
| Float32, order 2, (Center, Center) | 628 | 71 | 29 | 29.2 | 31.7 | – | – | 305 | 15.8 |
| Float64, order 1, (Node, Node) | 245 | 61.1 | 17.6 | 21.3 | 19.1 | – | – | 190 | 4.76 |
| Float64, order 1, (Center, Center) | 763 | 106 | 35.7 | 36.6 | 36.3 | – | – | 317 | 63.2 |
| **Float64, order 2, (Node, Node)** | **1400** | **133** | **43.7** | **48.3** | **45.8** | **830** | **36.4** | **448** | **110** |
| Float64, order 2, (Center, Center) | 631 | 107 | 31.1 | 31.1 | 32 | – | – | 317 | 16.2 |

The same at 32 threads (`flux_prototypes_t32.json`):

| case | recompute | scratch | patch global | patch tuple | patch tuple cm | recompute nn | patch nn | Fast | `deposit!` Ref |
|---|---|---|---|---|---|---|---|---|---|
| Float32, order 1, (Node, Node) | 25.4 | 6.92 | 2.15 | 1.89 | 1.54 | – | – | 37.4 | 0.61 |
| Float32, order 1, (Center, Center) | 62.6 | 13.4 | 3.18 | 3.21 | 3.13 | – | – | 60.7 | 5.34 |
| **Float32, order 2, (Node, Node)** | **120** | **21.3** | **4.24** | **4.12** | **4.47** | **64.3** | **3.43** | **91.7** | **6.74** |
| Float32, order 2, (Center, Center) | 60.3 | 13.1 | 3.1 | 3.04 | 3.28 | – | – | 61.4 | 1.67 |
| Float64, order 1, (Node, Node) | 29.8 | 14.1 | 2.78 | 2.87 | 2.97 | – | – | 33.7 | 0.609 |
| Float64, order 1, (Center, Center) | 82.9 | 30.4 | 4.44 | 5.38 | 5.06 | – | – | 53.2 | 4.23 |
| **Float64, order 2, (Node, Node)** | **159** | **66** | **7.2** | **7.31** | **7.03** | **92.8** | **4.31** | **79** | **8.23** |
| Float64, order 2, (Center, Center) | 83 | 30.3 | 4.71 | 4.6 | 4.72 | – | – | 52.7 | 2.72 |

Geometric means over the 8 cases per stagger (both grid sizes, orders, element types) of the time relative to `patch_tuple_cm`:

| | recompute, 1 thread | 32 threads | scratch, 1 thread | 32 threads | Fast, 1 thread | 32 threads |
|---|---|---|---|---|---|---|
| (Node, Node) | 21.8 | 13.5 | 2.36 | 2.50 | 10.6 | 17.2 |
| (Center, Center) | 20.6 | 13.4 | 2.21 | 2.34 | 9.62 | 15.9 |

Among the patch variants, `patch_tuple_cm` against `patch_tuple` is 0.92 to 1.01 and `patch_global` against `patch_tuple` is 0.95 to 1.03 (geometric means per stagger and thread count): a tie within the spread of single cases. `patch_global_cm` is 2.7 to 3.1 times slower: on the CPU each read-modify-write strides by the cell count.

**Choice: per-cell patches, accumulated in a register tuple, stored cell-major** (contract 13). The patch is 13 to 22 times cheaper than per-node recomputation and more than twice as cheap as per-particle scratch, and its scratch is per cell, not per particle (section 4). Between the tied variants the register tuple writes each entry once and needs no zeroing pass. Cell-major was taken because neighbouring work items (cells) then touch neighbouring addresses, which should coalesce on a GPU; on the CPU it ties, and the GPU expectation is not measured (no GPU run in this package). Accumulating a `Dual{Float32,3}` patch of order 2 at (Node, Node) in registers takes 432 values per work item, which a GPU would spill; that is part of the GPU follow-up.

**N = 1 stays as it is.** Its Reference flux keeps the per-node recomputation and with it T-260's digests and the electromagnetic module's bits. The engine's 2D Reference flux now costs less per particle than the 1D one at the same particle count (section 3: 0.64 to 0.76 of it on one thread). Moving N = 1 to the patch mechanism changes the N = 1 bits and the module's digests, which is outside this package's scope (`lib/ParticleMeshCore` only); it is left as a follow-up on #93.

**Nearest-node binning** (measured only). For the patch it saves 11 to 25 % at 1 thread and 11 to 41 % at 32 threads (`patch nn` against `patch tuple`, order 2 at (Node, Node), per case: 0.85, 0.89, 0.76, 0.75 at 1 thread; 0.89, 0.83, 0.65, 0.59 at 32 threads). Per-node recomputation saves about 42 to 47 %. The numbers favour it moderately for the chosen mechanism: it changes the sort key (contract 9) and the bits of `deposit!` at `Node()` staggers, so it goes to its own follow-up package.

## 3. The engine (ported)

`scripts/engine2d/flux_timing.jl ported` times the engine's `deposit_flux!` in both modes against the engine's 2D `deposit!` on the same particles, and against the engine's 1D Reference and Fast flux at the same particle count (`prod(n)` cells of 64), 1 and 32 threads. The engine's Reference matches its prototype (`patch_tuple_cm`: 44.9 against 42.5 ns per particle, Float32, order 2, (Node, Node), 1 thread, in two processes). ns per particle (`flux_ported_t{1,32}.json`):

| case | flux Ref, 1 thread | flux Fast | `deposit!` Ref | `deposit!` Fast | 1D flux Ref | 1D flux Fast | flux Ref, 32 threads | flux Fast | `deposit!` Ref | `deposit!` Fast | 1D flux Ref | 1D flux Fast |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Float32, 16 × 12, order 1, (Node, Node) | 20 | 195 | 6.19 | 17.1 | 27.8 | 48.6 | 7.59 | 197 | 1.94 | 16.7 | 8.76 | 47.4 |
| Float32, 16 × 12, order 1, (Center, Center) | 49.2 | 315 | 69.9 | 19.7 | 53.3 | 61.1 | 12.8 | 296 | 9.5 | 35.9 | 14.4 | 41.1 |
| Float32, 16 × 12, order 2, (Node, Node) | 41.4 | 458 | 108 | 35.6 | 83.3 | 78 | 18.4 | 417 | 18.7 | 44.6 | 23.1 | 60.7 |
| Float32, 16 × 12, order 2, (Center, Center) | 35.1 | 316 | 16.3 | 35.7 | 47.4 | 80 | 13.9 | 312 | 3.96 | 46.4 | 17.8 | 48 |
| Float32, 64 × 48, order 1, (Node, Node) | 20 | 187 | 6.27 | 16.3 | 28.6 | 48.1 | 2.08 | 41 | 0.682 | 2.58 | 2.85 | 4.23 |
| Float32, 64 × 48, order 1, (Center, Center) | 32.5 | 311 | 63 | 16.8 | 55.6 | 62.9 | 3.47 | 63.1 | 4.23 | 3.69 | 4.83 | 5.93 |
| **Float32, 64 × 48, order 2, (Node, Node)** | **44.9** | **449** | **108** | **35.3** | **83.9** | **75.3** | **5.49** | **93.5** | **6.61** | **8.66** | **7** | **6.85** |
| Float32, 64 × 48, order 2, (Center, Center) | 36.2 | 305 | 16.1 | 35 | 47.8 | 59.7 | 3.72 | 61.7 | 1.8 | 7.31 | 4.84 | 5.09 |
| Float64, 16 × 12, order 1, (Node, Node) | 19.3 | 192 | 5.48 | 19.7 | 24 | 48.1 | 12.3 | 138 | 2.07 | 10.1 | 10.8 | 23.7 |
| Float64, 16 × 12, order 1, (Center, Center) | 37 | 338 | 64.2 | 17.1 | 56 | 64.2 | 27.9 | 240 | 9.59 | 12.5 | 16.9 | 19.2 |
| Float64, 16 × 12, order 2, (Node, Node) | 46.1 | 468 | 109 | 52.4 | 80.5 | 83.4 | 29.1 | 335 | 13.9 | 29.4 | 33.7 | 34.2 |
| Float64, 16 × 12, order 2, (Center, Center) | 31.5 | 325 | 18.4 | 42.1 | 43.6 | 60.2 | 34.9 | 235 | 3.79 | 29.5 | 13.7 | 15.2 |
| Float64, 64 × 48, order 1, (Node, Node) | 19.5 | 201 | 4.94 | 16.3 | 24.9 | 47.4 | 3.26 | 34.9 | 0.686 | 2.7 | 2.56 | 4.52 |
| Float64, 64 × 48, order 1, (Center, Center) | 38.3 | 329 | 64.5 | 16.5 | 55.2 | 61.2 | 4.7 | 53.4 | 4.71 | 3.27 | 5.2 | 5.03 |
| **Float64, 64 × 48, order 2, (Node, Node)** | **47.8** | **452** | **112** | **35.5** | **83.8** | **75.8** | **7.77** | **78.5** | **7.4** | **7.97** | **8.14** | **8.14** |
| Float64, 64 × 48, order 2, (Center, Center) | 32.1 | 341 | 15.7 | 35.7 | 44.6 | 60.4 | 4.63 | 51.4 | 1.96 | 7.42 | 4.44 | 4.62 |

Geometric means over orders and element types, per stagger and grid size:

| | Ref / Fast, 1 thread | 32 threads | Ref / `deposit!` Ref, 1 thread | 32 threads | Ref / 1D flux Ref, 1 thread | 32 threads |
|---|---|---|---|---|---|---|
| (Node, Node), 16 × 12 | 0.098 | 0.060 | 1.16 | 2.63 | 0.64 | 0.91 |
| (Node, Node), 64 × 48 | 0.102 | 0.072 | 1.22 | 1.89 | 0.64 | 0.91 |
| (Center, Center), 16 × 12 | 0.116 | 0.076 | 1.10 | 3.35 | 0.76 | 1.31 |
| (Center, Center), 64 × 48 | 0.108 | 0.072 | 1.09 | 1.41 | 0.69 | 0.85 |

- **Against Fast.** The 2D Reference flux costs 6 to 12 % of the 2D atomic flux. Atomics on this CPU backend cost about 4 ns each on one thread, and the 2D flux issues `3·M₁·M₂` of them per particle (108 at order 2, (Node, Node)). Fast is not the fast path on the CPU; on a GPU this has not been measured.
- **Against `deposit!`.** The means mix two regimes. At (Node, Node) and order 2 the flux costs 0.42 of the 2D Reference `deposit!` at 1 thread (44.9 against 108), which still recomputes per node with a 4 × 4 cell reach. At 32 threads the two are within a factor of about 2 either way. At the natural staggers `deposit!` is cheaper, as its per-node work is small.
- **At 32 threads on 16 × 12 cells** (768 cells, 49 152 particles) the per-cell launch has too little work per thread, and the times there carry the launch overhead (WP-018 saw the same for 1152 particles).
- **N = 1 before and after.** The N = 1 kernels are unchanged, and their T-260 digests are reproduced; the 1D Reference flux sharing the per-axis weight helper `flux_axis_weights` changes no bit. Its cost is in the table above, against WP-009's figures on the same CPU model (`reports/wp-009-flux.md`, a different case).

## 4. Scratch per particle and per cell, and A7's memory budget

`flux_scratch.json`: values of `T` per particle (scratch) or per linear cell (patch) for the 2D flux stencil, and the engine's measured `Workspace` with `flux = (shape, stagger)` (64 × 48 cells, 16 per cell; the per-cell column is the difference to the workspace without `flux`, and equals the patch exactly):

| T | order, ρ's stagger | values | per-particle scratch, B/particle | patch, B/cell | nearest-node patch, B/cell |
|---|---|---|---|---|---|
| Float32 | 1, (Node, Node) | 48 | 192 | 192 | – |
| Float32 | 1, (Center, Center) | 75 | 300 | 300 | – |
| Float32 | **2, (Node, Node)** | **108** | **432** | **432** | **300** |
| Float32 | 2, (Center, Center) | 75 | 300 | 300 | – |
| Float64 | 2, (Node, Node) | 108 | 864 | 864 | 600 |
| Float64 | 2, (Center, Center) | 75 | 600 | 600 | – |
| Dual{Float32,3} | 2, (Node, Node) | 108 | 1728 | 1728 | 1200 |
| Dual{Float32,3} | 2, (Center, Center) | 75 | 1200 | 1200 | – |

(Order 1 for Float64 and the Dual is in the sidecar: 384 and 768 B at (Node, Node).) The rest of the `Workspace` is unchanged: 33 B per particle for Float32 and 37 B for Float64 plus 8 B per cell (contract 9).

Against A7's budget (a 16 GB card, 2D at roughly 10⁷ particles), Float32, order 2, ρ at (Node, Node): per-particle scratch would take 4.3 GB per species at 10⁷ particles (8.6 GB for Float64, 17 GB for `Dual{Float32,3}`, more than the card). The patch takes 432 B per cell: 270 MB per species at 16 particles per cell (6.25·10⁵ cells), 68 MB at 64 per cell, next to the existing 330 MB of the workspace. The patch lives in each species' workspace. A module with several species could share one, since it is reduced within the call; that is not done here.

## 5. Tests

T-300 to T-309 (`lib/ParticleMeshCore/test/flux2d.jl`) pass, with T-110 to T-117 and T-260 to T-269 unchanged. The full `ParticleMeshCore` suite passes at 1 thread (28 416 tests), T-303's subprocesses at 1 and 4 threads included. The GPU testsets did not run (no functional GPU in this session). The measured ratios and proposed multiples are in `tolerances.toml`, `status = "proposed"`:

- **T-300** (continuity): up to 2.41 of `eps·max|ρ|/Δt`, C = 8.
- **T-302** (Reference against Fast per node): 3.74, C = 32.
- **T-304** (reduction to 1D): 3.43, C = 8; Jy is exactly zero.
- **T-305** (Dual partials): 2.03, C = 16. The primal of the Dual run is bitwise the Float32 run for both orders and all four stagger pairs, so the bitwise assertion stands, with no measured fallback.
- **T-307** (C¹ through a crossing and the corner): the ratio to the finite-difference floor is at most 1.25 against C = 2, the floor at most 1.95e-8.
- **T-309** (sums): 1.59, C = 16.

The hand evaluation of T-301 (Esirkepov's weights written out with host B-splines) agrees per node within 1.25 eps of the prefactor. Its 1D check of the transverse average agrees within 1.59 eps.

## 6. Open

- No GPU run (non-goal): the 2D flux on CUDA, the cell-major layout's coalescing and the register pressure of the Dual patch are a follow-up for the GPU runner after merge.
- Nearest-node binning for even orders at `Node()` staggers: a follow-up issue (section 2).
- N = 1 on the patch mechanism, which changes its bits: a follow-up on #93.
- Esirkepov's paper not read; the scheme is verified against the two codes' sources only.
