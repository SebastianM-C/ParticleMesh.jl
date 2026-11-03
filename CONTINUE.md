# CONTINUE: WP-032, ParticleMesh's oblique Landau ensemble on the CPU box

Branch `claude/epic-cerf-3g9o79`. Batch `wp032-pm-landau`, 8 requests, submitted with
`scripts/cpu-box.sh submit`:

| tag | seeds | also |
|---|---|---|
| `wp032-pm-r1` | 20261031 20262031 20263031 20264031 (WP-023's four) | Smilei seed 20261031 first (`scripts/runs/wp032-smilei-timing`, analysed on the box) |
| `wp032-pm-r2` | 20265031 20266031 20267031 20268031 | |
| `wp032-pm-r3` | 20269031 20270031 20271031 20272031 | |
| `wp032-pm-r4` | 20273031 20274031 20275031 20276031 | |
| `wp032-pm-r5` | 20277031 20278031 20279031 20280031 | |
| `wp032-pm-r6` | 20281031 20282031 20283031 20284031 | |
| `wp032-pm-r7` | 20285031 20286031 20287031 20288031 | |
| `wp032-pm-r8` | 20289031 20290031 20291031 20292031 | |

Each PM request writes `scripts/runs/wp032-pm-landau-r<n>/` (`scripts/em2d/landau_ensemble.jl run`).
A request whose result has `ok = false` (a spot eviction) reruns on the next box by itself; if it
has no result after its timeout plus an hour, resubmit it under `<tag>-rerun` from this branch.

## Remaining steps

1. `scripts/cpu-box.sh status <tag>` for every tag; when all are back, `scripts/cpu-box.sh fetch <tag>` each.
2. Merge: `julia +1.13 --project=scripts scripts/em2d/landau_ensemble.jl merge scripts/runs/em2d-pm-landau-oblique-2d-32 scripts/runs/box-wp032-pm-r{1..8}/wp032-pm-landau-r{1..8}` (each request directory in its own box directory).
3. Compare: `julia +1.13 --project=scripts scripts/em2d/crosscode.jl compare scripts/runs/em2d-pm-landau-oblique-2d-32` (takes WP-025's 32-seed Smilei and PIConGPU ensembles by the case hash `431f4192`).
4. Checks: `julia +1.13 --project=scripts scripts/em2d/landau_ensemble.jl checks scripts/runs/em2d-pm-landau-oblique-2d-32 scripts/runs/box-wp032-pm-r1/wp032-smilei-timing/seed-20261031 scripts/runs/em2d-pm-landau-oblique-2d-32/checks`.
5. Sub-ensembles: `scripts/crosscode/subensembles.jl` of the merged run against each reference run.
6. Copy the merged run into `reports/wp-032-pm-landau-ensemble/` (per seed: observables, manifest, sidecars; PNGs only for the first seed, as WP-025), write `reports/wp-032-pm-landau-ensemble.md`, set WP-032 `done`, delete this file, run the breakers, open the PR with auto-merge (CLAUDE.md).
