# WP-068: the first optimizations of the trapped-electron objective on the GPU

**Summary.**
- L-BFGS-B maximized WP-065's J (the kinetic energy of the trapped electrons of the first bucket, any origin) at the a₀ = 1.9 energy budget, at λ₀/32 with the compensated gather in Float32 on an RTX 5090, from four starts: in WP-068's box, and under a peak power cap that at the fixed energy is a floor on the duration (CETAL-PW's 1 PW at 25 J carried to the case by τ/λ_p: τ ≥ 2.45/√n, up to τ = 12). Under the cap it also maximized J_born, the part of the ionization's electrons alone, and J_born + κ·tanh(J_plasma/κ) with κ = 0.05, where the plasma's own share adds at most 0.05.
- In the box, all four starts ended at focus 110 and density 0.10 n_c, their upper bounds. The best is J = 10.93 at (τ, w, focus, n) = (4.98, 7.90, 110, 0.10), a₀ = 2.34, against 0.026 at the base point. 98.4 % of it is the plasma's own electrons injected into the first bucket.
- Under the cap, all four again ended at focus 110 and 0.10 n_c, with τ at or just above the floor of 7.76 there. The best is J = 0.444 at (8.24, 6.47, 110, 0.10), a₀ = 2.01, and the others reached 0.23 to 0.31. 60 % of the best is still the plasma's own injection: at focus 110 and 0.10 n_c it sets in above a₀ ≈ 2.05 with the box's short pulses and already near 1.83 with the floor's longer ones. Without any plasma share, the highest J the capped run met is 0.20, at a₀ = 1.85.
- With the plasma's share left out, all four starts reach the same pulse: τ on the floor (7.76 at 0.10 n_c), focus 110 and density 0.10 n_c on their bounds, and the waist inside its range at 7.35, a₀ = 1.94. With the share capped, three starts end next to it (waist 6.90 to 7.54, the best J = 0.2685 at 7.38, a₀ 1.94) and the fourth at a waist of 8.32 with J = 0.227. J_born there is 0.2216 in both runs, 25 % above the capped optimum's 0.178 and 9 % above the best point with no plasma share the capped run met. The plasma still injects at that point (J_plasma 0.081 to 0.086), which neither objective rewards.
- The born-only optimum is converged and smooth: J_born is 0.22160 at λ₀/32, 0.22166 at λ₀/48 and 0.22157 at λ₀/64, L-BFGS-B started from it at λ₀/48 stops without a step (its projected gradient is below `pgtol`), and Float64 ForwardDiff agrees with central differences to 0.002 to 0.06 in d ln J/dz at λ₀/32 and 0.005 to 0.12 at λ₀/48. With the share capped, the agreement is 0.001 to 0.18 and J is +0.6 % at λ₀/48.
- The born electrons' part is about the same at the first two optima (J_born 0.173 and 0.178) and converged in resolution: at λ₀/48 the capped optimum's J_born is 0.1785 against 0.1778, while its J_plasma falls from 0.266 to 0.147. J at λ₀/48 is +8 % at the box's optimum and −27 % at the capped one.
- The gradient holds in the box: Float64 ForwardDiff agrees with central differences to 0.003 to 0.006 in d ln J/dz in duration, waist and density, and in focus to 5·10⁻⁴ at a step of 2⁻¹⁶ (0.21 at 2⁻¹³, where J's roughness dominates). Under the cap the optimum sits on the injection's onset, where the gradient is noise: Float32 and Float64 duals give −0.77 and 20.1 in waist, and central differences −19.0.
- Cost on the RTX 5090: a run with duals in four directions 10.6 s in the box (card shared) and 13.3 s under the cap (alone, with the window 3 λ₀ longer). A plain run is 4 to 5.4 s with two at a time. Each run's optimizations took 19 to 40 minutes, its scan of about 1,000 plain runs 35 to 44 minutes.
- The ionization is ADK without barrier suppression and takes no energy from the field (WP-059, WP-063), so the trapped charge is not a physical prediction.

| | |
|---|---|
| Work package | WP-068 (`trace/wp/WP-068-first-gpu-optimizations.md`); goals A0, A3, A7, A8 |
| Commits | box run: optimizers and check `5d502f1`, scan `db1309a`, report `eee5099` (`dirty = false`); capped run: `a56b788` (`dirty = false`); born-only, capped-share and λ₀/48 runs: `b06a59c` (`dirty = false`, the tree of `5c985b5`); recorded frames: `b06a59c` and `eeb3e0a` (`trapped2d_frames.jl` was untracked at `b06a59c`, the manifests hash the same file `eeb3e0a` commits). The later commits before each change only the scan, the report and tests. The figures here are rebuilt by the report script at the pull request's head, the scan figures at 1 px per unit to stay within F3's 1 MB per file |
| Runs | `wp068-opt`, `wp068-power`, `wp068-born`, `wp068-pcap` and `wp068-born48` on branches `runs/<run>-5090`; `scripts/optim/trapped2d_opt.jl opt` (`cap=cetal25` for all but the first; `objective=born`, `objective=plasmacap kappa=0.05`; `ppl=48 ny=3 maxiters=12 from=scripts/runs/wp068-born` for the last), `check` (the last at `ppl=64 ny=4 fdppl=48 fdny=3`), `scan` (n = 11, two shards; not for the last) and `report`; frames of single runs at the optima, `scripts/optim/trapped2d_frames.jl`, in `wp068-frames` on `runs/wp068-frames-cpu` |
| Input hashes | SHA-256 of every input in each run's `manifest.toml` under `wp-068-first-gpu-optimizations/` |
| Environment | Julia 1.13.4; NVIDIA GeForce RTX 5090, KernelAbstractions CUDA, Float32 (Float64 for the check's gradient and differences), Reference mode; AMD Ryzen 9 9950X host. The frames: KernelAbstractions CPU, 4 threads of an Intel Xeon at 2.1 GHz (a cloud VM), Float32, the optimization's numerics; their final J equals the GPU optimum's to 10⁻⁴ relative |
| Figures | `wp-068-first-gpu-optimizations/trapped2d_opt.png`, `trapped2d_opt_scan.png` (box), `power/` (cap), `born/`, `pcap/` and `born48/`, each with its JSON sidecar |
| Status | measured; the born-only and capped-share runs are not rerun by a breaker; the box run's four headline numbers were rerun by a physics breaker on the CPU in Float64, Fast deposition and a longer window, and agree; not independently reviewed, so no number here is to leave the repository before that review |

## Setup

The case is WP-065's (`scripts/optim/trapped2d_case.jl`) with `gather = CompensatedGather()`, at λ₀/32 and Δy = π/8, the delay fixed at 54, the budget of a₀ = 1.9 at τ = 6, w = 10. `TRAPPED_BOX` is duration 4 to 9, waist 6 to 14, focus 64 to 110, density 0.05 to 0.10 n_c, z = x/(6, 10, 80, 0.072). Starts: base (6, 10, 80, 0.072), long (8, 12, 100, 0.06), short (5, 8, 70, 0.09), narrow (7, 7, 104, 0.08). L-BFGS-B (OptimizationLBFGSB), `maxiters = 25`, `pgtol` 2⁻⁶ of the base point's largest gradient component in z. Each point is one run with duals in the four free directions, recorded as JSON by the point's bits, so a stopped optimization continues from its records. The check runs the best optimum at λ₀/48 (plain and with duals, Float32) and at λ₀/32 in Float64 with duals and central differences at 2⁻¹³; the scan runs J plain on 11 × 11 grids in the plane of each path and over the box for each pair of parameters through the best optimum.

Under the cap (`cap=cetal25`, `cetal_cap`): at a fixed energy the peak power is W/τ_FWHM whatever the waist, so the cap is τ_FWHM ≥ (25 fs/T_p)·λ_p with T_p = 2π/ω_p at 4.2·10¹⁸ cm⁻³, the middle of CETAL's jet: τ ≥ 2.455/√n, 7.76 at 0.10 n_c and 10.98 at 0.05 n_c. The duration lies between that floor and 12, and the solver's first coordinate is its place between the two, from 0 to 1 (`Coords`); the starts keep their waist, focus and density, with the duration at the same place between floor and 12 as it had between 4 and 9. A pulse of τ = 12 is still being emitted when the case's window passes its antenna, so every capped run has the window 3 λ₀ longer behind (`window_extend`) with the objective's position weight moved back by 6π; that leaves J unchanged to 10⁻¹² at λ₀/16 (T-976).

## Results

![paths](wp-068-first-gpu-optimizations/trapped2d_opt.png)

![paths under the cap](wp-068-first-gpu-optimizations/power/trapped2d_opt.png)

![J around the paths under the cap](wp-068-first-gpu-optimizations/power/trapped2d_opt_scan.png)

![paths, born electrons only](wp-068-first-gpu-optimizations/born/trapped2d_opt.png)

![J_born around the paths](wp-068-first-gpu-optimizations/born/trapped2d_opt_scan.png)

![paths, plasma's share capped](wp-068-first-gpu-optimizations/pcap/trapped2d_opt.png)

From `wp-068-first-gpu-optimizations/tables.md` and the `tables.md` of `power/`, `born/`, `pcap/` and `born48/` (printed by `trapped2d_opt.jl report`), in the box:

| start | from (τ, w, focus, n) | a₀ | J there | to (τ, w, focus, n) | a₀ | J there | J ratio | on the box's bounds | stop | iterations | points run | largest dark current weight on the path | largest J_plasma/J on the path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 6.0, 10.0, 80.0, 0.072 | 1.9 | 0.02601 | 4.98, 7.897, 110.0, 0.1 | 2.343 | 10.93 | 420.1 | focus=hi, density=hi | Success | 10 | 28 | 0.4029 | 0.991 |
| long | 8.0, 12.0, 100.0, 0.06 | 1.503 | 0.007473 | 4.957, 6.331, 110.0, 0.1 | 2.617 | 8.094 | 1083.0 | focus=hi, density=hi | Success | 9 | 36 | 0.567 | 0.991 |
| short | 5.0, 8.0, 70.0, 0.09 | 2.324 | 0.01231 | 5.083, 7.398, 109.8, 0.1 | 2.395 | 10.46 | 850.0 | density=hi | MaxIters | 25 | 64 | 0.8062 | 0.9874 |
| narrow | 7.0, 7.0, 104.0, 0.08 | 2.097 | 0.1188 | 5.066, 7.868, 110.0, 0.1 | 2.328 | 10.91 | 91.82 | focus=hi, density=hi | Success | 12 | 34 | 0.6622 | 0.991 |

Under the cap:

| start | from (τ, w, focus, n) | a₀ | J there | to (τ, w, focus, n) | a₀ | J there | J ratio | on the box's bounds | stop | iterations | points run | largest dark current weight on the path | largest J_plasma/J on the path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 10.29, 10.0, 80.0, 0.072 | 1.451 | 0.001794 | 8.235, 6.472, 110.0, 0.1 | 2.009 | 0.4439 | 247.5 | focus=hi, density=hi | Success | 8 | 31 | 0.2487 | 0.6084 |
| long | 11.6, 12.0, 100.0, 0.06 | 1.248 | 0.0003632 | 7.762, 7.093, 110.0, 0.1 | 1.979 | 0.2967 | 817.0 | τ: floor→12=lo, focus=hi, density=hi | Success | 6 | 31 | 0.2834 | 0.2574 |
| short | 8.946, 8.0, 70.0, 0.09 | 1.737 | 0.003457 | 7.762, 8.288, 110.0, 0.1 | 1.833 | 0.2338 | 67.64 | τ: floor→12=lo, focus=hi, density=hi | Success | 7 | 30 | 0.6921 | 0.1166 |
| narrow | 10.67, 7.0, 104.0, 0.08 | 1.698 | 0.02262 | 8.016, 7.462, 110.0, 0.1 | 1.899 | 0.3075 | 13.6 | focus=hi, density=hi | Success | 10 | 33 | 0.6612 | 0.3517 |

Under the cap, J = J_born (J_plasma/J is the plasma's part over J_born, which the objective leaves out):

| start | from (τ, w, focus, n) | a₀ | J there | to (τ, w, focus, n) | a₀ | J there | J ratio | on the box's bounds | stop | iterations | points run | largest dark current weight on the path | largest J_plasma/J on the path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 10.29, 10.0, 80.0, 0.072 | 1.451 | 0.001794 | 7.762, 7.348, 110.0, 0.1 | 1.945 | 0.2216 | 123.5 | τ: floor→12=lo, focus=hi, density=hi | Success | 7 | 21 | 0.2897 | 1.368 |
| long | 11.6, 12.0, 100.0, 0.06 | 1.248 | 0.0003632 | 7.762, 7.346, 110.0, 0.1 | 1.945 | 0.2216 | 610.2 | τ: floor→12=lo, focus=hi, density=hi | Success | 6 | 15 | 0.7129 | 0.3706 |
| short | 8.946, 8.0, 70.0, 0.09 | 1.737 | 0.003457 | 7.762, 7.348, 110.0, 0.1 | 1.945 | 0.2216 | 64.11 | τ: floor→12=lo, focus=hi, density=hi | Success | 10 | 37 | 0.6714 | 1.069 |
| narrow | 10.67, 7.0, 104.0, 0.08 | 1.698 | 0.02262 | 7.762, 7.346, 110.0, 0.1 | 1.945 | 0.2216 | 9.797 | τ: floor→12=lo, focus=hi, density=hi | Success | 6 | 10 | 0.3677 | 0.8358 |

Under the cap, J = J_born + 0.05·tanh(J_plasma/0.05):

| start | from (τ, w, focus, n) | a₀ | J there | to (τ, w, focus, n) | a₀ | J there | J ratio | on the box's bounds | stop | iterations | points run | largest dark current weight on the path | largest J_plasma/J on the path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| base | 10.29, 10.0, 80.0, 0.072 | 1.451 | 0.001794 | 7.762, 6.899, 110.0, 0.1 | 2.006 | 0.2675 | 149.1 | τ: floor→12=lo, focus=hi, density=hi | Success | 8 | 23 | 0.2753 | 1.067 |
| long | 11.6, 12.0, 100.0, 0.06 | 1.248 | 0.0003632 | 7.762, 7.375, 110.0, 0.1 | 1.941 | 0.2685 | 739.2 | τ: floor→12=lo, focus=hi, density=hi | Success | 8 | 33 | 0.6828 | 0.3198 |
| short | 8.946, 8.0, 70.0, 0.09 | 1.737 | 0.003457 | 7.762, 8.324, 110.0, 0.1 | 1.829 | 0.2266 | 65.55 | τ: floor→12=lo, focus=hi, density=hi | Success | 7 | 29 | 0.7305 | 0.09957 |
| narrow | 10.67, 7.0, 104.0, 0.08 | 1.698 | 0.02262 | 7.762, 7.54, 110.0, 0.1 | 1.92 | 0.2642 | 11.68 | τ: floor→12=lo, focus=hi, density=hi | Success | 7 | 33 | 0.4649 | 0.248 |

The short start of the capped-share run stopped at a waist of 8.32, where J_born is lower (J = 0.2266, J_plasma/J ≈ 0.10): along its path the plasma's share switches on and off between points 10⁻³ apart, so its gradient in waist and focus jumps (d ln J/dz from −8 to +9 in waist) and the line search settled there. The other three reached 0.264 to 0.2685.

The checks at each run's best optimum:

| run at the optimum of `base` | J | d ln J/dz (τ, w, focus, n) | dark weight | J_plasma/J | seconds |
|---|---|---|---|---|---|
| box: λ₀/32, Float32, duals | 10.927 | 0.024, -0.0194, 3.71, 2.27 | 0.2679 | 0.9842 | 22.5 |
| box: λ₀/32, Float64, duals | 10.9269 | 0.0461, -0.0304, 3.62, 2.24 | 0.2679 | 0.9842 | 40.8 |
| box: λ₀/48, Float32, duals | 11.7976 | 1.14, 0.441, 2.17, 1.1 | 0.1936 | 0.9865 | 42.3 |
| cap: λ₀/32, Float32, duals | 0.443914 | 3.29, -0.773, 9.85, 18.9 | 0.2043 | 0.5995 | 13.3 |
| cap: λ₀/32, Float64, duals | 0.44386 | 4.51, 20.1, 8.81, 26.4 | 0.1966 | 0.5994 | 50.5 |
| cap: λ₀/48, Float32, duals | 0.325655 | -11.2, 2.48, -5.65, 0.872 | 0.2673 | 0.4519 | 50.3 |
| born only (optimum of `short`): λ₀/32, Float32, duals | 0.221599 | -1.86, -0.00467, 2.77, 1.16 | 0.2801 | 0.3663 | 13.2 |
| born only: λ₀/32, Float64, duals | 0.221616 | -1.85, -0.00828, 2.77, 1.15 | 0.2859 | 0.3664 | 49.8 |
| born only: λ₀/48, Float32, duals | 0.221658 | -1.79, -0.0146, 2.76, 1.06 | 0.2712 | 0.5042 | 49.7 |
| born only: λ₀/48, Float64, duals | 0.221718 | -1.79, -0.0157, 2.75, 1.07 | 0.2652 | 0.5048 | 134.0 |
| born only: λ₀/64, Float32, duals | 0.221573 | -1.78, -0.0254, 2.7, 1.09 | 0.2116 | 0.5231 | 105.0 |
| capped share (optimum of `long`): λ₀/32, Float32, duals | 0.268456 | -1.65, -0.413, 2.33, 2.63 | 0.2979 | 0.3198 | 13.2 |
| capped share: λ₀/32, Float64, duals | 0.26847 | -1.63, 0.61, 2.15, 2.69 | 0.3075 | 0.3196 | 49.8 |
| capped share: λ₀/48, Float32, duals | 0.270155 | -1.4, -1.83, 2.18, 1.58 | 0.2904 | 0.3885 | 49.9 |

The λ₀/48 and λ₀/64 rows of the born-only optimum come from `wp068-born48`: L-BFGS-B started at the λ₀/32 optimum on the λ₀/48 grid (Δy = π/12) and took no step, since τ, focus and density push against their bounds and the waist's component of the projected gradient, 0.0032 in z, is below `pgtol` (0.0096); its check is at λ₀/64 and Δy = π/16, with central differences at λ₀/48.

![the born-only optimum re-optimized at λ₀/48](wp-068-first-gpu-optimizations/born48/trapped2d_opt.png)

## Not shown

The rest of each `tables.md`: every point each solver reported, the plain λ₀/48 runs (equal to the dual runs' J), and the central differences at 2⁻¹³ against Float64 ForwardDiff (box: 0.003, 0.005, 0.21, 0.006 in d ln J/dz; cap: 0.36, 39.0, 5.0, 9.8; born only: 0.0016, 0.0021, 0.050, 0.057, and at λ₀/48 0.019, 0.0051, 0.073, 0.12; capped share: 0.0011, 0.026, 0.18, 0.077). The born-only and capped-share runs cost 84 and 119 runs with duals, median 13.2 s each, and their scans 995 and 990 plain runs. Along the capped paths, d ln J/dz jumps between −141 and +145 in waist among points 10⁻³ apart once the plasma's share is above zero; the solver stopped on its relative change in J with a projected gradient of 1.03 against `pgtol` 2.4·10⁻⁴, and its base start passed J = 0.458 at point 9 before ending at 0.444. After converging, L-BFGS-B's line search asked again for points within roundoff of the optimum; those came from the records and cost no run. The box run's scan (`trapped2d_opt_scan.png`) and the physics breaker's runs: CPU and GPU J agree to 2·10⁻¹³ in Float64, Fast and Reference deposition to 5·10⁻¹³, the plain gather gives 7.5 % less J than the compensated one, and the counted bunch sits at window x ≈ 31 to 34, inside the depleted pulse's remnant (on-axis |E_y| peak at 29.6).

## Open

- With J of any origin, both the box and the capped run end on the plasma's own injection; under the cap on its onset, where J is not converged in resolution (−27 % at λ₀/48, all of it in J_plasma) and the gradient is noise. Leaving the plasma's share out, or capping it, gives a converged optimum with a sound gradient, at the same pulse in both forms.
- Focus and density are on their bounds in every run, and under the cap the duration is on its floor, so the optima move with the box and the cap; the box was set by the draft, not by physics. The next step is a box from the experiment (the jet's density range, the focal positions the setup reaches).
- At the born-only optimum the plasma still injects (J_plasma = 0.081 at λ₀/32, 0.112 at λ₀/48, 0.116 at λ₀/64): the born bunch shares the first bucket with self-injected charge the objective ignores.
- At the box's optimum the λ₀/48 gradient in duration and waist (1.14, 0.44) says that optimum is not converged in those two at λ₀/32.

## Data

`wp-068-first-gpu-optimizations/wp068-opt/`, `wp068-power/`, `wp068-born/`, `wp068-pcap/` and `wp068-born48/` hold each run's `manifest.toml` and `published.toml`, which names its run branch with every point's record, each start's `result.json`, the check's and the scan's runs and the SHA-256 of each. `scripts/rundata/run-data.sh fetch reports/wp-068-first-gpu-optimizations/<run>/published.toml` brings a run back into `scripts/runs/<run>/`, and `julia +1.13 --project=scripts scripts/optim/trapped2d_opt.jl report <out dir> scripts/runs/<run>` rebuilds its figures and tables (`reports/wp-068-first-gpu-optimizations` for `wp068-opt`, its subdirectories `power`, `born`, `pcap` and `born48` for the others). `frames/<label>/` holds the manifest and `published.toml` of each frame run (`runs/wp068-frames-cpu`): the optima of the box, capped, born-only and capped-share runs, the base point, and the best point with no plasma share on the capped run's paths (7.95, 7.92, 110, 0.10).
