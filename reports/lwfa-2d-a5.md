# LWFA 2D: A5's first case in ParticleMesh, Smilei and PIConGPU

**Summary.**
- A5's first 2D laser-wakefield case (a₀ = 2, λ₀/48, 10 777 steps, little injection) run from one case file in ParticleMesh, Smilei v5.1 and PIConGPU 0.8.0 and compared on the lattice points they share, with the launched pulses matched.
- ParticleMesh and Smilei agree to 0.004 % in the vacuum amplitude, 0.05 mrad in the carrier and 0.16 % in the wake's E_x. The table has every pair and observable.
- Either code against PIConGPU is 0.07 % and 0.7 mrad apart in vacuum. In amplitude and phase that has the form of a launch term of PIConGPU's, which is modelled and not removed; the centroid distances to PIConGPU are not explained.
- Smilei's trapped bunch is 1.0 % to 1.4 % heavier than ParticleMesh's, and its wake differs from both other codes from t = 700; neither is explained.
- ParticleMesh on the CPU and on CUDA give axis lineouts that differ by at most 2.4 × 10⁻¹¹ at any output, far below every difference between two codes.
- This is agreement between three implementations of one scheme at one resolution. It is not a validation and shows no convergence; `wp-054-lwfa-resolution-scan.md` has the scan over resolution, and A7 has the acceptance rule that came out of both.

| | |
|---|---|
| Work package | WP-053 (`trace/wp/WP-053-lwfa-three-codes.md`); goals A5, A7 |
| Case | `scripts/lwfa/cases/lwfa-2d.toml`, and `lwfa-2d-focus0.toml` (focus on the injection plane) in vacuum only |
| Code and runs | ParticleMesh (CPU box request `runs/lwfa-a5-2`, `5b729a7`: `6ead5bd` plus the run directories) and PIConGPU at `6ead5bd`, Smilei (`runs/lwfa-a5-4-smilei`) with inputs from `2a8d467`; both are commits of PR #413's branch. Each run's manifest is `lwfa-2d-a5/lwfa-a5-4-compare/manifest-<run>.toml`, and `manifest.toml` there has the hashes of the summaries compared |
| Environment | ParticleMesh: Julia 1.13.3, Float64, Reference mode, 16 threads on an AMD EPYC 9655, and CUDA on an RTX 4080 SUPER. Smilei v5.1, 16 OpenMP threads on the same CPU. PIConGPU 0.8.0, CUDA, Float64, on the RTX 4080 SUPER |
| Data | `lwfa-2d-a5/lwfa-a5-4-compare/`: `comparison.json` with every output, and five figures with sidecars. `lwfa-a5-4-focus0-compare/` for the second case, `lwfa-a5-2-cuda-pair/pair.json` for the two backends, `runs/` for the summaries that are on no run branch. The other directories are the three earlier run sets |
| Review | Four independent passes by fresh numerics-reviewer subagents (claude-fable-5-2, 2026-11-12) on the long version of this report in PR #413, findings in `lwfa-2d-a5/review.md`. This short version had one more pass of the same kind, which recomputed the table and every number from the committed data; its findings are applied |

## Results

The largest distance of each pair over the run:

| Largest over the run, at λ₀/48 | ParticleMesh–Smilei | ParticleMesh–PIConGPU | Smilei–PIConGPU |
|---|---|---|---|
| vacuum: carrier phase (mrad) | 0.052 | 0.7 | 0.65 |
| vacuum: centroid (10⁻³ c/ω₀) | 0.77 | 1.5 | 2.0 |
| vacuum: amplitude, relative rms (%) | 0.0038 | 0.066 | 0.065 |
| plasma: carrier phase (mrad) | 0.052 | 0.55 | 0.5 |
| plasma: centroid (10⁻³ c/ω₀) | 0.75 | 1.6 | 1.7 |
| plasma: amplitude, relative rms (%) | 0.032 | 0.056 | 0.056 |
| wake E_x over 1.5 λ_p, relative rms (%) | 0.16 | 0.12 | 0.15 |
| wake minimum, relative difference (%) | 0.086 | 0.1 | 0.11 |
| wake maximum, relative difference (%) | 0.073 | 0.042 | 0.06 |
| maps of E_y, all rows, relative rms (%) | 0.058 | 0.07 | 0.077 |
| maps of E_y, flank, relative rms (%) | 0.07 | 0.13 | 0.14 |
| maps of E_y, far wing, relative rms (%) | 0.29 | 0.68 | 0.66 |
| bunch weight, relative difference (%) | 1.4 | 0.14 | 1.2 |
| highest occupied bin, difference (mₑc²) | 0 | 0.2 | 0.2 |

The table is the output of `julia +1.13 --project=scripts scripts/lwfa/lwfa.jl table reports/lwfa-2d-a5/lwfa-a5-4-compare/comparison.json`, and T-720 holds this text to it. It counts the vacuum outputs from t = 200, the plasma outputs from t = 400, the maps at t = 300, 700 and 1200 and the bunch from t = 800. A distance is a relative rms difference on shared nodes, or the difference of two codes' values; `compare`'s docstrings in `scripts/lwfa/lwfa.jl` define each. The maps leave out the 48 cells at the back of the shared columns: in a few columns there Smilei's E_y differs from both other codes' by 20 % to 40 % (ParticleMesh–Smilei over all rows is 0.15 % with them and 0.06 % without).

With the focus on the injection plane, in vacuum, ParticleMesh and Smilei are 0.05 mrad, 0.0004 c/ω₀ and 0.003 % apart, and either against PIConGPU at most 0.7 mrad, 0.0025 c/ω₀ and 0.04 %.

## How the launches were matched

The header of `scripts/lwfa/lwfa.jl` derives each term from the codes' sources, and nothing is fitted. Smilei needed four corrections (T-698, T-710, T-712): its carrier is a sine without Gouy phase, its injection node is two cells behind the boundary, its boundary condition launches a plane wave 1.00198 times the prescribed amplitude, and that condition damps oblique components, which widens the launched beam to the waist √(w₀² + 1). ParticleMesh runs the opt-in `WavefrontTime` envelope (T-695 to T-697, T-709). PIConGPU's launch is left as it is; a 1D model of it gives 0.83 of Smilei's oblique term and a focus 0.44 c/ω₀ further out (T-713), against 0.82 and 0.42 to 0.47 fitted from the runs.

## Not shown, and open

- Correctness and convergence: one run per code at one resolution.
- Injection: the bunch is an estimated 260 light macro-particles, and no run kept particles. PIConGPU's spectrum is over its whole domain (#412).
- ParticleMesh in Float32, in its Fast mode, or as the engine stands after `6ead5bd`.
- Any PIConGPU run that tests the model of its launch, and a repeat of any Smilei or PIConGPU run.
- Unexplained: Smilei's wake difference from t = 700 and its heavier bunch, Smilei's E_y at the back of its window, the far wing of the maps, and the centroid distances to PIConGPU.

Stepping-loop wall times of the plasma run, single runs: ParticleMesh 2589 s on the 16 CPU threads and 394 s on CUDA, Smilei 389 s, PIConGPU 142 s after 75 s of compilation. They ran on different hardware with different diagnostics, ParticleMesh in Reference mode before any optimization round, so no performance conclusion is drawn.
