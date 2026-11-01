# Numerics review of WP-017 (cross-code stage 2a: PIConGPU next to Smilei)

| | |
|---|---|
| Role | numerics-reviewer (fresh subagent, `.claude/agents/numerics-reviewer.md`) |
| Model | claude-opus-5-5 |
| Date | 2026-11-01 |
| Scope | `origin/main...HEAD` on `worktree-wp-017-crosscode` at the time of the review: d61a812, 917f460, 2d94590, 00182e6, and cd9e9b9 (the `tolerances.toml` entries, which landed during the review). Covered: `scripts/crosscode/picongpu.jl`, `read.jl` (`PIConGPURead`), `analyze.jl`, `compare.jl`, `test/picongpu.jl` (T-250 to T-256), `scripts/crosscode/README.md` ("PIConGPU specifics"), `reports/crosscode-picongpu/README.md` and its sidecars, `questions/wp-017-a7-thresholds.md`, the #226 experiment. The raw dumps under `scripts/runs/` were used for recomputation. The FP32 Landau placeholders were expected and were not reviewed. |

Written by Claude Code (claude-opus-5-5). The findings are as the reviewer returned them; the answers are in `README.md`, "Review".

## Summary

The deck generator, the unit conversion, the quasi-1D mapping, the reader and the initial-field construction are correct. I checked them against PIConGPU 0.8.0's source and recomputed them from the dumps. The `FieldBackgroundE` argument holds exactly. The Gauss stencil in `analyze_2d` is right for both codes' staggering. The two-stream agreement and the #226 diagnosis reproduce.

There is one blocking finding. PIConGPU's kinetic-energy scalar is not (γ − 1)mc² below γ = 1.005. Because of that, the energy rows of the report, one stated cause in it and one open item are wrong. When the exact kinetic energy is computed from the same files, the two-stream energy-drift difference between the codes disappears.

The charge-density glitch conclusion is right, but the stated argument does not prove it. Stronger evidence is in the data and is given below.

The rest of the checklist does not apply here:

- The diff has no engine code: no kernels, no deposition variants, no element-type paths, no parameter-dependent branches. Genericity, the agreement between deposition variants, smoothness and reference-mode determinism do not apply.
- The pusher's half-step centring matters only through #226 (finding M3 and "Verified").

## Blocking

**B1. PIConGPU's kinetic-energy scalar switches to p²/2m below γ = 1.005, so the energy checks compare different quantities.**

`algorithms/KinEnergy.hpp` returns `mom2 / (2 mass)` when `gamma < GAMMA_THRESH`, with `GAMMA_THRESH = 1.005` (`param/speciesConstants.param:39`). `plugins/EnergyParticles.x.cpp:120` uses it for the `Ekin_Joule` column, and `scripts/crosscode/read.jl:217` reads that column as `Ukin`. The second column, `E_Joule`, is the exact Σ γmc² (`EnergyParticles.x.cpp:125`).

Exact kinetic energy = `E_Joule` − rest energy. The rest energy is E(0)/γ₀ for the cold beams and n·L (or n·L²) for the thermal cases. With it I get (script in the reviewer's `$TMPDIR`, from the committed `.dat` files):

| run | max \|ΔU/U\|, as reported (threshold diagnostic) | max \|ΔU/U\|, exact (γ − 1) | Smilei |
|---|---|---|---|
| two-stream, 128 cells, FP64 | 1.218 × 10⁻³ | **1.016 × 10⁻³** | 1.015 × 10⁻³ |
| two-stream, 256 cells, FP64 | 1.086 × 10⁻³ | **0.545 × 10⁻³** | 0.545 × 10⁻³ |
| two-stream, 128 cells, FP32 | 1.224 × 10⁻³ | 1.004 × 10⁻³ | — |
| Landau seed 20261101 | 2.96 × 10⁻⁷ | 2.44 × 10⁻⁷ | 3.36 × 10⁻⁷ |
| oblique 2D, seed 20261031 | 3.21 × 10⁻⁷ | 2.72 × 10⁻⁷ | 1.33 × 10⁻⁶ |

The final drifts change sign too: the two-stream runs go from +3.7 × 10⁻⁴ to −7.2 × 10⁻⁵ at 128 cells (Smilei −5.7 × 10⁻⁵), and from +4.1 × 10⁻⁴ to −3.5 × 10⁻⁵ at 256 cells (Smilei −3.3 × 10⁻⁵). The cause: the beams start at γ₀ = 1.00504, just above the threshold. After saturation, slowed electrons are counted with p²/2m, which overstates their energy by about p⁴/8.

What this changes:

- `reports/crosscode-picongpu/README.md:76`: the two-stream energy row is an artefact.
- `README.md:80` and `:187`: the open item "does not halve at 256 cells" is the diagnostic, not the dynamics. The codes agree to three digits.
- `README.md:80` and `scripts/crosscode/README.md:116`: "computed as (γ − 1)mc²" is wrong.
  - Computed by quadrature for T = 0.0025: the exact relativistic kinetic energy of PIConGPU's load is −0.310 % from (3/2)nT, and the threshold diagnostic gives −0.241 %.
  - Measured, the diagnostic is −0.25 % (mean of 8 Landau seeds) and −0.27 % (mean of 4 oblique seeds). The diagnostic/exact ratio is +6.9 × 10⁻⁴ in both, as predicted.
  - The "−0.30 %" in `README.md:75` is seed 20261101 alone.

**Fix:** in `read_picongpu_scalars`, set `Ukin` = `E_Joule` − Σ_s n_s m_s V, taking the rest energy from the case. Re-analyze, then redo rows 75–76, the text at line 80, the open items at lines 180 and 187, and the Temperature bullet of `scripts/crosscode/README.md`. Add the threshold to the "PIConGPU specifics" list and to `questions/wp-017-lwfa-case.md`: a warm or relativistic LWFA energy balance would hit it. None of the rates is affected, because they use E only.

## Major

**M1. The glitch argument has a gap; the conclusion holds on other evidence.** (`reports/crosscode-picongpu/README.md:78`)

"Esirkepov keeps any Gauss defect, so a missed current or a lost particle would persist" does rule out lost particles and missed or wrong current deposits. It does not rule out a particle skipped in the push. Such a particle does not move, deposits no current and leaves no Gauss defect, so the inference "the fields and the particles are therefore consistent" does not follow from it. The report keeps the push open (line 180), but the paragraph reads as a proof.

Recomputing on seed 20261101 confirms every count: 21 of 671 dumps are short by 1.09 × 10⁻⁴ to 8.27 × 10⁻⁴; the other dumps are at 7.8 × 10⁻¹⁶; the Gauss residual is 9.3 × 10⁻¹⁵ at t = 0, ≤ 1.3 × 10⁻¹⁴ at good dumps and 0.0219 at bad ones. The data also give much stronger evidence that only the derived density is wrong:

- **Two derivations, one state.** The openPMD `FieldTmp` and the `chargeConservation` plugin's own `FieldTmp` derive the density from the same particle state at the same iterations (both use `fields_every`). Their spikes fall on different steps: 21 against 20 events, overlapping only at step 1208. So the same state is turned into different densities: the derivation is nondeterministic.
- **The energy plugin sees every particle.** `EnergyParticles` runs at the same steps over the same area (CORE+BORDER) and is smooth to 10⁻⁸ relative. It shows no dip at any glitch step, while a loss of 10⁻⁴ of the electrons would show at 10⁻⁴.
- **The deficit sits in one supercell.** At steps 20, 36, 1208 and 2680, the residual is confined to one interior y-row (rows 3, 3, 2 and 4, never the periodic seam) and to 19 contiguous x-columns (64, 1 to 18): one 16 × 1 supercell plus the TSC footprint. About 0.35 % of that supercell's electrons are missing, about 350 macro-particles or about 22 frames of 16. That points at the frame traversal of the deposit, not at the guard exchange.

**Fix:** replace the Esirkepov sentence with these three observations. State that a skipped push would leave no Gauss trace and stays untested. Pass the localization to the `upstream` reduction (`scripts/runs/upstream-fieldtmp-race/`).

**M2. The quasi-1D Gauss and total-charge numbers have no committed source.** (`README.md:72–73`, `:78`)

`analyze.jl` has a Gauss check only in `analyze_2d`. The schema-1 path writes neither a Gauss residual nor a per-dump total charge, so the Landau and two-stream entries of the checks table, and the 21-of-671 glitch counts, come from an uncommitted script. I reproduced them (two-stream: 8.76 × 10⁻¹⁴ max over 165 dumps). Still, F3 wants every reported number traceable to a sidecar.

**Fix:** add the quasi-1D check to `analyze.jl`: the backward-difference stencil on the 2D dumps before the y-average, and the total charge per dump. Write both into `observables.json`.

**M3. The #226 numbers are right; label them as relative.** (`README.md:95`, `:97`)

"+8.4 × 10⁻⁴ from both codes" and "−1.0 × 10⁻⁴ from them" are relative differences. The absolute differences are 2.9 × 10⁻⁴ and 3.5 × 10⁻⁵, and the report's other differences are absolute. My rerun of `exp.jl` gives γ = 0.345577 for the as-is case (the report says 0.345578; #226 says 0.345577). Severity: wording, but the threshold proposal reads these numbers.

## Minor

- **m1.** `README.md:76`, Landau energy row. The per-seed range is 2.6 to 4.3 × 10⁻⁷, not 3.0 to 3.8 × 10⁻⁷. Smilei's is 2.7 to 3.6 × 10⁻⁷, not "3.5 × 10⁻⁷". It gets regenerated under B1 anyway.
- **m2.** `analyze.jl:111`. PIConGPU prints the energy components with 6 digits (the report notes this for `energy_ratio`), and the two-stream fits use `Uelm_Ex`.
  - Refitting on the 16-digit total `Uelm` (Ey ≤ 10⁻³³, B = 0 in quasi-1D) moves γ by −7.6 × 10⁻⁹ (128 cells) and +3.2 × 10⁻⁸ (256 cells).
  - The reported 256-cell difference of +4.4 × 10⁻⁸ is therefore at the quantization level. It becomes +7.5 × 10⁻⁸, so `T-215.wp017` `c256` changes from 1.3 × 10⁻⁷ to about 2.2 × 10⁻⁷ relative.
  - The 128-cell saturation energies agree to 2.9 × 10⁻⁶ on full precision, not 10⁻⁶ (`README.md:93`).
  - Use `Uelm` for quasi-1D PIConGPU runs.
- **m3.** `README.md:185`. Two-stream is cold, so it cannot show that the thermal-momentum difference "does not matter". It shows this only for the in-cell lattice.
- **m4.** `questions/wp-017-a7-thresholds.md:32`. A constant 2 × 10⁻⁴ factor in U does not change γ over a fixed window: the no-kick run has that factor and still equals both codes' γ to six digits. The relaxation argument does not follow as written.
- **m5.** FP32 Gauss is not reported. Two-stream FP32 has 7.1 × 10⁻⁷ at t = 0 (7 × 10⁻⁴ of the perturbation) and 5.8 × 10⁻⁵ at most. That is consistent with an FP32 roundoff random walk in E over 5253 steps (estimate ~4 × 10⁻⁵). For A7's "Gauss to roundoff" per precision, add an FP32 column to the checks table.
- **m6.** `analyze.jl:526–528`. The comment still describes only Smilei's dual-point convention; the new comment below it is the correct one.
- **m7.** Provenance (F3):
  - `picongpu-two-stream-fp64-736a428f/manifest.toml` has `[analysis]` at 917f460 with `dirty = true`. Re-analyze from a clean tree.
  - The `runs` entries in `compare/*/comparison.json` point to the uncommitted `scripts/runs/` instead of the committed report directories.
- **m8.** `README.md:45`. In quasi-1D, a column holds 50 000 macro-particles but only 6250 distinct x positions, each repeated 8 times. The noise is unaffected because the velocities are independent. The quiet-lattice form factor, however, is the 6250-per-cell one (T-255 uses it correctly). Worth one clause.

## Verified

- **Tests.** `julia +1.13 --project=scripts scripts/crosscode/test/runtests.jl`: all 27 testsets pass, including T-250 to T-256 and T-253. `scripts/trace/build.jl --check` passes. `tolerances.toml` (cd9e9b9) parses, and its lookups by id are unaffected.
- **Units (A1).** The constants in `picongpu.jl:31–35` equal `param/physicalConstants.param`, and n_c, the E unit and the energy unit follow from them. The temperature conversion and the per-area or per-depth energy unit are independently confirmed by the measured initial kinetic energy: −0.25 % against the −0.24 % predicted for PIConGPU's diagnostic (B1).
- **Quasi-1D mapping.**
  - The domain rules are as stated: `MappingDescription.hpp:78–79` (2·guard + 1), `DomainAdjuster.hpp:205` (three supercells), and the Esirkepov margin of 3 cells for TSC (`Esirkepov2D.hpp:47–57`). The Yee CFL check is a compile-time assert.
  - The ky = 0 part of the Yee update is the 1D update. The y-sum of 2D Esirkepov's Jx is the 1D Esirkepov current.
  - Δy = 8Δx gives a limit of 0.992Δx > 0.95Δx.
- **`deposit_form_factor` and `background_field`.**
  - Derived by hand: ρ_m = Σ_d Σ_j S(d + u_j) cos(kΔ(d + ½))/n; the sine part cancels by the lattice's symmetry; d ∈ {−2, …, 1}.
  - The field E_d = (ρ̂/K²)κ_d sin(·) at E_x (i + ½, j) and E_y (i, j + ½) has discrete divergence ρ̂ cos at the nodes, and zero discrete curl, because a_x κ_y = a_y κ_x.
  - The positions match `fields/YeeCell.hpp` (E_x (0.5, 0), E_y (0, 0.5), FieldTmp (0, 0)). The cell index passed to the functor is global and guard-corrected (`cellwiseOperation.hpp`).
  - The `chargeConservation` plugin's backward stencil gives 1.2 × 10⁻¹² of a cell's perturbation charge at t = 0 (two-stream; recomputed from the file and the cell volume), and it stays bounded over every run.
- **Equivalence to an initial E.** From `Simulation.hpp:529–577` and pmacc `SimulationHelper.cpp:241–262`, the order per step is:
  1. `movingWindowCheck` adds the background;
  2. plugins and dumps see the total field;
  3. the push uses it;
  4. `subtract` (with `duplicateFields`, a restore from a copy) runs before the field solver and J.

  So E_total = E_grid + E_bg, with E_grid(0) = 0. ∂_t E_total = ∂_t E_grid = ∇×B − J. B is driven by ∇×E_grid = ∇×E_total, since E_bg is discretely curl-free. That is Maxwell's equations from E⁰ = E_bg, exactly. In FP32 the representation leaves an absolute error floor of about eps₃₂·|E_bg|, far below the noise floor here.
- **Reader.** On real files: `meshesPath = "fields/"`, `axisLabels ["y", "x"]`, `gridSpacing` y first (8.42 against 1.05 in quasi-1D), `position` x first (E_x [0.5, 0]), datasets shown as (Nx, Ny), `time` = step × dt.
- **Gauss stencil in `analyze_2d`.** `sx` = 0 for Smilei (dual point at −Δ/2) and 1 for PIConGPU. My independent backward-difference computation gives 1.089 × 10⁻¹² (PIConGPU), equal to the sidecar. Smilei's 6.5076 × 10⁻¹⁰ is bit-identical to WP-019's value.
- **Report numbers against the sidecars.** Every number in these places agrees with the committed JSON and `container.txt`, except those named in m1, m2 and B1:
  - the two-stream table;
  - the Landau ensemble table: ensemble values, jackknife errors, maxima, ranges, ω_r, floors, per-seed values and spreads, the 0.29, 0.21 and 0.14 ratios, F = 2.38;
  - the oblique table: all rows, including 1.46, 1.25, 2.49, 0.33, 2.3, and the 1.6 and 1.9 errors from the Yee value;
  - the |Ê(0)| values and the 2D Gauss numbers;
  - the compile and run times and ns per particle-step.
- **#226.**
  - I reran `/tmp/claude-1000/ts/exp.jl base nokick`. Baseline: r(1) = 1.01255, r(18) = 0.98963, identical to the stored WP-012 series; γ = 0.345577. No kick: r = 1.0002 at every time, γ = 0.345612, equal to PIConGPU and Smilei to six digits.
  - `lin.jl` gives 0.98928 and γ_lin = 0.3508758. Its fluid equations and the sign of the initial momentum kick are correct.
  - PIConGPU takes the loaded momenta as p^{−½}: the first push of step 0 uses E⁰ including the background.
- **Pre-registration.** d61a812 (15:36:37Z) precedes the first PIConGPU compile (15:37:14Z). T-256 checks WP-019's entries by hash.

## Not verified

- Smilei's side of the initial-momentum convention, and the 131/128-column explanation of Smilei's 2D `Uelm`: not checked in Smilei's source.
- Whether the frame skip of M1 also occurs in the push. This needs per-step particle counts or a pushed-particle checksum.
- The cell-centred-weight explanation of ParticleMesh's remaining 2 × 10⁻⁴ in U (the report marks it untested).
- The roughly 0.6 s per openPMD dump.
- The FP32 Landau run, which was still running.
