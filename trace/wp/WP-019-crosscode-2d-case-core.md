+++
id = "WP-019"
status = "proposed"
goals = ["A1", "A5", "A6", "A7"]
depends_on = ["WP-003", "WP-014"]
adds_dependencies = []

[checked]
A1 = "69f5bec32fe5d073e16e7811300bcb1ef5092a73"
A5 = "d8ba66a9b95c54fa5e30ede80993135d051424b6"
A6 = "aff99231bff58eb556176499946572ea44ab38dc"
A7 = "25b4f7853f52a9efd57ebd467f84844c7b105f70"
+++
# WP-019: the scripts-only 2D case core: case schema 2, the 2D Smilei generator and reader, landau-oblique-2d.toml and weibel-2d.toml with theory and pre-registered observables before any run

This is item 4 of the 2D3V slice (`docs/plan/architecture-draft.md` §0.8): the Smilei half of the 2D cross-code path, on two cases with exact linear theory, so that the 2D generator and case files are proven before the 2D engine and the module's 2D step arrive. §0.8 orders it first, in parallel with the engine port, because it retires the case-definition thin spot of the slice before any 2D engine code exists. A5 lists only the 1D set and `weibel.toml` is 1D, so the 2D Weibel case, its growth rate, observables and windows must be fixed and pre-registered before the module's 2D validation and WP-017's two-code comparison test against them.

## Scope

`scripts/crosscode/` only for code: the case loader `load_case`, `smilei.jl`, `read.jl`, `theory.jl`, `analysis.jl`/`analyze.jl` as far as the pre-registered 2D observables need, `cases/`, `test/runtests.jl` and `README.md`. Also a new `reports/crosscode-smilei-2d/` for the Smilei 2D runs, `questions/` files, and in `docs/plan/architecture-draft.md` the deletion of §0.8 item 4 and of this package's own sentence in §0.8's ordering text. That sentence stands on its own line, separated from WP-018's sentence by a line neither package edits, and WP-018 edits §0.7, item 1 and its own sentence, so the two packages touch different hunks.

The interface this package owns is the case description of a 2D periodic problem (schema 2) and its translation to Smilei. WP-017 consumes it for PIConGPU and depends on this package: WP-017 keeps the PIConGPU container, generator and openPMD reader for PIConGPU's layout, the quasi-1D runs and their y-average, the oblique seed ensemble in both codes, the two-code comparison and the A7 threshold proposal.

**Schema 2.**

- `[grid]` gains per-axis `length`, `cells` and `boundaries` (arrays, periodic only).
- The perturbation gains the wavevector form `perturbation = { amplitude, mode = [mx, my] }`, meaning n(x, y) = n₀ (1 + amplitude cos(2π (mx x/Lx + my y/Ly))).
- `particles_per_cell` is per axis, matching the engine's product lattice (§0.8 item 1).
- The drift is expressible in all three velocity components, so a beam along z can be written. Schema 1's scalar `drift` (along x) and `transverse_drift = [v_y, v_z]` keep their meaning; the exact schema-2 key form is the package's choice, documented in the README.
- A version check selects the schema; schema-1 files stay valid and mean a 1D grid.
- The `seeds` field is WP-012's: this package does not define it, and if WP-012 merges first, schema 2 keeps it.

**Smilei 2D.** `smilei.jl` writes `2Dcartesian` namelists from schema 2: cell lengths and counts per axis, the step count, patches per axis by the 2D patch rules, periodic field and particle boundaries on both axes, the oblique profile as a Python function of (x, y) (Smilei's `cosine` profile is separable, so it cannot express an oblique mode), and `mean_velocity` from the drift. The unit conversion and profile parameters stay in one tested function (`smilei_parameters`, A1), and `prepare_run` keeps its interface. Every Smilei-specific choice is verified from the Smilei v5.1 source of the existing image and listed in the README's Smilei-specifics section, as WP-003 did: how a non-separable profile is passed, the 2D patch constraints, how `regular` placement handles a per-axis count, `mean_velocity` for a cold drift, the initial Poisson solve in 2D, and which scalar columns hold the per-component field energies. `read_fields` gains Smilei's 2D layout (points × points × outputs, with per-axis positions).

**Cases** (A1 units):

- `landau-oblique-2d.toml`: Landau damping with the same temperature, density and |k| as `landau.toml` (k = 10 ω₀/c, |k|λ_D = 0.5). The wavevector is at an angle to the grid that is not 45° (for example mode [2, 1]), with box lengths chosen so the mode is periodic, so axis mix-ups show. For an isotropic Maxwellian the rate equals the 1D one.
- `weibel-2d.toml`: two cold counter-streaming electron beams along z (±0.3 c, 0.5 n_c each, as `weibel.toml`), opposite density perturbations so the net density stays uniform, k oblique in the plane with |k| = 2 ω_p/c, immobile ions. This exercises Jz and both in-plane Esirkepov components once the 2D module exists. Its growth rate comes from the cold-fluid relation for k ⊥ v₀, derived in the header of `scripts/crosscode/theory.jl` as T-216 does for 1D. A rotation about the axis of k maps the 1D case (k along x, beams along y) onto this one, so the 2D value at |k| must equal the 1D relation at the same |k|, v₀ and ω_p, which the tests check. The relation stays marked unverified against the literature.

Both cases use Δt = 0.95 of the 2D Courant limit 1/√(1/Δx² + 1/Δy²) (§0.8 item 5). Their observables and fit windows are pre-registered in a commit before any run:

- Landau: the electrostatic energy from Smilei's scalars, the mode amplitude projected on k from the field dumps (E·k̂), and the ratio of the Ex and Ey energies, which for a longitudinal wave on the Yee grid is (sin(kxΔx/2)/Δx)² / (sin(kyΔy/2)/Δy)², equal to (kx/ky)² only in the continuum limit; the case file states the value at its resolution.
- Weibel: the in-plane magnetic energy and the Fourier amplitude at k of the in-plane B component perpendicular to k (B·k̂ vanishes by ∇·B = 0).

The expected values are proposals for the owner to accept in A5.

**Runs.** One Smilei 2D run per case through the machine's runner (`command: smilei scripts/runs/<tag>`, `gpu: no`), and only if the existing Smilei image and runner path run `2Dcartesian` without a new image. Otherwise the runs are marked absent and `questions/wp-019-smilei-2d-image.md` says why.

## Deliverables

1. **Namelist hashes before any change**: hashes of the `landau` and `two-stream` Smilei namelists from the current generator, committed for T-270 (the commit is named in the PR).
2. **Schema 2** in `load_case` and `scripts/crosscode/README.md`: per-axis grid, the mode vector, per-axis particles per cell, the three-component drift, a version check, and the rule that schema-1 files keep their meaning. Validation rejects a non-integer mode, a non-periodic boundary and a box in which the mode is not periodic. The README's Smilei-specifics section is extended with the 2D choices verified from the pinned v5.1 source, and its reader schemas with the 2D layout.
3. **`scripts/crosscode/smilei.jl`**: `smilei_parameters` and `namelist` for schema 2 (`2Dcartesian`, per-axis cell lengths and counts, the 2D patch rules per axis, periodic boundaries on both axes, the Python oblique profile, `mean_velocity` from the drift). Schema-1 namelists are unchanged and `prepare_run` is unchanged in interface.
4. **`scripts/crosscode/read.jl`**: `read_fields` for Smilei's 2D field layout, returning points × points × outputs with per-axis positions.
5. **`scripts/crosscode/theory.jl`**: the 2D Weibel cold-fluid growth function for k ⊥ v₀ with its derivation in the file header (A1 units, relativistic factors stated, marked unverified against the literature), a helper giving |k| and the angle from (mode, lengths), and the oblique Landau value through the existing `landau_root` at |k|λ_D.
6. **Case files** `scripts/crosscode/cases/landau-oblique-2d.toml` and `scripts/crosscode/cases/weibel-2d.toml`, with headers in the style of `weibel.toml` (sizes in cells per axis, particles per cell per axis and steps stated), theory values and formulas, and pre-registered observables and windows, in a commit that precedes every run commit (named in the PR).
7. **Analysis**: `analysis.jl`/`analyze.jl` extended only as far as the pre-registered 2D observables need: sums of per-component scalar energies, the Ex/Ey energy ratio, the projection of a 2D field on the mode (mx, my) and the rate fit on it, with the standard plot set, JSON sidecars and manifests (F3).
8. **Tests** T-270 to T-275 in `scripts/crosscode/test/runtests.jl` (no Smilei run inside the suite).
9. **Runs and report**, if the existing image runs 2D: one Smilei 2D run per case, each with its manifest, `observables.json` and plots with JSON sidecars (F3); the runner requests and their exact commands in the PR. `reports/crosscode-smilei-2d/README.md` follows `reports/README.md` and covers:
   - the matching settings (shape order 2, Esirkepov, no filter, FP64, the time-step rule, particles per cell);
   - the Smilei-specific choices verified from source;
   - the measured rates against theory, and the Ex/Ey energy-ratio check;
   - wall times and particle-steps, hardware named by CPU model.

   A fresh numerics-reviewer subagent's findings go in `reports/crosscode-smilei-2d/review.md`, each answered. Otherwise `questions/wp-019-smilei-2d-image.md` and the runs marked absent in the report.
10. **`questions/wp-019-a5-2d-proposals.md`**: the proposed A5 expected values of both 2D cases (theory, plus the single-run Smilei measurement where it exists, marked as one seed) and their derivation, for the owner to accept by editing A5.
11. **Plan text and trace**: `docs/plan/architecture-draft.md` §0.8 item 4 deleted as done and this package's sentence deleted from §0.8's ordering text (current state only), with what they settled carried in the PR description; `build.jl --check` passes.

## Non-goals

- No engine, `lib/` or module code, no ParticleMesh column in 2D (it waits for the module's 2D step, §0.8 item 5), and no edit of `/Project.toml`, `scripts/Project.toml` or any `Manifest.toml`; no new Julia dependency.
- No PIConGPU: no container, generator, openPMD reader for PIConGPU's layout, quasi-1D runs, y-average helper or Smilei–PIConGPU comparison (WP-017).
- No seed ensemble and no `seeds` field (WP-012 owns it); the oblique Landau rate over seeds in both codes stays in WP-017.
- No new container image and no change to the machine's runner.
- No LWFA schema fields (laser antenna, moving window, density ramp, injected bunch), no bounded boundaries, no 3D.
- No A7 agreement thresholds (WP-017 proposes them from the two-code spread) and no accepted A5 value: everything is a proposal; no edit of `docs/goals/`.
- No change of the pre-registered observables or windows after a run; post-hoc variants appear only as labelled sensitivity.
- No edit of `landau.toml`, `two-stream.toml` or `weibel.toml` beyond README cross-references, and no change to their meaning. No 1D Weibel Smilei run: if the shared `mean_velocity` path lifts the generator's current refusal of `transverse_drift`, the newly generatable 1D `weibel` namelist is listed in the PR.
- No `docs/plan` edit beyond deleting §0.8 item 4 and this package's ordering sentence.

## Acceptance

Test ids T-270 to T-279, in `scripts/crosscode/test/runtests.jl`. T-010, T-012, T-013, T-160 and T-161 pass unchanged, and `julia +1.13 --project=scripts scripts/crosscode/test/runtests.jl` is green on the CPU.

- T-270 schema 2 keeps schema 1 [A1, A5, WP-019]: `landau`, `two-stream` and `weibel` load to the same values as before, and the `landau` and `two-stream` Smilei namelists are byte-identical to the hashes committed before the change. Both 2D cases parse to their per-axis grid, mode vector, per-axis particles per cell and three-component drift. Validation rejects a non-integer mode, a non-periodic boundary and a box in which the mode is not periodic.
- T-271 2D unit conversion and namelist [A1, A5, WP-019]: `smilei_parameters` of each 2D case reproduces hand-computed per-axis cell lengths and counts, the time step and step count, densities and the drift. The namelist has `2Dcartesian`, a power-of-2 patch count per axis that divides the cells with more than 5 cells per patch at order 2, periodic boundaries on both axes and the drift as `mean_velocity`. Both case files have Δt equal to 0.95 of 1/√(1/Δx² + 1/Δy²) to roundoff.
- T-272 the oblique profile [A5, WP-019]: the profile read back from a generated namelist, evaluated at the cell centres, equals n₀ (1 + amplitude cos(2π (mx x/Lx + my y/Ly))) to roundoff; the opposite-phase beams of `weibel-2d` give a uniform net density to roundoff.
- T-273 theory of the 2D cases [A5, A7, WP-019]: the oblique Landau |k| equals `landau.toml`'s k to roundoff, and its theory value equals the 1D one to roundoff. The 2D Weibel value is a root of the cold-fluid relation in the `theory.jl` header, and equals WP-014's 1D relation (`lib/ParticleMeshElectromagnetic/test/theory.jl`, included read-only) and `weibel.toml`'s theory value at the same |k|, v₀ and ω_p to roundoff. Both modes are periodic in their boxes with the stated |k| and angle, and each case file's `theory` field equals the function's output.
- T-274 the 2D mode fit [A7, WP-019]: on synthetic 2D fields with a known oblique mode amplitude, a known rate and added noise, the projection on k and the pre-registered fit recover the rate within its reported error. A field along a different mode projects to zero to roundoff, and the Ex/Ey energy ratio of a synthetic longitudinal wave equals its Yee-grid form to roundoff.
- T-275 the 2D Smilei reader [WP-019]: a small file written by the test in Smilei's 2D field layout is read into the documented schema, values and positions.
- The case files with observables, windows and theory are in a commit that precedes every run commit in the PR history; every pre-registered observable names its signal, method and window.
- Either the runs with manifests, observables, plots, sidecars, the report and the answered review are in the PR, or `questions/wp-019-smilei-2d-image.md` explains why they are absent. `questions/wp-019-a5-2d-proposals.md` exists and the PR points to it.
- The report, manifests, sidecars, run requests and PR name the CPU by model and contain no hostname, ssh alias, machine user name or path outside `~` (A6).
- No file under `lib/`, no `Project.toml` and no Manifest changed (checked on the diff). `build.jl --check` passes and CPU CI is green. Tier 1.

## Depends on

- WP-003: the Smilei container and runner path, the case schema 1, the generator, the reader and the analysis this package extends.
- WP-014: `weibel.toml`, the transverse-drift schema extension and the 1D Weibel cold-fluid relation (`lib/ParticleMeshElectromagnetic/test/theory.jl`) that T-273 compares against.

It runs in parallel with WP-008, WP-015 and WP-018: it touches only `scripts/` for code, plus a new report directory, questions and one deleted §0.8 item with its ordering sentence; it adds no dependency and no image, and stays out of WP-012's `seeds` field. WP-017 builds on it. Test ids T-270 to T-279, after WP-018's T-260 to T-269.

## Budgets

- **A7:** theory identities (equal |k|, theory values equal to the 1D ones, profile values, the Yee-grid Ex/Ey energy ratio) to roundoff; fit recovery on synthetic signals within the reported error; no agreement threshold.
- **CPU for Smilei:** each 2D run within the cost of WP-003's 1D Landau reference run (2.56·10⁷ electrons, 379 s on 8 threads, `reports/crosscode-smilei-1d/README.md`), with cells per axis, particles per cell and steps chosen from that bound and stated in the case header; wall time and particle-steps recorded per run.
- **Tests:** the `scripts/crosscode` test suite stays within seconds to a minute (no Smilei run inside it).
- No GPU and no throughput budget.
- **A5:** theory values and single-run measurements are proposals for the owner.
- **Dependencies:** none (HDF5.jl is already in `scripts/`); no image built. Tier 1.
