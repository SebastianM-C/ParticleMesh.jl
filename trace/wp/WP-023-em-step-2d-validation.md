+++
id = "WP-023"
status = "proposed"
goals = ["A1", "A2", "A4", "A5", "A7", "A8"]
depends_on = ["WP-019", "WP-020", "WP-021", "WP-022"]
adds_dependencies = []

[checked]
A1 = "69f5bec32fe5d073e16e7811300bcb1ef5092a73"
A2 = "62bdc24bd734ccab5ce4c723c2a3d72b5d6a8336"
A4 = "0997b929ff6d0c55f79b6f562c66d6e86c8f8e24"
A5 = "d8ba66a9b95c54fa5e30ede80993135d051424b6"
A7 = "25b4f7853f52a9efd57ebd467f84844c7b105f70"
A8 = "3429520bcb753b60ac02a9e6f89007c64e13f927"
+++
# WP-023: the 2D electromagnetic step, Gauss's law with particles, and the 2D cases against theory and Smilei

The second half of `docs/plan/architecture-draft.md` §0.8 item 5. It joins WP-020's current, WP-021's initial solve and WP-022's field solver and push into a periodic 2D3V electromagnetic PIC step, and validates it on the two 2D cases of WP-019, whose Smilei runs are committed in `reports/crosscode-smilei-2d/`. A spike in the A0 sense.

## Scope

`lib/ParticleMeshElectromagnetic` (src and test), `scripts/` for the long runs and the comparison, `reports/em-module-2d.md` with figures, §0.8 item 5 and the module's part of §0.7, `questions/`.

- `step!` at N = 2 with `deposit_flux!` (WP-020) and `initialize!` at N = 2 with WP-021's solve, in both deposition modes.
- `problem_from_case` (or its 2D counterpart) reads schema-2 cases (`landau-oblique-2d`, `weibel-2d`); fix #183 (the seeds list ignored) on the way if it is in the path.
- CI-sized versions of both cases in the tests (fewer cells and particles per cell, an amplitude override as T-214 does); the full-size runs in `scripts/`, reading the same case files Smilei ran.

## Acceptance

Tests T-330 to T-335, CI sizes, Float32, Float64 and `Dual{Float32,3}` where they apply:

- T-330 Gauss's law after every step, as T-212 does: the backward-difference divergence of E minus (ρ − mean ρ), scaled by n·max|ρ| and the step count, both modes [A2, A7, WP-023].
- T-331 the electrostatic edge case of T-213 holds in 2D [A7, WP-023].
- T-332 Reference bitwise across reruns, thread counts and launch shapes for a full step [A4, WP-023].
- T-333 total energy drift over the CI run stays within a measured bound, recorded as proposed [A7, WP-023].
- T-334 oblique Landau at CI size: damping rate and frequency within a tolerance set from the measured seed spread, mode ratio Ex/Ey to the case's k ratio [A5, A7, WP-023].
- T-335 Weibel at CI size: growth rate within a measured tolerance of the cold-fluid value of `scripts/crosscode/theory.jl` [A5, A7, WP-023].
- The full-size runs of both cases against theory and against the committed Smilei sidecars, with the same analysis code (`scripts/crosscode/analysis.jl`), in the report: growth or damping rates, frequencies, energy histories, with seed spreads where the case has seeds. Proposed agreement values go to `questions/wp-023-2d-agreement.md`; A7's threshold needs the Smilei–PIConGPU spread (WP-017), so none is asserted here.
- CPU time per particle-step of the full 2D step (1 thread and the machine's count), recorded as an A7 proposal.
- Existing tests pass. `build.jl --check` passes; CPU CI green.

## Non-goals

No open boundaries, laser, moving window or current filter (the LWFA packages after this one); no GPU run (requested from the GPU runner after merge); no PIConGPU (WP-017); no `Project.toml` change unless the scripts environment needs a package (F2).

## Depends on

WP-020, WP-021, WP-022, WP-019 (the schema-2 cases and the Smilei runs). Test ids T-330 to T-339.
