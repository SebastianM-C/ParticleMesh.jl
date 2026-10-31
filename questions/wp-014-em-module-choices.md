# WP-014: choices made while building the electromagnetic module

Work package: WP-014. Files: `lib/ParticleMeshElectromagnetic/`, `scripts/crosscode/cases/weibel.toml`, `scripts/em1d/`.

None of these blocked the work; each is the assumption I proceeded on, for the owner to confirm or change.

## 1. Thermal loading: a velocity-space Maxwellian, converted to momentum

The engine's `load_quiet!` draws velocities, `drift + v_th·z`. The module's `u` is the momentum `γv`, so after loading each particle's velocity is converted, `u = v/√(1 − v²)`. For the Landau case (v_th = 0.05 c) this is a nonrelativistic Maxwellian in velocity, which is what the linear theory of `scripts/crosscode/theory.jl` assumes. Smilei instead loads a Maxwell–Jüttner distribution in momentum (the caveat in `landau.toml`). The alternative is to treat the drawn values as momenta directly. The two agree to O(v_th²) and are identical for the cold cases.

Assumption: velocity space, matching the theory.

## 2. The edge cases have no transverse thermal spread

`landau.toml` says `velocity = "maxwellian"`, which Smilei loads isotropically in 3D. WP-014 requires the edge cases to run "with no transverse momenta or fields" (T-213), so `problem_from_case` puts the thermal spread along x only, unless it is called with `transverse_thermal = true`. The 1D linear theory sees only the x distribution, so the rate is unaffected. The kinetic energy differs from Smilei's by the two transverse thirds.

Assumption: 1D1V thermal loading for the edge cases; the keyword is there for a later cross-code comparison.

## 3. The Landau test perturbation is 5 %, not the case's 1 %

At the case's amplitude of 0.01, the damped Ex energy falls below the noise floor of 4096 particles per cell before the fit window opens. The peak-envelope fit then finds no points (measured). The case file's 400 000 particles per cell would take hours on the CPU in CI. T-214 therefore uses `amplitude = 0.05`, as the engine's T-141 does, with the case file otherwise unchanged. The driver `scripts/em1d/case.jl` runs the case file as is unless told otherwise.

Assumption: the override is acceptable for the test; the report gives the amplitude next to every number.

## 4. Test dependencies beyond ForwardDiff and Test

WP-014 names ForwardDiff and Test as the test dependencies. The tests also use KernelAbstractions (the CPU backend), TOML (case files and tolerances), Statistics (needed by `scripts/crosscode/analysis.jl`, whose window and peak fits the tests reuse so that the module's rates are measured exactly as the Smilei runs are) and SpecialFunctions (the Landau root in `scripts/crosscode/theory.jl`). All of them are stdlibs or already in the shared Manifest.

Assumption: these are within the package's intent (no new registry package).

## 5. The umbrella package does not re-export the module

`ParticleMesh` (the root package) still depends only on `ParticleMeshCore`. `scripts/` loads `ParticleMeshElectromagnetic` directly through its own `[sources]` entry. Making the umbrella depend on the module would add a second edit to `/Project.toml` beyond the workspace entry, which WP-014 does not ask for. Also, both the electrostatic reference and the module export `initialize!`, `step!` and `run!`, so re-exporting both would clash.

Assumption: no re-export; the user layer (A1, A5) decides later what the umbrella exposes.
