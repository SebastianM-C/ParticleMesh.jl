# Smilei on the 2D case core (oblique Landau damping, 2D Weibel)

| | |
|---|---|
| Work package | WP-019 (goals A1, A5, A6, A7) |
| Runs | `smilei-landau-oblique-2d-c8709f77`, `smilei-weibel-2d-dca1230a` |
| Commit of the inputs | `310c2ee63b0c56d0c661091d878878e7767fa8b2` (clean tree, recorded in each `manifest.toml`); the case files and their observables were fixed in that commit, before the runs. It and the analysis commit below are on the branch `wp-019-runs-provenance`, the history before the PR was rebased onto `main`; the PR's commits carry the same content under new ids |
| Case hashes (SHA-256) | `landau-oblique-2d.toml` `c8709f77…` and `weibel-2d.toml` `dca1230a…`; full hashes in each run's `manifest.toml` under `inputs` |
| Code | Smilei v5.1, commit `438d43d8ab727d02dd8f4402c50393343e871b99`, CPU build; image `localhost/particlemesh-smilei:e4f0148b8952fc87`, id `7030c120c62ce14dbae971cce76a3973f0a031ef63632a6c4bf4fa20e3937b4c`, from `scripts/crosscode/containers/smilei.Containerfile` (unchanged since WP-003) |
| Hardware | Intel Core i9-14900K; OpenMP with one thread per patch (4 × 4 patches, 16 threads), one MPI rank |
| Analysis | `scripts/crosscode/analyze.jl` (`analyze_2d`) at `b80311dd4e4a877b40feed45fe8f7f41ea816db6`, clean tree (each manifest's `analysis`). It evaluates the pre-registered observables as at `310c2ee` and adds loading and staggering checks and one labelled sensitivity |
| Review | numerics-reviewer (`.claude/agents/numerics-reviewer.md`), fresh subagent; findings and answers in `review.md` |

This report records numbers and sets no tolerance (A7). The expected values it measures are proposals for the owner to accept in A5 (`questions/wp-019-a5-2d-proposals.md`). Each run directory holds the run's manifest, namelist, `container.txt` (timing record), `observables.json`, and the standard plots with their JSON sidecars.

## Setup

Both cases are 2D periodic, `2Dcartesian`, in the units of A1 with the total electron density 1 n_c, so ω_p = ω₀. Case files: `scripts/crosscode/cases/`, schema 2 (`scripts/crosscode/README.md`); namelists from `scripts/crosscode/smilei.jl`.

Matching settings, the A5 list:

| Setting | Both cases |
|---|---|
| Shape order | 2 (Smilei `interpolation_order = 2`) |
| Current deposition | Esirkepov (Smilei's `Projector2D2Order`, by its source) |
| Filtering | none |
| Precision | FP64 |
| Time step | 0.95 of the 2D Courant limit 1/√(1/Δx² + 1/Δy²) |
| Particle loading | regular lattice per cell (`regular_number` per axis), density profile sampled at cell centres |
| Ions | immobile (frozen for twice the run), 1 per cell |

| | Oblique Landau damping | 2D Weibel |
|---|---|---|
| Physics | thermal electrons, v_th = 0.05 c, \|k\|λ_D = 0.5, density perturbation 1 % | two cold beams at ±0.3 c along z, 0.5 n_c each, opposite density perturbations 0.1 %, \|k\| = 2 ω_p/c |
| Mode, angle | [2, 1], atan(1/2) = 26.57° | [2, 1], 26.57° |
| Box, cells | 1.40496 × 1.40496 c/ω₀, 128 × 128 (Δ = 0.2195 λ_D) | 7.02481 × 7.02481 c/ω₀, 256 × 256 (Δ = 0.02744 c/ω₀) |
| Δt, steps run | 0.0073733; 3390 to t = 24.9956 (the case asks for 3391, see below) | 0.0184333; 2170 to t = 40.000 |
| Particles per cell | 32 × 32 electrons (momenta random, seed 20261031) | 8 × 8 per beam, cold |
| Electrons in total | 1.68 × 10⁷ | 8.39 × 10⁶ |
| Field dumps | Ex, Ey, Rho every 16 steps | Bx, By, Rho every 16 steps |

The Smilei-specific choices, verified in the v5.1 source, are listed in `scripts/crosscode/README.md` ("Smilei specifics", the 2D part): the profile passed as a numpy function of (x, y), the patch rule per axis, `regular_number`, `mean_velocity` for a cold drift, the initial Poisson solve and the constant Smilei adds after it, the per-component energy scalars, the step count, and the 2D dump layout with its dual-axis positions.

Two facts about Smilei found in the review apply to these runs:

- **The step count.** Smilei runs (int)(simulation_time/timestep) steps, and the generator passed `number_of_timesteps = 3391`, which Smilei turns into simulation_time = 3391·Δt; that quotient truncated to 3390. The last field dump is at iteration 3376 either way, so no observable changes. The generator now passes simulation_time = (n + ½)Δt in 2D (T-271), and the step counts and particle-steps below are Smilei's (`n_time` in `smilei.log`).
- **The thermal loading.** Smilei's `maxwell-juettner` initialization samples, below T = 0.1 m_e c², the kinetic energy from the classical Maxwell–Boltzmann energy distribution and sets |p| = √((1 + E)² − 1). The measured initial kinetic energy agrees: 7.4034 × 10⁻³ against (3/2) n T L² = 7.4022 × 10⁻³ (+0.016 %), where a Maxwell–Jüttner load would give about 7.4254 × 10⁻³ (+0.31 %). The `caveat` in the pre-registered case file says Maxwell–Jüttner; that wording is wrong, and the case file is left as registered. The O(T) relativistic correction to the Landau root is not computed.

## Linear theory

Rates are those of the field amplitude in units of ω_p; energies go as e^{±2γt}. Values from `scripts/crosscode/theory.jl`, checked by T-273.

**Oblique Landau damping.** For an isotropic Maxwellian the dispersion relation depends on |k| only, so the root at |k|λ_D = 0.5 is that of `landau.toml`: ω = 1.4156619 − 0.1533595 i, **γ = 0.1533595**, ω_r = 1.4156619.

**2D Weibel.** The cold-fluid relation for k ⊥ v₀ is derived in the header of `theory.jl`: X² − A X − B = 0 with X = ω², A = k² + ω_p²/γ₀³, B = ω_p² v₀² k²/γ₀, γ = √(−X₋). At |k| = 2, v₀ = 0.3 (γ₀ = 1.0482848): **γ = 0.2637255**. The derivation shows that a rotation about k maps the 1D Weibel case of WP-014 onto this one. T-273's numerical agreement of the two functions holds by construction, since both evaluate the same formula, so the evidence for the 2D relation is the derivation, which the review re-derived term by term. The relation is unverified against the literature. The case file's `expectation` gives the large-|k| limit as ω_p v₀/√γ₀ = 0.296; the correct value is 0.2930. The case file is left as registered.

**Ex/Ey energy ratio of the oblique Landau wave.** E = −∇φ by the Yee differences gives Ex ∝ (2/Δx) sin(k_xΔx/2) and Ey ∝ (2/Δy) sin(k_yΔy/2), so the ratio is (sin(k_xΔx/2)/Δx)² / (sin(k_yΔy/2)/Δy)² = **3.9975909** at this resolution, against (k_x/k_y)² = 4 in the continuum.

## Results

Uncertainties are one standard error of the fitted slope, from the residuals; they do not include systematic effects. All four rate and ratio observables were pre-registered in `310c2ee`.

| Case, observable | Method | Measured | Theory | Relative difference |
|---|---|---|---|---|
| Landau, `damping_rate_mode`: E·k̂ of mode [2, 1] from the dumps | maxima of \|E·k̂\|², t ≥ 3, above 10 × floor (5 maxima, t = 4.75 to 13.61) | **0.1609 ± 0.0078** | 0.1533595 | +4.9 % (1.0 standard error) |
| | frequency from the spacing of those maxima (2.18 to 2.25 apart) | 1.4185 | 1.4156619 | +0.2 % |
| Landau, `damping_rate_energy`: Uelm_Ex + Uelm_Ey | same rule | none: fewer than three maxima above 10 × the floor (3.97 × 10⁻⁸; initial energy 5.04 × 10⁻⁷) | 0.1533595 | — |
| Landau, `energy_ratio`: Uelm_Ex / Uelm_Ey | at t = 0 | 3.98629 | 3.99759 (Yee), 4 (continuum) | −0.28 % (Yee) |
| Landau, `energy_ratio_mode`: \|Êx\|² / \|Êy\|² of mode [2, 1] | median at the 5 fitted maxima | 3.9450 | 3.99759 | −1.3 % |
| Weibel, `growth_rate_mode`: B·ê, ê = ẑ × k̂, of mode [2, 1] | log-linear fit where \|B·ê\|² lies between 10⁻⁴ and 10⁻² of its saturation value (30 outputs, t = 12.39 to 20.94; saturation t = 32.15) | **0.2609 ± 0.0019** | 0.2637255 | −1.1 % (1.5 standard errors) |
| Weibel, `growth_rate_energy`: Uelm_Bx_m + Uelm_By_m | same rule (121 points, t = 12.31 to 21.16; saturation t = 32.00) | 0.2611 ± 0.0009 | 0.2637255 | −1.0 % (2.9 standard errors) |

The Landau field-energy fit was expected to fail: the case file states before the run that the sum over all 128² modes reaches its noise floor within a few maxima. The projected mode gives 5 usable maxima, as many as the 1D reference run of WP-003, with a standard error of 4.9 % against 9 % there. The single-mode noise level was not compared between the two runs, so this does not test the noise argument the case header uses.

**The Ex/Ey ratio and Smilei's constant field.** At t = 0 the box mean of Ey is 1.68 × 10⁻⁵, 3.8 % of the Ey wave amplitude, and that of Ex is 1.6 × 10⁻⁶ (`field-snapshots.json`, `box_mean_at_first_output`). Smilei adds this uniform field after its Poisson solve (`scripts/crosscode/README.md`; a minimal run reproduces it, issue #157). It adds energy to Uelm_Ey and lowers the ratio.

A labelled post-hoc sensitivity, not a pre-registered observable: with the box mean of each field removed, the t = 0 ratio is 3.9975909124103444 (`manifest.toml`, `checks`), the Yee value to roundoff. So Smilei's initial field is the Yee-grid solution, and the pre-registered −0.28 % comes from the uniform field alone. The Yee value differs from the continuum one by 0.06 %: the pre-registered ratio does not resolve that difference because of the offset, and the sensitivity resolves it exactly.

After t ≈ 10 the scalar ratio falls to about 1 as isotropic noise takes over the field energy, while the mode-coefficient ratio stays near the Yee value (`energy-ratio.png`). The box mean of Ey changes sign every ≈ π/ω_p, a k = 0 plasma oscillation that the case's initial condition does not contain; it does not project on k.

Loading and staggering checks (`manifest.toml`, `checks`):

| | Landau | Weibel |
|---|---|---|
| Box mean of Rho at t = 0 | −3.0 × 10⁻¹⁶ | 2.6 × 10⁻¹⁷ |
| Largest deviation of Rho from its mean at t = 0 | 9.980 × 10⁻³ | 5.1 × 10⁻¹⁴ |
| Amplitude of Rho in mode [2, 1] at t = 0 | 9.98019 × 10⁻³; α n₀ = 10⁻² times sinc(0.975 kΔ/2) sinc³(kΔ/2) per axis (the cell-sampled profile compressed by the 0.975 placement, and the quadratic shape) = 9.98019 × 10⁻³ | 1.4 × 10⁻¹⁶ (opposite-phase beams: uniform net charge) |
| Gauss's law at t = 0, max \|∇·E − ρ\| / max \|ρ\| | 6.5 × 10⁻¹⁰ | — |
| max \|∇·B\| / max \|B\| over all outputs | — | 2.2 × 10⁻¹³ |
| Initial kinetic energy | 7.4034 × 10⁻³ ((3/2) n T L² = 7.4022 × 10⁻³) | 2.38276 ((γ₀ − 1) n₀ L² = 2.38276) |
| Largest relative change of the total energy | 1.3 × 10⁻⁶ | 2.9 × 10⁻³ (nonlinear stage) |

The two staggering checks take the reader's dual-axis positions at (i − ½)Δ, which is where Smilei puts them (`scripts/crosscode/README.md`); with the openPMD reading (i + ½)Δ, Gauss's law fails by 8.8 × 10⁻² of max |ρ|.

## Cost

| | Landau | Weibel |
|---|---|---|
| Wall time (container `start` to `end`) | 716 s on 16 threads | 228 s on 16 threads |
| Thread-seconds | 11 460 | 3 640 |
| Particles (mobile) | 1.68 × 10⁷ | 8.39 × 10⁶ |
| Particle-steps (mobile) | 5.69 × 10¹⁰ | 1.82 × 10¹⁰ |
| Wall time per mobile particle-step | 12.6 ns at 16 threads | 12.5 ns at 16 threads |

WP-019 budgets each run within the cost of WP-003's 1D Landau reference run: 2.56 × 10⁷ electrons, 6.86 × 10¹⁰ particle-steps, 379 s on 8 threads, i.e. 3 030 thread-seconds. The cases were sized in particle-steps before the runs, and both stay below that bound. **In CPU time both runs exceed it**: Landau by 3.8 times (11 460 thread-seconds) and Weibel by 1.2 times (3 640). Per particle-step the 2D runs used 201 thread-ns against 44 in 1D. That ratio mixes the dimension with the thread count and the core type: the i9-14900K has 8 performance and 16 efficiency cores, the 1D run used 8 threads and these 16, and the 1D case was not timed at 16 threads. Smilei's own log reports a push time of about 195 ns per particle per core.

## Figures

Per run directory, each with a JSON sidecar of the numbers it shows:

- `mode-amplitude.png`: the squared mode coefficient of the projected field, the points used, the fit and the theory slope.

  ![Landau mode amplitude](smilei-landau-oblique-2d-c8709f77/mode-amplitude.png)
  ![Weibel mode amplitude](smilei-weibel-2d-dca1230a/mode-amplitude.png)

- `field-energy.png`: the summed field-energy scalars on a log scale.
- `energy-ratio.png` (Landau): Uelm_Ex/Uelm_Ey and the mode-coefficient ratio, with the Yee and continuum values.

  ![Landau energy ratio](smilei-landau-oblique-2d-c8709f77/energy-ratio.png)

- `field-snapshots.png`: the first projected component (Ex, Bx) at four times, colour range centred on zero. The oblique [2, 1] pattern shows the axes are not swapped.
- `energy-conservation.png`: the relative change of the total energy.

## Verification status

- T-270 to T-275, with T-010, T-012, T-013, T-160 and T-161, pass on the CPU: `julia +1.13 --project=scripts scripts/crosscode/test/runtests.jl`. They cover schema 2 keeping schema 1, the 2D unit conversion and namelist, the oblique profile, the 2D theory, the 2D mode fit on synthetic fields and the 2D reader.
- The reader and the analysis ran on both runs' output, and the figures were inspected before the numbers were recorded. The staggering checks above confirm the reader's dual-axis positions on the run data.
- The review reproduced every fitted number from the raw output. Before the dual-axis fix the projections used positions one cell off on dual axes. That is a constant phase per component: the rates changed only in the sixth digit (Landau 0.160935 to 0.160930), but the leak of E into the direction perpendicular to k fell from 3.9 × 10⁻⁴ to 1.5 × 10⁻⁸ in energy.
- Not verified:
  - the 2D Weibel relation against the literature;
  - the relativistic correction to the Landau root;
  - why Smilei's corner correction does not vanish here. A minimal run reproduces it and attributes the Ey offset to one of the two corner values alone (issue #157), but that is inferred from the numbers, not traced in the source.

## Known limitations

- One run per case, one seed: the Landau standard error is statistical within one realization of the thermal noise, and the spread over seeds is not measured (WP-017 runs the seed ensemble in both codes).
- The peak-envelope rule can count one noise-split maximum twice, which leaves the rate but shortens one spacing and biases the frequency. It also stops at the first maximum larger than its predecessor. Both are described in issue #145. The Landau fit here used 5 maxima, 2.18 to 2.25 apart, so no maximum was split.
- The uniform field Smilei adds at t = 0 is in every Smilei field-energy scalar of these cases.
- Smilei's per-component energy scalars exceed ½ ΔxΔy Σ F² over the distinct dump points by 131/128 at t = 0 of the Landau run (measured, cause not traced).
- Because of these two points, a comparison of absolute field energies with another code has to remove the box mean and account for that factor; ratios and mode-projected observables need neither.
- The Weibel energy fit includes every in-plane mode. The review found mode [2, 1] holding 99.99 % of the in-plane magnetic energy through the fit window and 97.8 % at saturation, so the faster modes at larger |k| did not overtake the seeded one.
