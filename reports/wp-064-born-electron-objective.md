# WP-064: can the energy of ionization-born electrons be an optimization objective?

**Summary.**
- A 2D LWFA run at λ₀/16 whose pulse ionizes a nitrogen dopant's K shell inside the wake traps born electrons at every a₀ from 1.5 to 2.4. At a₀ = 1.65, 46 % of the born weight is in the first bucket above u_x = 5 at t = 300, at a mean energy of 15 mₑc².
- The objective J, the kinetic energy of the born species times a smooth position weight, rises by a factor 1900 over that range without a step between the 13 runs; at a₀ = 1.65 d ln J / d ln a₀ is 22.
- ForwardDiff's derivative is the local slope: at a₀ = 1.65 a central difference at h = 1.3·10⁻⁵ agrees to 1.6·10⁻⁷, and at a₀ = 2.3 (physics breaker, not in the run record) central differences converge onto it, from 2 % at h = 0.018 to 10⁻⁹ at h = 7·10⁻⁸. Nine runs spaced 9·10⁻⁶ apart there show no step.
- J is smooth but not gentle: secants over steps from 10⁻⁵ to 0.02 differ from the local slope by 0.1 % to 2 %, and the slope across the 33 fine-scan runs (a₀ ± 0.0064) by 1.7 %. With the `Fast()` deposition J differs by 2·10⁻¹⁵ (breaker), so this structure is the run's, not amplified rounding.
- The guard is needed, and over the whole window. Where J reads (window x ≥ 12, the first bucket) plasma electrons above 5 mₑc² appear from a₀ = 1.65, a few macro-particles from the density ramp's foot up to a₀ = 2.175 (18 % of the trapped born weight at 2.1), and from a₀ = 2.25 they outweigh the born ones (11 times at 2.4, with 11 times J's energy), where J's slope halves. In the bucket behind it the plasma injects at every a₀ of the scan: 8 times the trapped born weight at a₀ = 1.5, 79 times at 1.65. J sees none of this.
- A plain run takes 32 s on 4 threads, a run with one dual 57 s.
- J's value depends on choices the scan fixes (breaker): the read-out at t = 250 gives 23 % less with the same d ln J / d ln a₀ (22.1 against 21.9), since the electrons are still gaining energy; a birth every second step 1 % less; other edges of the position weight change J by up to 0.6 % and its derivative by up to 1.5 %.
- Two runs at λ₀/32 (a₀ = 1.6 and 2.0, exploratory, not in the run record) gave J 22 % and 11 % higher and the trapped born weight 20 % and 8 % higher.
- The rates are ADK without barrier suppression (WP-063's level is not used yet) and the ionization takes no energy from the field (WP-059), so the injected charge is not a physical prediction. No optimizer has run.

| | |
|---|---|
| Work package | WP-064 (`trace/wp/WP-064-born-electron-objective.md`); goals A0, A1, A3, A4, A7 |
| Commits | the run's code `bbc3bb9` (`dirty = false`); a later commit drops the table of ionization energies from the driver's list of hashed inputs, since `main` now reads them from Mendeleev.jl (#462), with which the base run's J has the same bits |
| Runs | `optim-born2d-20261117T024038`, from `julia +1.13 --project=scripts -t 4 scripts/optim/born2d.jl 16` (2517 s); it repeats the bits of J and of every born-electron number of `optim-born2d-20261117T012425` (commit `55c4ffd`), whose guard read only ahead of x = 12 |
| Input hashes | SHA-256 of every input file in `wp-064-born-electron-objective/optim-born2d-20261117T024038/manifest.toml` |
| Environment | Julia 1.13.4, Float64, Reference mode, the KernelAbstractions CPU backend; Intel Xeon Processor @ 2.10GHz (the model string the CPU reports), 4 vCPUs of a cloud VM, 4 threads |
| Figures | `born2d.png` in the run's directory, with its JSON sidecar (every number of the run) |
| Status | measured, not independently reviewed; no number here is to leave the repository before that review |

## Setup

The case (`scripts/optim/born2d_case.jl`): a moving window of 14 λ₀ × 128 cells of Δy = π/4 with PMLs, a pulse of waist 10 and duration 6 focused at x = 80 into a plasma of 0.072 n_c, and nitrogen at 1 % of that density on the 36 rows around the axis in the section x = 76 to 100, with the levels N⁵⁺ → N⁶⁺ → N⁷⁺ at 0.8 µm (WP-059's `FieldIonization`, a birth at every step). The objective is `born_energy`: J = Δx·Δy·Σ w(γ − 1)S(x) over the alive electrons of the born species, with S a C² step from 0 at window x = 12 to 1 at x = 24; the back band ends at x = 3.5, the first bucket lies between x = 27 and 40 and the pulse is at x = 47 at the read-out (t = 300, 900 steps). The guard is `plasma_peak_energy`, `plasma_fast_weight` and `plasma_fast_energy`, over the plasma's electrons of non-zero weight above 5 mₑc², read twice: ahead of x = 12, where J reads, and over the whole window ahead of the back band. The scan goes through the Optimization.jl problem of `problem.jl` in z = a₀/2.

## Results

| a₀ | J | dJ/da₀ (ForwardDiff) | born electrons trapped | born mean energy | plasma above 5.0 mₑc², where J reads: weight | its energy | its highest γ − 1 | plasma above 5.0 mₑc², whole window: weight | its energy |
|---|---|---|---|---|---|---|---|---|---|
| 1.5 | 0.001028 | 0.02129 | 8.038e-5 | 11.62 | 0.0 | 0.0 | 0.2911 | 0.0006484 | 0.006854 |
| 1.575 | 0.003997 |  | 0.0002743 | 13.64 | 0.0 | 0.0 | 0.3249 | 0.003208 | 0.03291 |
| 1.65 | 0.01198 | 0.1589 | 0.0007441 | 15.41 | 1.001e-5 | 0.0001313 | 16.79 | 0.05858 | 0.9467 |
| 1.725 | 0.03004 |  | 0.001689 | 17.2 | 0.0001018 | 0.001708 | 22.62 | 0.09117 | 1.741 |
| 1.8 | 0.06596 | 0.621 | 0.003433 | 18.95 | 0.0003377 | 0.006122 | 24.79 | 0.1854 | 3.573 |
| 1.875 | 0.1301 |  | 0.006262 | 20.56 | 0.0008303 | 0.02087 | 26.34 | 0.297 | 5.718 |
| 1.95 | 0.2357 | 1.732 | 0.01057 | 22.13 | 0.001753 | 0.04707 | 27.87 | 0.4692 | 9.292 |
| 2.025 | 0.3962 |  | 0.01675 | 23.56 | 0.003008 | 0.08547 | 29.35 | 0.6559 | 13.17 |
| 2.1 | 0.6266 | 3.491 | 0.02512 | 24.89 | 0.004424 | 0.1251 | 30.79 | 0.84 | 17.09 |
| 2.175 | 0.9302 |  | 0.0356 | 26.1 | 0.00616 | 0.1867 | 32.17 | 1.109 | 22.36 |
| 2.25 | 1.301 | 5.087 | 0.0479 | 27.14 | 0.05165 | 1.522 | 33.55 | 1.201 | 24.99 |
| 2.325 | 1.65 |  | 0.06049 | 27.25 | 0.403 | 11.3 | 34.86 | 1.482 | 31.76 |
| 2.4 | 1.944 | 2.468 | 0.07268 | 26.72 | 0.7811 | 20.94 | 36.07 | 1.832 | 39.02 |

| Quantity | Value |
|---|---|
| Grid, steps, threads | λ₀/16, 900 steps, 4 threads |
| Base run (a₀ = 1.65): born weight, trapped and counted | 0.001602, 0.0007441 (46.44 %) |
| Base run: plasma electrons above 5.0 mₑc² where J reads (window x ≥ 12.0), against the trapped born weight | 1.001e-5 (1.345 %) |
| Base run: the same over the whole window ahead of the back band | 0.05858 (78.72 times) |
| J and dJ/da₀ at the base | 0.01198, 0.1589 |
| Steepness d ln J / d ln a₀ | 21.89 |
| Best abs(FD − AD)/abs(AD), at h | 1.607e-7, 1.259e-5 |
| The floor of A7: the spread of the three difference quotients around that step, over abs(AD) | 0.005138 |
| Fine scan (33 runs, spacing 0.0004028): largest and RMS departure from a cubic, in changes of J per spacing | 0.04519, 0.01673 |
| Seconds per plain run (49 runs) | 32.36 |
| Seconds per run with one dual (8 runs) | 57.14 |

The tables are the output of `julia +1.13 --project=scripts scripts/optim/born2d.jl table reports/wp-064-born-electron-objective/optim-born2d-20261117T024038`. Weights are in n_c·(c/ω₀)², J in n_c·mₑc²·(c/ω₀)², both per unit length along z. "Trapped" is the born weight with u_x > 5 times the position weight. The roughness row divides the departure of the 33 fine-scan values from the least-squares cubic through them by abs(dJ/da₀) times the spacing.

![The base run in time, the read-out, the scan in a₀, the trapped weights and the finite differences](wp-064-born-electron-objective/optim-born2d-20261117T024038/born2d.png)

## Verification status

- T-830 to T-835 (`scripts/optim/test/runtests.jl`): the position weight's closed form and derivatives; J against the closed form on hand-placed electrons, with the plasma's species, free slots and electrons behind the weight leaving its bits unchanged; the guard on hand-placed plasma particles; ForwardDiff against central differences in a₀ (with the case's weight and with one that cuts through the electrons) and in the dopant's density through a λ₀/8 run to t = 150 (within 10⁻⁵); the summary on hand-placed electrons.
- The run asserts that born electrons are trapped, that at every a₀ the trapped ones lie where S = 1 (the smallest counted share is 0.9995), that a run with duals returns the plain run's J, and that the best central difference is within 10⁻⁴ of ForwardDiff.
- The physics breaker reproduced J at a₀ = 1.5, 1.65, 2.1 and 2.4 and the derivative at 1.65 to the last bit on 1 and 2 threads, and in a run stepped in two pieces.
- The finite-difference check follows A7's scan over the step, but its quotients do not form a plateau: the spread of the three around the best step is 0.5 % of the derivative, set by the structure of J named in the summary, not by rounding.

## Not shown

- Where the percent-level structure of J comes from. The back-edge retirements of the plasma's particles are known to roughen the interior fields late in a run (#370); whether that or the births cause it here was not separated.
- Any other parameter than a₀. The read-out time and the weight's edges were varied only by the breaker, at two or three points.
- The born weight as a fraction of what was released, at any a₀ but the base: by t = 300 the atoms have left the window, and only the base run records the released weight in time.
- A converged resolution, the GPU, Float32 (the case promotes to Float64).
- Whether the plasma's injection is physical. Where J reads and below a₀ = 2.25 it is 4 to 67 macro-particles of ramp weight (breaker); in the second bucket it was counted, not studied.
- The seconds per run on another machine: the breaker's runs shared the cores and took longer, at the same ratio of about 1.8 between a dual and a plain run.
