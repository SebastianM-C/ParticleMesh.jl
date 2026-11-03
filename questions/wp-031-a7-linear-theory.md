# WP-031: what tolerance against linear theory the cold cases support (A7)

Work package: `trace/wp/WP-031-offset-convergence.md`; measurements: `reports/wp-031-offset-convergence.md`. A7 asks for agreement thresholds against the benchmark set, set by measurement and accepted by the owner editing A7. This file proposes; nothing binds until then.

**Question.** A test of a code's growth rate on the cold cases against linear theory: against which reference value, and within what tolerance?

## What was measured

Smilei v5.1 on the CPU box (AMD EPYC 9655), Δx at the case's Courant number, Δt at fixed Δx, the perturbation amplitude, the particles per cell and the seed varied; each pre-registered fit compared with the same fit, over the same samples, of the exact linear cold-fluid solution of the case's initial state (`cold_ivp`, `scripts/crosscode/theory.jl`). θ is the dispersion root.

| | two-stream `growth_rate` | Weibel `growth_rate_mode` | Weibel `growth_rate_energy` |
|---|---|---|---|
| offset γ/θ − 1 at the case's resolution | −1.500 % | WM0 | WE0 |
| its limit Δx, Δt → 0 | −1.407 % | WMinf | WEinf |
| the transient (linear solution, same samples) | −0.96 % to −1.04 % (one sample more or less) | WMlin | WElin |
| residual γ/θ − γ_lin/θ, limit | −0.446 % (amplitude × 1/10 and × 10: −0.49 %, −0.48 %) | WMres | WEres |
| discretization part of the residual at the case's resolution | −0.019 % (Δx), +0.012 % (Δt) | WMdisc | WEdisc |
| order of convergence | 2 (observed 2.1 to 2.5) | WMord | WEord |

The offsets are not discretization error. They are the transient of the other roots in the window, which the linear solution gives exactly, plus a residual at a fixed fraction of saturation that does not depend on the resolution, the amplitude, the particles per cell or the seed, and that ParticleMesh reproduces within 0.02 % on two-stream: the onset of saturation, physics of the case rather than of any code. At the case's resolution the discretization error of the fitted rate is at most about 2 × 10⁻⁴ θ (WEIBELDISC2).

## Proposal

**P1. Reference value: the linear solution through the run's own samples, plus the measured nonlinear residual.** For a run of a cold case, γ_ref = γ_lin + r_NL θ, with γ_lin the fit of `cold_ivp`'s solution over the samples the run's window took (`point_fits`, `scripts/crosscode/convergence.jl`) and r_NL the converged residual above (two-stream −0.446 %; Weibel WMres for the mode, WEres for the energy). A code passes when **|γ − γ_ref| ≤ 1 × 10⁻³ θ**. The value is five times the largest discretization part measured at the cases' resolutions (2 × 10⁻⁴ θ), rounded to a power of ten; the measured spread of r_NL over amplitude, particles per cell and seed (±0.04 % on two-stream, WEIBELSPREAD) lies inside it. A test of this kind catches a 0.1 % error in a code's growth rate, which a comparison with the root cannot (P2).

**P2, not recommended: the dispersion root with a tolerance wide enough for the offset.** Against θ alone the tolerance has to cover the transient and the onset of saturation, at least 1.6 % (two-stream) and 1.2 % (Weibel) at these windows, and the transient moves by 0.07 % with one sample more or less in the window. Such a test passes a code whose rate is wrong by a per cent.

Conditions of P1: the run's window and γ_lin use the same samples (as WP-017's P1 asks for the two-code comparison); the initial state of the run is the case's (both codes load it; ParticleMesh's half-step kick differs from Smilei's convention by O(Δt), 10⁻⁴ θ at the case's Δt, inside the tolerance, #226); r_NL is a measured constant of each case and window, so a change of the case or the window rule needs a new scan.

## Not proposed

- A value for the 1D Landau case (#167): noise-limited at 8 % to 10 % per ensemble; the scan cannot resolve discretization effects there (report, §4).
- A change of the cases' registered windows: a window further from saturation shrinks r_NL (−0.22 % at [10⁻³, 10⁻²] on two-stream) but enlarges the transient; with γ_lin computed exactly the transient costs nothing, so the windows can stay.

## Open questions for the owner

1. Whether A7's agreement with linear theory should be read as P1 (a reference value that includes the transient and the measured onset of saturation) rather than as agreement with the dispersion root.
2. Whether r_NL, which is measured with Smilei, may enter a reference value at all, or whether the tolerance should instead be one-sided around γ_lin (−0.6 % ≤ (γ − γ_lin)/θ ≤ +0.1 %).

Assumption taken here: none accepted; the numbers are proposals.

Affected: A5, A7; the cold cases `two-stream.toml`, `weibel-2d.toml`; WP-012, WP-017, WP-023.
