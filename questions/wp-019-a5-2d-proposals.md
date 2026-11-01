# WP-019: proposed A5 expected values of the two 2D cases

**Question.** A5 lists only the 1D set. Should A5 take the two 2D periodic cases of WP-019, `landau-oblique-2d` and `weibel-2d` (`scripts/crosscode/cases/`), with the expected values below? The owner accepts by editing A5; nothing here is accepted until then.

**Assumption I proceeded on.** The theory values are the expected values. The single Smilei runs are recorded next to them as one seed each, not as expected values, and no agreement threshold is proposed: A7 bases thresholds on the Smilei–PIConGPU spread, which WP-017 measures.

| Case, observable | Proposed expected value (theory) | Derivation | Smilei, one run (one seed) |
|---|---|---|---|
| `landau-oblique-2d`, damping rate γ/ω_p of E·k̂ of mode [2, 1] | 0.1533595 (ω_r = 1.4156619 ω_p) | least-damped root of 1 + (\|k\|λ_D)⁻²[1 + ζZ(ζ)] = 0 at \|k\|λ_D = 0.5; the isotropic Maxwellian makes it the 1D root of `landau.toml` (`theory.jl`, T-273) | 0.1609 ± 0.0078 (5 maxima), ω_r = 1.4184 |
| `landau-oblique-2d`, Ex/Ey field-energy ratio of the initial wave | 3.9975909 (Yee grid at 128 × 128; 4 in the continuum) | (sin(k_xΔx/2)/Δx)² / (sin(k_yΔy/2)/Δy)² (`yee_energy_ratio`, T-273, T-274) | 3.98629 as pre-registered; 3.9975909124 with the uniform field Smilei adds at t = 0 removed (labelled sensitivity) |
| `weibel-2d`, growth rate γ/ω_p of B·ê, ê = ẑ × k̂, of mode [2, 1] | 0.2637255 | cold-fluid relation for k ⊥ v₀ derived in the header of `scripts/crosscode/theory.jl`, equal to the 1D Weibel relation of WP-014 at the same \|k\| = 2, v₀ = 0.3 c, ω_p (T-273); unverified against the literature | 0.2609 ± 0.0019 (mode), 0.2611 ± 0.0009 (in-plane magnetic energy) |

Numbers, method and runs: `reports/crosscode-smilei-2d/README.md`.

**Points the owner may want to decide with it.**

- The 2D Weibel relation is derived here, not cited. Accepting it as an expected value accepts the derivation.
- Smilei adds a uniform E after its Poisson solve (`scripts/crosscode/README.md`, Smilei specifics), so its total field energies of these cases contain a k = 0 oscillation that the case does not. The mode-projected observables are immune; a comparison on field-energy scalars is not.
- The Landau standard error (4.9 %) is one realization of the thermal noise. The seed spread in both codes is WP-017's.

**Affected:** A5 (`docs/goals/A5-benchmarks.md`), WP-017 and the module's 2D validation (§0.8 item 5), which test against these values.
