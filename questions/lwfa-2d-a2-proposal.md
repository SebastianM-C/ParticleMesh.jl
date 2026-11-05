# The 2D LWFA case: proposed values for A5

`questions/wp-017-lwfa-case.md` lists what A5's 2D LWFA case needs, and A5 gives no values. To get a first ParticleMesh–Smilei comparison running, `scripts/lwfa/cases/lwfa-2d-a2.toml` proposes a set. **Nothing in it is the owner's; replace any value and the scripts regenerate both codes' inputs.**

| Parameter | Proposed | Why this value |
|---|---|---|
| λ₀ | 0.8 µm | only for the SI numbers in the report (A1 fixes ω₀ as the unit) |
| a₀ | 2 | mildly nonlinear: a clear wake, little or no self-injection, so the two codes can be compared deterministically before injection physics enters |
| w₀ | 28 c/ω₀ (3.6 µm) | k_p·w₀ = 2.8, close to the 2√a₀ matching rule of thumb (not checked against a cited source here) |
| τ | 26 /ω₀, the 1/e half width of the field envelope (intensity FWHM 30.6 /ω₀ ≈ 13 fs) | c·τ_FWHM ≈ λ_p/2, the resonance rule of thumb |
| n₀ | 0.01 n_c (≈ 1.7 × 10¹⁹ cm⁻³ at 0.8 µm) | λ_p = 62.8 c/ω₀ fits 2.5 wake periods behind the pulse in the box |
| ramp | 260 c/ω₀ of vacuum, then a 60 c/ω₀ sin² up-ramp, then the plateau | the C¹ profile ParticleMesh has (`DensityRamp`); no down-ramp |
| focus | at the start of the plateau | |
| box | 1536 × 256 cells ahead of the injection plane, 301 × 201 c/ω₀; transverse periodic | 7.2 waists wide, so periodic y stands in for an absorber, which the two codes do not share (`wp-017-lwfa-case.md` §3) |
| resolution | Δx = λ₀/32, Δy = 4Δx, Δt at 0.95 of Yee's 2D limit | |
| particles per cell | 2 × 2 electrons, immobile ions | |
| steps | 7189 (t_end ≈ 1301 /ω₀, about 1000 c/ω₀ of propagation in plasma) | |
| diagnostics | E_x, E_y and ρ_e every 553 steps; the electron kinetic-energy histogram | |
| polarization | linear, in the simulation plane (E along y) | |
| filter | none | PIConGPU can only do one pass (`wp-017-lwfa-case.md` §3); none keeps Gauss's law exact in ParticleMesh |
| moving window | from t = 256, at c, along x | after ParticleMesh's antenna has finished |

**Assumption:** these values stand until the owner sets A5's case. They are sized for the CPU box (minutes per code), not for an experiment.
