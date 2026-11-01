# WP-015: choices made while building the pushers

Work package: WP-015. Files: `lib/ParticleMeshElectromagnetic/src/pushers.jl`, `src/test_particles.jl`, `ext/ParticleMeshElectromagneticOrdinaryDiffEqCoreExt.jl`, `test/pushers.jl`, `test/pushers_sciml.jl`, `scripts/pushers/`, `reports/pushers.md`.

None of these blocked the work; each is the assumption I proceeded on, for the owner to confirm or change.

## 1. The initial conditions of Higuera and Cary's integrable system

T-236 runs their system (arXiv:1701.05605v1, Eqs. 43–51) at H = 4, p_z = 0 and Δt = 1/10, with p_y,0 ∈ {1.625, 1.775}. The paper states no other initial condition, and those two p_y values are not in arXiv v1 either. Their only other hint is a Vay resonance near p_y ≈ 1.7.

Assumption: each orbit starts on their section plane x = 0, at y = 0, so u_z = 0 and p_x = √(H² − 1 − p_y²) > 0.

## 2. Higuera–Cary is written in Boris's rotation form

Smilei and the paper write the Higuera–Cary rotation through the midpoint, u⁺ = 2ū − u⁻ with ū = (u⁻ + (u⁻·t)t + u⁻×t)/(1 + t²). In exact arithmetic that is the same map as Boris's two-cross-product form with s = 2t/(1 + t²). In T-231 the midpoint form drifted |u| by 1.6 eps per step and the gyrophase by 14 eps per step. The Boris form gives 0.6 and 0.4, as plain Boris does. The module uses the Boris form.

Assumption: rounding behaviour decides between equal maps.

## 3. The test-particle step synchronizes x and u

`push_particle` drifts half a step, pushes, and drifts half a step, so the saved states carry x and u at the same time. This is the PIC leapfrog relabelled, with u at integer and x at half steps; xⁿ is the mean of the neighbouring half-step positions. It needs no stagger start and makes `saveat` and callbacks see synchronized states. It differs from the PIC kernel's single full drift only in rounding.

Assumption: this is what the SciML interface should expose. A stagger with one full drift per step, as in the PIC step, would need a half-step start and an interpolated output instead.

## 4. Order assertions in T-235

- The invariants' orders are asserted over the work package's ω₀Δt range.
- The closure order is asserted on one pair: 0.025 → 0.0125 in Float64, and 0.05 → 0.025 in Float32 and duals. At a₀ = 1/2, Vay's leading closure error changes sign near ω₀Δt ≈ 0.07, so its order over 0.2 → 0.025 is pre-asymptotic. The extra Δt = 0.0125 is where it reaches 2. Float32 reaches its roundoff floor at 0.0125.
- For duals, the order is asserted on the value. The partials (the derivatives of the errors with respect to a₀) are only recorded: the last step, T_p − NΔt, depends on a₀ and jumps with N, so they do not fall at a clean order.

Assumption: these are the readings of "an observed order of 2 within a measured tolerance" that the measurements support.
