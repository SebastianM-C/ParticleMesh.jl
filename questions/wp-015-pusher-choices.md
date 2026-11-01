# WP-015: choices made while building the pushers

Work package: WP-015. Files: `lib/ParticleMeshElectromagnetic/src/pushers.jl`, `src/test_particles.jl`, `ext/ParticleMeshElectromagneticOrdinaryDiffEqCoreExt.jl`, `test/pushers.jl`, `test/pushers_sciml.jl`, `scripts/pushers/`, `reports/pushers.md`.

None of these blocked the work; each is the assumption I proceeded on, for the owner to confirm or change.

## 1. Higuera and Cary's integrable system (their Eq. 48) is not run

T-236 asks for Higuera and Cary's Eq. 48 with H = 4, p_z = 0, p_y,0 ∈ {1.625, 1.775} and Δt = 1/10. The repository records the parameters (`docs/plan/pushers-literature.md`) but not the system itself, and the session that built WP-015 could not reach the paper (arxiv.org was outside its network allowance). Writing the system down from memory would risk a fabricated test case.

Options: (a) a session with access to Phys. Plasmas 24, 052104 (2017) adds Eq. 48 to `pushers-literature.md` and the case to T-236 and `scripts/pushers/energy.jl`; (b) drop the case and keep only the second system.

Assumption: (a), as a follow-up. T-236 runs the second system (a static non-uniform B with a static non-quadratic potential: B = r ẑ, φ = 10⁻²/r, the field of Qin et al. 2013) at two initial conditions chosen here, one nonrelativistic and one with γ₀ ≈ 1.42.

## 2. Vay's volume factor is derived here, not copied

T-234 compares Vay's one-step Jacobian determinant with "Higuera and Cary's ratio J(u_i)/J(u_f) (their Eqs. 40–42)". Without the paper, the test derives the ratio from Vay's implicit equation: det = J(u⁻)/J(u⁺) with J(u) = 1 + (τ² + (u·τ)²)/γ(u)⁴, τ = (q/m)(Δt/2)B (derivation in the header of `test/pushers.jl`). The test confirms it to a few eps for γ up to 10³, so the formula is right for this implementation. Whether it is the same expression as their Eqs. 40–42 is not checked.

Assumption: the derived form is what the acceptance criterion means; option (a) of §1 should also compare it with the paper.

## 3. Higuera–Cary is written in Boris's rotation form

Smilei and the paper write the Higuera–Cary rotation through the midpoint, u⁺ = 2ū − u⁻ with ū = (u⁻ + (u⁻·t)t + u⁻×t)/(1 + t²). In exact arithmetic that is the same map as Boris's two-cross-product form with s = 2t/(1 + t²). In T-231 the midpoint form drifted |u| by 1.6 eps per step and the gyrophase by 14 eps per step. The Boris form gives 0.6 and 0.4, as plain Boris does. The module uses the Boris form.

Assumption: rounding behaviour decides between equal maps.

## 4. The test-particle step synchronizes x and u

`push_particle` drifts half a step, pushes, and drifts half a step, so the saved states carry x and u at the same time. This is the PIC leapfrog relabelled, with u at integer and x at half steps; xⁿ is the mean of the neighbouring half-step positions. It needs no stagger start and makes `saveat` and callbacks see synchronized states. It differs from the PIC kernel's single full drift only in rounding.

Assumption: this is what the SciML interface should expose. A stagger with one full drift per step, as in the PIC step, would need a half-step start and an interpolated output instead.

## 5. Order assertions in T-235

- The invariants' orders are asserted over the work package's ω₀Δt range.
- The closure order is asserted on one pair: 0.025 → 0.0125 in Float64, and 0.05 → 0.025 in Float32 and duals. At a₀ = 1/2, Vay's leading closure error changes sign near ω₀Δt ≈ 0.07, so its order over 0.2 → 0.025 is pre-asymptotic. The extra Δt = 0.0125 is where it reaches 2. Float32 reaches its roundoff floor at 0.0125.
- For duals, the order is asserted on the value. The partials (the derivatives of the errors with respect to a₀) are only recorded: the last step, T_p − NΔt, depends on a₀ and jumps with N, so they do not fall at a clean order.

Assumption: these are the readings of "an observed order of 2 within a measured tolerance" that the measurements support.
