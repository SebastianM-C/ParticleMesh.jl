# WP-016: choices made while building the quadratic-unitary pusher and the resolution scan

Work package: WP-016. Files: `lib/ParticleMeshElectromagnetic/src/pushers.jl`, `test/pushers_resolution.jl`, `test/pushers_sciml.jl`, `test/tolerances.toml`, `scripts/pushers/resolution/scan.jl`, `reports/pushers.md`.

None of these blocked the work. Each one is the assumption I proceeded on, for the owner to confirm or change.

## 1. Same signature, one square root

The package asks for `QuadraticUnitary` to have the same `push_momentum(u, E, B, qm_h)` signature as the other pushers, and for its update to need no square root. Gordon and Hafizi need none because they carry γ = u⁰ with u and step in proper time. The module's particle state is u alone, so γ = √(1 + |u|²) is recomputed on entry: one square root, the same as Boris. The γ′ that the spinor update returns is dropped. It equals √(1 + |u′|²) to roundoff (T-241).

Options:
- (a) Keep the signature (taken).
- (b) Carry γ as a fourth particle component. That changes the particle layout and the step for every pusher, which is outside this package.

## 2. What T-241's Minkowski-norm check measures

With γ recomputed from u, γ² − |u|² = 1 holds by construction, so the check could not fail. T-241 therefore steps the four-velocity through `quadratic_unitary_step` and carries γ without ever recomputing it. That checks det Λ = 1 of the real expansion.

## 3. Where Eq. 22 is evaluated

Eq. 22 gives Δs = (Δt/γ)(1 − (q/m)(E·u)Δt/(2γ²)). The module evaluates it at the incoming momentum uⁿ⁻¹ᐟ², with the fields at time n, which is the point where the leapfrog step starts. In the variables of `push_momentum` that is κ = (q/m)Δs/2 = (qm_h/γ)(1 − qm_h(E·u)/γ²). It needs only qm_h, so the negative half step of `initialize!` still works.

## 4. γ − u_z of `QuadraticUnitary` has no order (T-243)

The fields at the particle in the plane wave form a null field, E_x = B_y. There Ψ·Ψ = 0, the truncation is exact, and every step preserves γ − u_z (their Eq. 14), whatever Δs is. Its error is roundoff. T-243 therefore exempts it from the order check and asserts T-241's roundoff bound instead.

## 5. The asymptotic range of the drift and position errors (T-243)

The package expected second order wherever a₀·ω₀Δt ≤ 1/2. That holds for the two invariant errors, γ − u_z and u_x + qA_x, of every pusher; the largest deviation is 0.15. It does not hold for the drift and position errors of Boris, Vay, Higuera–Cary and the exact rotation at a₀ ≥ 5:

- At a₀ = 10 their observed orders are 0.8 to 1.5 over ω₀Δt = 0.2 → 0.025, and they reach 2 only from ω₀Δt ≈ 0.006 on.
- At the same a₀·ω₀Δt = 0.25, a₀ = 5 is at order 1.9 and a₀ = 10 at 1.1. So a₀·ω₀Δt alone does not set the asymptotic range of these two errors.

T-243 asserts:
- the invariant orders on the package's grid where a₀·ω₀Δt ≤ 1/2;
- the drift and position orders on a grid extended to ω₀Δt = 0.2/2⁷, where a₀·ω₀Δt ≤ 1/16;
- that every error falls over the package's grid.

The acceptance text of WP-016 says so.

## 6. The exact-rotation fix rewrites the formula, not the map (#203)

Rodrigues' formula, cos θ·u⁻ + …, multiplies u⁻ by the rounded cos θ ≈ 1. Writing it as an increment with cos θ − 1 = −kθ² does not remove the drift either: the issue reported that, and I reproduced it at about 1200 eps over 10⁴ steps. What removes it is writing the same rotation in Boris's two-cross-product form, u⁻ + (u⁻ + u⁻×t)×s, with t = tan(θ/2)·ŵ and s = sin θ·ŵ: the Cayley form at the exact half angle.

tan(θ/2) is unbounded at θ = π, so this form is used for θ² < 1. Above that, Rodrigues' formula is used. The two are chosen with `ifelse`, and the unused value is kept finite. The map is the same in exact arithmetic, so the algorithm is unchanged.

## 7. The LWFA time steps

The report is asked to apply the thresholds to "the CFL Δt of the planned LWFA runs at their a₀". A5's 2D LWFA case is not specified yet: it has no a₀, resolution or Δt anywhere in the repository. The report therefore applies the thresholds to representative steps, and states those as assumptions:
- 0.95 of the 2D Courant limit;
- Δy = 4Δx;
- 16 to 64 cells per laser wavelength along x;
- a₀ ∈ {1, 2, 3, 5, 10}.

The check should be redone once the A5 case exists.

## 8. The reference orbit

The package suggests `Vern9` where the closed form needs inverting. Over one period no inversion is needed. For Gordon and Hafizi's diagnostic, which ends at a lab time that is not a period, η is found from t = η + z(η) by Newton's method. That is exact to roundoff and more accurate than `Vern9`. T-237 already checks `Vern9` against the closed form.
