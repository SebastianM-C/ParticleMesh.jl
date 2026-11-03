# Choices WP-027 took that its text did not settle

## 1. Which absorbing layer "simple absorbing layer" means

A2 asks for "a simple absorbing layer first, PML later". **Assumption:** the matched lossy layer, applied by operator splitting. A conductivity σ(y) damps E and B alike: after each Yee sub-step of length τ, every component is multiplied by exp(−σ·τ) at its own row. With the same σ for E and B, the continuum medium has the impedance of vacuum, so a wave at normal incidence enters it without reflection. At oblique incidence it does reflect, and that reflection grows with the angle (T-430, `reports/wp-027-open-boundaries.md`). The splitting leaves the curl kernels and the antenna untouched. It is unconditionally stable, since each factor is at most one.

Alternatives not taken:

- The semi-implicit lossy Yee update, (1 − σΔt/2)/(1 + σΔt/2), inside the curl kernels. It agrees with the splitting to second order in σΔt.
- Damping E only. That is not matched, and so reflects at normal incidence too.
- PIConGPU's exponential damping absorber. From its documentation, not checked in its source: it multiplies the fields in its absorber cells by a factor per step. That is close to this layer with σ constant in time, but its grading and its constants are not verified here.

## 2. Where the layer lives

Smilei puts its PML cells outside the simulation box (documentation, not checked in its source). **Assumption:** the layer is the outer `cells` rows of the box on each side along y. The engine's grid then stays a `PeriodicGrid{2}`, the loader fills the whole box as before, and the two layers meet at the periodic seam as one absorbing slab. Consequences:

- The plasma is loaded in the layer too. Its particles there see damped fields, and Gauss's law does not hold in the layer (T-431 measures how far from it the layer is).
- A case that wants the plasma clear of the layer must say so in its density profile. That is the LWFA case's business, together with WP-028's profile.

## 3. The parameters

The package asks for the thickness and the strength as case parameters. **Assumption:**

- `cells`: the thickness, in cells along y.
- `strength`: σ at the box edge, a rate in units of ω₀ (A1).
- `order`: the grading σ = strength·(depth/thickness)^order, 2 by default.

They form the numerics' `AbsorbingLayer`. A case file gives them as `[numerics.absorbing_layer]`, next to `grid.boundaries = ["periodic", "open"]`. They are numerics, not differentiable parameters (A3).

There is no default in the case files yet. The measurements give the trade-off. At a fixed nominal attenuation strength·cells = 32, R at 0.4 rad falls from about 10⁻³ (8 cells) to about 10⁻⁸ (32 cells). At 16 cells the transmission through the slab falls with the strength while the reflection grows (the report's figures). The LWFA case (`questions/wp-017-lwfa-case.md`) still has to pick values.

## 4. Which particles leave

**Assumption:** a particle is retired when it crosses the box edge along the open axis, the periodic seam y = 0 ≡ L_y. Under the Courant limit, an ordinary move changes a particle's cell along y by at most one. A crossing of the seam changes it by n₂ − 1. The particle is retired after the current of that last move is deposited, so its charge leaves inside the layer, as does its field. Nothing is injected back: the box loses plasma through its open edges. The x seam stays periodic.

## 5. What is left out

- The initial field is still the periodic solve of contract 14. With an open axis it is the field of the periodic images, which differs for a charged initial state. A neutral plasma, the LWFA case, has none.
- Axis 1 is not open: the moving window (WP-028) handles the longitudinal boundaries.
- N = 1 has no open boundary.
