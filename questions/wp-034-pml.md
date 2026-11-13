# Choices WP-034 took that its text did not settle

## 1. Which PML

The owner chose a convolutional PML like PIConGPU's. **Assumption:** the complex-frequency-shifted stretched-coordinate PML in the recursive-convolution form of Roden and Gedney (2000), with PIConGPU 0.8.0's coefficients b = exp(−(σ/κ + α)τ), c = σ(b − 1)/(κ(σ + κα)), read in its source (`include/picongpu/fields/absorber/pml/Pml.kernel` at commit `6bb3470e`). Smilei v5.1 uses a uniaxial PML with auxiliary D and H fields and a bilinear time rule instead (`src/ElectroMagnSolver/PML_Solver2D_Yee.cpp` at commit `438d43d8`). Its discretization is not implemented here, only its profiles.

The matched layer of WP-027 stays: the PML is a second `absorber`, so WP-027's tests and report keep their meaning.

## 2. How it enters the step

**Assumption:** as a correction after each Yee sub-step. The plain sub-step already added the vacuum term ±τ·D_yF, and the correction adds ±τ·((1/κ − 1)·D_yF + ψ) on the layer rows, with ψ advanced first. The Yee kernels, the antennas and the interior are untouched. A fused kernel would round in another order and nothing else. The four y-difference terms are Bx (D⁺_yEz), Bz (D⁺_yEx), Ex (D⁻_yBz) and Ez (D⁻_yBx).

Every convolution field advances once per step with τ = Δt, as in PIConGPU: E's in the E step, B's in the second B half step from E^{n+1}, and the first B half step of the next step reuses that ψ_B, which was made from the same E. An earlier version advanced ψ_B in both half steps with τ = Δt/2. It is consistent and stable too, but the physics breaker measured 21× (8 cells, normal incidence) to 2.7·10⁵× (16 cells, 0.4 rad) more reflection with it, and a continuum reflection falling as Δ² instead of Δ⁴. Step 1 starts ψ_B at zero, so its first half step reads 0 instead of c·D⁺_yE⁰, which differs only for fields that start nonzero inside the layer. The discrete theory of the PML (`pml_rt`) is derived for this scheme.

## 3. Where it sits and its profile

**Assumption:**

- **Placement.** WP-027's rows: the outer `cells` rows on each side along y, meeting at the seam. This is also where PIConGPU puts its absorber, inside the box. Smilei's PML lies outside its box.
- **Depth.** Counted in integer half rows, as WP-027 counts it, so the inner edge is exactly vacuum in every element type.
- **Profiles.** σ = σ_max·r^m, κ = 1 + (κ_max − 1)·r^{m_κ}, and α = α_max·(1 − r)^{m_α} inside the layer, where r is the relative depth. These are PIConGPU's profiles, with an order of κ's own so that Smilei's (σ = 20u², κ = 1 + 79u⁴) can be expressed.
- **Default σ_max.** PIConGPU's optimum 0.8·(m + 1)/Δy when not given, in ω₀ units.
- **Default α_max.** Zero. PIConGPU's 0.2 S/m is (0.2/ε₀)/ω₀ ≈ 9.6·10⁻⁶ in ω₀ units at λ₀ = 0.8 µm, which is computed, not taken from a PIConGPU run.
- **Grading orders.** Integers. PIConGPU allows non-integer ones, and its default 4.0 is an integer.

Smilei measures its profile over (cells + ½)·Δ, with primal and dual rows offset by half a cell. ParticleMesh's depth convention differs from it by up to half a cell.

## 4. The state of the convolution fields

The convolution fields are field state: they are among the arrays a run evolves (`evolved`, WP-056), so `state_partition` holds them, a snapshot saves them and a gradient through the harness's state vector sees them. `initialize!` zeroes them with the fields, and no step treats them specially (T-744). They are not parameters (A3).

## 5. What is left out

- **Plasma in the layer.** The current deposited in the layer is not damped, and with α = 0 a plasma that fills the layer builds up a large field there. In `open_plasma_2d` on 16 × 32 cells (2 × 2 particles per cell, 6000 steps, Float64), the field energy ½Σ(E² + B²) at step 6000 was 188 in the interior and 180 in the layer with the default PML of 4 cells, 83 and 546 with 8 cells, 0.93 and 6.7 with α_max = 0.2, 2.3 and 6.3 with α_max = 1, and 0.54 and 0.0024 with WP-027's matched layer of 4 cells (rising over the run in every PML case). The mechanism is the CPML's low-frequency limit: with α = 0, s_y = κ + σ/(iω) diverges at ω → 0, so a slow current in the layer meets no y-curl. **Assumption:** the default stays PIConGPU's α = 0, which a cross-code comparison needs, and a run with plasma in the layer sets α_max (0.2 to 1) or keeps the plasma out of the layer. Damping the current in the layer, as the PML literature does for charged particles, is left for later (#343).
- **The time rule.** Only PIConGPU's exponential rule is implemented, not Smilei's bilinear one.

## 6. The moving window's back edge

The window's x-guards are its field-free outside, which reflects what reaches the back edge (WP-028's questions §3). **Assumption:**

- **The numerics.** `EMNumerics(; back_absorber = PML(...))`, a second absorber next to `absorber`, refused without a window. It takes the same `PML` as the y axis. There is no case-file key yet, because a case file cannot describe a moving window yet.
- **Placement.** The window's first `cells` columns, graded towards the back face, depth in integer half columns as on the y axis. The front has no layer: the undisturbed outside comes in there. Behind the layer the x-guards stay zero, so what the layer leaves reflects there and crosses it again.
- **The terms.** The four x-difference terms of the 2D curls (By from D⁺ₓEz, Bz from D⁺ₓEy, Ey from D⁻ₓBz, Ez from D⁻ₓBy), each with its convolution field, on PIConGPU's once-per-step rule. Bz and Ez also take the y-PML's terms where the two layers meet, each with its own ψ.
- **The shift.** The profile is fixed in the window, which moves at c. A shift moves E, B and every convolution field one column back together, so each ψ stays with its field's point, and the last layer column starts from zero. PIConGPU keeps its PML at the back of its moving window and disables only the far side (`fields/absorber/pml/Pml.hpp`, `getLocalThickness`). It slides by whole devices' domains rather than single cells, so how its ψ moves is not comparable cell by cell.
- **The measures.** A packet into the back edge of a static window, against `back_rt`, the scheme's time-harmonic reflection of the layer with the window's zero x-guards behind it (T-576). And T-472's pulse against free space, a static box with 100 more columns behind (T-570). A layer moving at c has no time-harmonic theory of its own; the static window and the shifts' bitwise checks (T-571, T-574) are taken to cover it together.
- **The antenna.** A plane must stay out of the layer while it emits, as it must stay off the x-guards (WP-028). The state refuses a layer that reaches it earlier.
- **α at the back edge.** The default stays α = 0. In a moving window every profile tried is bounded. In a long static phase neither choice is safe: α = 0 with plasma in the layer builds up field (§5, #343), and α > 0 in vacuum grows a uniform Bz (#357).
- **In the window** (WP-028), the PML of the open y axis works as in a fixed box. The window's zero x-guards follow the PML's correction, and the convolution fields shift with E and B (T-549).
