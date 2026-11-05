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

The convolution fields are part of the field state but not of the arrays in `state_partition`. **Assumption:** `step!` sets them to zero at step 1. Every run from step 1 is then the same run, including a solve of the SciML harness, which takes its steps in order on one integrator. A gradient through the harness's state vector does not see them. They are not parameters (A3), and a PML run through the harness differentiates the fields and particles only.

## 5. What is left out

- **Plasma in the layer.** The current deposited in the layer is not damped, and with α = 0 a plasma that fills the layer builds up a large field there. In `open_plasma_2d` on 16 × 32 cells (2 × 2 particles per cell, 6000 steps, Float64), the field energy ½Σ(E² + B²) at step 6000 was 188 in the interior and 180 in the layer with the default PML of 4 cells, 83 and 546 with 8 cells, 0.93 and 6.7 with α_max = 0.2, 2.3 and 6.3 with α_max = 1, and 0.54 and 0.0024 with WP-027's matched layer of 4 cells (rising over the run in every PML case). The mechanism is the CPML's low-frequency limit: with α = 0, s_y = κ + σ/(iω) diverges at ω → 0, so a slow current in the layer meets no y-curl. **Assumption:** the default stays PIConGPU's α = 0, which a cross-code comparison needs, and a run with plasma in the layer sets α_max (0.2 to 1) or keeps the plasma out of the layer. Damping the current in the layer, as the PML literature does for charged particles, is left for later and tracked as an issue.
- **The time rule.** Only PIConGPU's exponential rule is implemented, not Smilei's bilinear one.
- **The back edge of the moving window.** Folded into WP-034, but it waits for WP-028 (pull request #323) to merge.
