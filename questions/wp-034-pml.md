# Choices WP-034 took that its text did not settle

## 1. Which PML

The owner chose a convolutional PML like PIConGPU's. **Assumption:** the complex-frequency-shifted stretched-coordinate PML in the recursive-convolution form of Roden and Gedney (2000), with PIConGPU 0.8.0's coefficients b = exp(−(σ/κ + α)τ), c = σ(b − 1)/(κ(σ + κα)), read in its source (`include/picongpu/fields/absorber/pml/Pml.kernel` at commit `6bb3470e`). Smilei v5.1 uses a uniaxial PML with auxiliary D and H fields and a bilinear time rule instead (`src/ElectroMagnSolver/PML_Solver2D_Yee.cpp` at commit `438d43d8`). Its discretization is not implemented here, only its profiles.

The matched layer of WP-027 stays: the PML is a second `absorber`, so WP-027's tests and report keep their meaning.

## 2. How it enters the step

**Assumption:** as a correction after each Yee sub-step. The plain sub-step already added the vacuum term ±τ·D_yF, and the correction adds ±τ·((1/κ − 1)·D_yF + ψ) on the layer rows, with ψ advanced first. The Yee kernels, the antennas and the interior are untouched. A fused kernel would round in another order and nothing else. The four y-difference terms are Bx (D⁺_yEz), Bz (D⁺_yEx), Ex (D⁻_yBz) and Ez (D⁻_yBx).

The B convolution fields advance on each half step with τ = Δt/2, so each half step holds ∂_yE fixed over its own half. PIConGPU updates its B convolution once per step (its kernel says so). The discrete theory of the PML (`pml_rt`) is derived for the scheme as implemented here.

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

- **Current in the layer.** It is deposited and not damped. PIConGPU damps the current deposited in its PML (Lehe et al. 2022), which WP-035's runs may show is needed.
- **The time rule.** Only PIConGPU's exponential rule is implemented, not Smilei's bilinear one.
- **The back edge of the moving window.** Folded into WP-034, but it waits for WP-028 (pull request #323) to merge.
