# WP-029: the binomial current filter, its transfer function, its effect on Gauss's law and on the 2D rates

| | |
|---|---|
| Work package | WP-029 (goals A2, A4, A7, A8) |
| Commits | the filter, its wiring and T-440 to T-446 `14c7336`; the script `e75ce22`; this report in WP-029's pull request |
| Runs | `em2d-binomial-filter-20261103T193911` (commit `e75ce22`, `dirty = false`) |
| Input hashes | SHA-256 of every input file in the run's manifest: `wp-029-binomial-filter/em2d-binomial-filter-20261103T193911.manifest.toml` |
| Environment | Julia 1.13.3, KernelAbstractions 0.10 CPU backend; Intel Core i9-14900K; 8 threads (the Reference step is bitwise across thread counts, T-443) |
| Figures | `wp-029-binomial-filter/transfer.png`, `gauss.png`, `weibel_rate.png`, each with its JSON sidecar; `summary.json` holds every number |
| Status | measured, not independently reviewed; no number here is to leave the repository before that review |

## 1. Setup

`lib/ParticleMeshElectromagnetic/src/filter.jl`: `BinomialFilter(p)` is a numerics choice of `EMNumerics` (A10), with the pass count from the case file's `filter_passes` (`numerics_from_case`, a non-negative integer, otherwise an `ArgumentError`). After the deposited current is folded, the 1D and 2D steps apply `p` passes of `(J[j−1] + 2J[j] + J[j+1])/4` along every resolved axis to Jx, Jy and Jz, then the mesh solve. The step works the same way in both deposition modes because the filter acts on the mesh after the deposition. Each pass fills the guards of its source, writes every interior node once, and alternates between the field and the scratch field `ρ` (only `initialize!` reads it otherwise). The kernel is one work item per node, so its bits do not depend on the launch shape. The charge is not filtered.

Closed forms. One pass along axis d multiplies the Fourier mode m_d of n_d cells by cos²(π m_d/n_d) on any stagger. A half-cell shift of the samples changes only the phase of each coefficient, and the stencil is a convolution. So p passes over every axis transfer H_p(m) = Π_d cos^{2p}(π m_d/n_d). The filter F commutes with the backward differences of the continuity equation, so each step changes D⁻·E by F(ρⁿ⁺¹ − ρⁿ). Two exact consequences follow: Qⁿ = D⁻·Eⁿ − Fρⁿ is constant, and the plain Gauss residual is rⁿ = D⁻·Eⁿ − (ρⁿ − ρ̄) = (F − 1)(ρⁿ − ρ⁰) + r⁰. In the linear theory of a mode m, ∂ₜE = ∇×B − FJ multiplies the plasma response by H_p(m), so ω_p² → H_p·ω_p².

Zero passes. Before the change a scratch script recorded 30 state digests: the 1D cases landau, two-stream and weibel (64 per cell, one time unit), and the 2D cases landau-oblique-2d and weibel-2d (32 × 32 cells, 4 × 4 per cell, 40 steps at four times the case Δt). Each was run in the Reference and Fast modes for Float32, Float64 and `Dual{Float32,3}` at one thread. After the change all 30 were identical. T-364 and T-365 give the same numbers as on `main` (Landau γ −1.10 %, ω +2.6 % from theory; Weibel γ −1.35 %).

## 2. Transfer function (`transfer.png`)

A random field went through `binomial!` (1D: 64 cells; 2D: 32 × 24 cells) with p = 1, 2, 4, 8 passes. Every DFT coefficient of the output, compared with H_p times the input's coefficient, agrees to at most 0.33 eps·p·N·max|input| in 1D and 0.16 in 2D. The ratio has no imaginary part beyond roundoff, so the filter does not shift the phase of any mode. The tests (T-440) do the same for 16 cells (1D) and 12 × 10 (2D), with p = 1, 2, 3, 5 and all three element types, partials included: at most 0.18. They also sample one mode on each of the four 2D staggers (at most 0.54), and they check bit for bit that a constant is kept, each axis' Nyquist mode becomes +0.0, a mirror symmetry is kept and zero passes touch nothing.

## 3. Gauss's law (`gauss.png`, T-441)

Oblique Landau case at T-364's test size (32 × 32 cells, 16 × 16 electrons per cell, 5 % perturbation, t = 18, 610 steps), checked every 10 steps:

| passes | max over the run of max\|rⁿ\| / max\|ρⁿ − ρ̄\| | max \|Qⁿ − Q⁰\| / (eps·(n₁+n₂)·max\|ρ\|·√(k+1)) | max over modes \|r̂ − r̂⁰ − (H_p − 1)(ρ − ρ⁰)^\| / max\|(ρ − ρ⁰)^\| at t = 18 |
|---|---|---|---|
| 0 | 9.66e-14 | 0.173 | 2.2e-14 |
| 1 | 0.814 | 0.0604 | 1.8e-14 |
| 2 | 0.939 | 0.0496 | 1.9e-14 |

So the filtered Gauss law holds to roundoff, and the plain one is violated by an amount of the order of the charge fluctuation itself. The violation is concentrated where it is expected: mode by mode, |r̂ − r̂⁰|/|(ρ − ρ⁰)^| lies on 1 − H_p(m) (right panel). The grid-scale noise modes, which dominate max|ρ − ρ̄|, have H_p ≈ 0 and are almost entirely missing from D⁻·E. The physical mode (2, 1) of the case misses 1 − H_p = 4.7 % (p = 1) or 9.2 % (p = 2) of its charge change at test size. At the case's full size (128 × 128) that is 0.30 % for one pass (H_1 = 0.99699). In the tests (T-441: 1D Landau case with p = 1 and 3, the two-species 2D plasma of T-360 with p = 1 and 2, both modes, every type), |Qⁿ − Q⁰| stays below 0.173 at that scale (bound 2), and the mean of rⁿ stays zero to 0.04 of eps·scale·Π n_d. The plain residual relative to the charge deviation reaches 0.70 (1D, p = 1) to 1.04 (2D, p = 2).

## 4. The rates against the filtered theory (`weibel_rate.png`)

Weibel at T-365's test size (64 × 64 cells, 4 × 4 per beam per cell; cold, so there is no seed noise), with the case's window fit. The theory is `weibel_cold_growth_2d(k, v₀; ωp2 = H_p)`, the cold relation with ω_p² → H_p·ω_p² at the mode (2, 1):

| passes | H_p(2, 1) | γ measured | γ theory | relative |
|---|---|---|---|---|
| 0 | 1 | 0.26017 | 0.26373 | −1.35 % |
| 4 | 0.9529 | 0.25627 | 0.25858 | −0.90 % |
| 8 | 0.9080 | 0.25271 | 0.25350 | −0.31 % |
| 16 | 0.8245 | 0.24296 | 0.24351 | −0.23 % |

The filter lowers γ by 6.6 % from p = 0 to p = 16, and the filtered theory predicts 7.7 %. The measured rate stays within 1.35 % of the filtered theory at every pass count, which is the unfiltered run's own distance from theory at this resolution.

Landau at the size of §3, fitted on the seed mean of the complex mode amplitude over the case's first 8 seeds (the rule of its ensemble observable), against √H·landau_root(kλ_D/√H). The fit is noise-limited at this size: with no filter, γ = 0.1815 against 0.1534 and ω = 1.3798 against 1.4157, from 4 maxima. With 2 passes, γ = 0.2313 against 0.1705 and ω = 1.3352 against 1.3796. With 4 passes fewer than three maxima clear the noise floor, so there is no fit. The frequency shift with two passes is −3.2 % measured and −2.6 % in theory. The rate does not settle the filtered Landau theory at this size; the Weibel rows do the theory comparison.

## 5. Verification status and open items

- Tests: T-440 (transfer, closed form and exact identities), T-441 (filtered Gauss law), T-442 (the step filters all three folded components, and the mesh solve takes the filtered current), T-443 (bitwise across reruns, launch shapes and 1 and 4 threads; Fast agrees with Reference to 12.6 of T-362's scale), T-444 (the case files' `filter_passes`; zero passes leave the guards and the scratch alone), T-445 (allocation flat with the filter), T-446 (the same on CUDA), and T-374, which now covers the filtered steps.
- No cross-code comparison of a filtered case yet: the Smilei and PIConGPU generators still refuse `filter_passes ≠ 0` (#291).
- Gauss's law with the filter holds only in its filtered form. Filtering ρ for diagnostics, or a compensated filter (both are options in other codes), is not implemented.
