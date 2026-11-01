# WP-021: four choices of the periodic solve, taken with the owner before the code

WP-021 as cut left four points open or stated them in a way that conflicts with A4 or with the field layout. The owner settled each before the implementation; they are recorded here with the options so that a later change of mind has its context. The contract that implements them is `docs/plan/architecture-draft.md` §0.7 contract 14.

## 1. T-314 against A4: is the Dual primal bitwise the Float32 run?

**Question.** WP-021 asked T-314 for "primal bitwise equal to the Float32 run". A4 and §0.7 (element types) say the value part of a dual run is not the primal run bit for bit, because the compiler may fuse operations differently on the two paths. Which holds for the solve?

**Options.**
- Assert bitwise agreement across element types for this operator.
- Assert agreement within a measured `C·eps(T)` and record whether it happened to be bitwise.

**Assumption WP-021 proceeded on (owner's decision).** The second. T-314 asserts that the primal agrees with the Float32 run within `C·eps(T)` (`tolerances.toml`, entry T-314) and that the partials of parameters in `s` equal the Float32 solve of the partials of `s` (the solve is linear in `s`), within the same bound. Whether the primal was bitwise is recorded in the measured ratios and in an `@info` line, not asserted; on the CPU backend with Julia 1.13.3 it was bitwise for every grid of the test. Bitwise agreement across element types is not a contract.

**Affects.** WP-021 (T-314), `lib/ParticleMeshCore/test/spectral.jl`, `tolerances.toml`.

## 2. Staggering: forward or backward difference for the gradient?

**Question.** WP-021 asked for "the backward-difference gradient that the module turns into E". With `s` and `φ` on ρ's nodes (index `j`) and Ex half a cell to the right of the node at the same index (the 1D module, `deposit_flux!`, the §0.8 item 5 layout), is the gradient a forward or a backward difference?

**Options.**
- Backward difference `(φ[j] − φ[j − e_d])/Δ_d` stored at `j`: the face half a cell to the left, which is not the layout of Ex and Ey.
- Forward difference `(φ[j + e_d] − φ[j])/Δ_d` stored at `j`.

**Assumption WP-021 proceeded on (owner's decision).** The forward difference: `G_d[j] = (φ[j + e_d] − φ[j])/Δ_d` at index `j`, the face half a cell to the right, `Center()` on axis `d` (for `N = 2` Gx at (Center, Node) and Gy at (Node, Center), the Yee layout of Ex and Ey; for `N = 1` Ex at `Center()` as in the 1D module). Its backward-difference divergence `Σ_d (G_d[j] − G_d[j − e_d])/Δ_d` equals `s − mean(s)`. The WP text is corrected.

**Affects.** WP-021, `gradient_periodic!`, WP-023 (which takes Ex and Ey from it).

## 3. Sign convention

**Question.** Which sign does the engine solve, and which does the module apply?

**Options.**
- The engine solves `∇²φ = −s` (the electrostatic potential for `s = ρ`) and the module takes `E = −∇φ`.
- The engine solves `∇²φ = s − mean(s)` and returns `G = +∇φ`, so `div G = s − mean(s)`.

**Assumption WP-021 proceeded on (owner's decision).** The second. The engine names no equation: it solves `+L φ = s − mean(s)` and returns `G = +∇φ`. A module that wants `div E = s` takes `E = G`; with `s = ρ` that is the electric field in the convention `div E = ρ`. The docstrings of `inverse_laplacian_periodic!` and `gradient_periodic!` and contract 14 state it.

**Affects.** `lib/ParticleMeshCore/src/spectral.jl`, contract 14, WP-023.

## 4. The API

**Question.** What are the engine's names and calls, given that no exported name may contain Gauss, Poisson or Electrostatic (T-171)?

**Options.**
- One combined call that returns the gradient.
- A workspace allocated once and two calls, the solve and the gradient.

**Assumption WP-021 proceeded on (owner's suggestion, chosen by the implementer).** Two calls and a workspace: `SpectralWorkspace(backend, T, grid)` holds the twiddle tables and the real and imaginary spectrum arrays at fixed size; `inverse_laplacian_periodic!(φ, s, grid, ws; workgroupsize = nothing)` and `gradient_periodic!(G::NTuple{N}, φ, grid; workgroupsize = nothing)`. The outputs have their guards filled, the guards of the inputs are not read, the mean of φ is zero (the k = 0 mode is set to zero), `N = 1` and `N = 2`. Two calls, because φ itself is a useful output (a module may want the potential for a diagnostic) and the gradient is a separate, cheap mesh operator; a module calls both.

**Affects.** `lib/ParticleMeshCore/src/spectral.jl` and its exports, contract 14, WP-023.
