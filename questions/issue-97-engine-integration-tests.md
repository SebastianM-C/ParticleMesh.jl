# What replaces the electrostatic reference in the engine's tests (issue #97)

## Question

The engine's own test suite ran the electrostatic reference (`es_reference.jl`, the WP-010 spike's step) as an integration test: T-136 (bits across reruns and threads), T-137 (Gauss after every step), T-141 and T-142 (Landau and two-stream against theory), T-143 (the gradient against `fd_scan`), T-144 (allocation per step) and T-170 (the WP-011 split kept the bits). With the reference deleted, should the engine keep a step of its own for these, or rely on the module's step?

## Assumption proceeded on

The engine keeps no step. A physics-free "proxy step" made of the same engine pieces would be the reference under another name, which A9 rules out. Each claim moves to where it belongs:

- **Engine claims stay in the engine suite, at kernel level.**
  - Bits across reruns, thread counts and launch shapes: the subprocess digests of `threads_worker.jl`. These are T-172 to T-177, T-113, T-260 and T-266, and they cover the scan, the sort, the deposit, `deposit_flux!`, load, gather, guards and advance.
  - T-136 keeps its sort-permutation check.
  - T-144 measures the allocation of every N = 1 engine operation, as T-268 does at N = 2.
- **Step-level claims are the module's, which already tests each one:**
  - Gauss after every step: T-212.
  - Landau and two-stream against theory: T-214 and T-215.
  - The gradient through a whole step: T-219, which `fd_scan`'s §0.7 example now cites.
  - Bits across reruns and 1 and 4 threads: T-218.
  - Allocation per step: T-220.
- **Removed outright:** T-170 (the split's bits) and T-260's recheck of T-136's digests. Both compared the reference's own runs with recorded values.

CI runs the module's suite next to the engine's, so a change in an engine kernel is still exercised by a full step on every pull request. The engine does not depend on the module.

The same change points WP-012's ParticleMesh column at the module (`scripts/em1d/crosscode.jl`). WP-013's CUDA validation already runs the module's step rather than the reference (re-derived in #141).
