# Spike: Enzyme reverse mode through the 2D step

| | |
|---|---|
| Work package | WP-050 (goals A0, A3, A7, A8) |
| Commit of the runs | `31bddb0b4fcfe0aac235ec0ae56d533649777096` (clean tree) |
| Runs | the nine `enzyme-*` runs in `wp-050-enzyme-spike/`, each a `result.toml` and a `manifest.toml` |
| Input hashes | SHA-256 of `scripts/enzyme/chain_rule.jl`, `scripts/enzyme/setup.jl`, `lib/ParticleMeshElectromagnetic/test/em2d_cases.jl`, the side environment's `Project.toml`, `Manifest.toml` and its `step2d.jl`, in each manifest |
| Environment | Julia 1.13.3, 4 threads; side environment: Enzyme 0.13.219 (Enzyme_jll 0.0.307, LLVM.jl 9.13.2, GPUCompiler 2.10.0), KernelAbstractions 0.9.45 CPU backend, AcceleratedKernels 0.5.12, ForwardDiff 1.4.7; Intel Xeon Processor @ 2.10GHz, a 4-core cloud container |
| Status | measured, not independently reviewed; the timings are single samples on a shared machine |

## 1. What blocks Enzyme in the repository

Enzyme cannot be installed next to the engine. Every KernelAbstractions 0.10 release requires LLVM.jl 10 (so do GPUCompiler ≥ 2.20 and UnsafeAtomics 0.4), while Enzyme 0.13.219, the latest release, caps LLVM.jl at 9.13.2, and Enzyme's main branch on 2026-11-09 asks for `LLVM = "9.10"`. `Pkg.add("Enzyme")` in `lib/ParticleMeshElectromagnetic/test` fails to resolve. No test was added, and the package is parked until an Enzyme release accepts LLVM.jl 10 (#385).

The spike therefore runs in a side environment that `scripts/enzyme/setup.jl` builds: copies of the two `lib/` packages with KernelAbstractions relaxed to 0.9.45 (the engine uses only `@kernel`, `@index`, `@Const`, `allocate`, `zeros`, `ones` and `get_backend`), and Enzyme, ForwardDiff at pinned versions. KernelAbstractions 0.9.45 still has the EnzymeCore extension with the reverse rules for CPU kernel launches; 0.10 dropped it, so a future Enzyme-on-0.10 needs those rules back.

## 2. What else blocks, and the workarounds

| Attempt | Outcome |
|---|---|
| The whole pipeline inside the gradient, `α ↦ EMState + initialize! + step!` | Enzyme aborts the process in type analysis ("illegal insertion", `TypeTree.h:364`) while compiling `map(problem.species)` in `EMState`. Not reduced; the spike differentiates a prebuilt state instead (#385, item 3). |
| `step!` on a prebuilt state, one species | works, Fast and Reference mode |
| `step!` on a prebuilt state, two species | `EnzymeInternalError`, "Illegal address space propagation", on the `for s in st.species` loops over the species tuple. Reduced to an Enzyme-only example: #384. With the loops written as `foreach` (`setup.jl --foreach`: the push loop, `kinetic!` and the `zip(st.species, st.ws)` loops) two species work in Float64 and Float32; with only the push loop rewritten, Float32 still fails in `kinetic!` (found by the physics breaker). |

The rewrite is not in the engine: WP-047 edits the same push loop, and the change belongs with the move into the test suite (#385, item 2).

## 3. The check

The case is WP-023's `plasma_step_2d` (Esirkepov current, Quadratic shape, Boris, Yee, 10 × 8 cells, 3 × 3 particles per cell; one species is its electrons alone), the parameter α the electrons' density perturbation amplitude (it enters through the weights, A3), the functional f the field energy Σ(Ex² + Ey² + Ez² + Bx² + By² + Bz²) after n steps of the initialized state x.

- **α direction.** Enzyme's gradient ∇f with respect to every floating-point array of x (about 20 000 entries for one species, 36 000 for two), dotted with ForwardDiff's tangent dx/dα of `initialize!`, against ForwardDiff's df/dα of the whole run. The α tangent is zero on Bx, By, Bz, Ez, the offsets and u_z, so this direction does not test the gradient there.
- **Random direction.** A Dual state with a random tangent v on every array, whose forward derivative must equal ⟨∇f, v⟩. This covers the arrays the α direction leaves untested.
- **Verdict.** Both within 64·eps(T)·n·c, where c = Σ|c_k|/|Σc_k| is the cancellation of the per-array contributions to the α direction.

Both checks compare two AD modes of the same code, so they do not see a physics defect: the mutation breaker flipped the sign of the push and dropped Jz, and the identity still held to 0.0. Physics correctness rests on the module's T-tests.

## 4. Results

| Mode | Steps | Species | T | Seed | α direction, relative | Random direction, relative | c | Verdict | Repeat bitwise |
|---|---|---|---|---|---|---|---|---|---|
| Fast | 1 | 1 | Float64 | 7 | 3.7e-16 | 3.8e-16 | 1.00 | pass | yes |
| Fast | 1 | 1 | Float32 | 7 | 2.0e-7 | 1.6e-7 | 1.00 | pass | yes |
| Reference | 20 | 1 | Float64 | 7 | 4.6e-15 | 3.6e-16 | 4.36 | pass | yes |
| Fast | 100 | 1 | Float64 | 3 | 6.5e-13 | 6.4e-15 | 21.4 | pass | yes |
| Fast | 100 | 1 | Float32 | 3 | 2.2e-3 | 3.7e-5 | 21.4 | pass | yes |
| Fast (`--foreach`) | 1 | 2 | Float32 | 7 | 1.0e-7 | 1.1e-7 | 1.00 | pass | yes |
| Fast (`--foreach`) | 20 | 2 | Float32 | 11 | 1.4e-6 | 8.8e-8 | 1.09 | pass | yes |
| Reference (`--foreach`) | 20 | 2 | Float64 | 11 | 4.1e-15 | 1.2e-14 | 1.09 | pass | yes |
| Reference, workgroup 16 | 20 | 1 | Float64 | 7 | 2.1e-15 | 8.4e-16 | 4.36 | pass | **no** |

Two species without `--foreach` fail as section 2 says (run, not in the table). The error of the α direction grows with the steps and with c: at 100 steps it is 2900·eps in Float64 and 1.8e4·eps in Float32, and the ratio of the two follows eps(T) within a factor of about 6 (physics breaker), which reads as amplified roundoff rather than an AD defect; the random direction stays within 30·eps(T) in Float64. The physics breaker also ran 60 steps (Reference, seed 3), the positive species' amplitude as the parameter with the kinetic energy as the functional (40 steps, two species, seed 5), all within the same bounds.

**Determinism.** With the default workgroup size (1024 in KernelAbstractions 0.9) each kernel of this case is one workgroup and runs serially, and repeats are bitwise equal. With workgroup size 16 on 4 threads, the reverse pass differs bitwise between two calls in Reference mode too, while the primal does not (#386; likely the adjoint of the gathers, a scatter that Enzyme accumulates with atomics, not checked). Reverse-mode gradients are not yet reference mode in the sense of A4.

**Cost** (single samples after compilation, on a shared 4-core machine; take them as orders of magnitude). Enzyme's first call compiles for 97 to 129 s. One reverse pass costs 20 to 50 primal runs (0.29 s against 6.6 ms at 20 steps; 1.6 s against 61 ms at 100 steps), one Dual pass 1 to 2. The reverse pass gives the gradient with respect to all 20 000 to 36 000 state entries at once, so it pays off beyond a few dozen parameters; for the few parameters of WP-043 and WP-045, ForwardDiff stays cheaper.

## 5. Open issues

- #385: move the check into the test suite once Enzyme accepts LLVM.jl 10; the `foreach` rewrite of the species loops; the unreduced abort when the state is built inside the gradient.
- #384 (`upstream`): Enzyme's "Illegal address space propagation" on a loop over a homogeneous tuple of structs.
- #386: the threaded reverse pass is not bitwise reproducible.
- Not tried: the moving window, the antenna, the open boundaries and the PML, the filter, GPU (KernelAbstractions 0.9's GPU backends with Enzyme), checkpointing for long runs (the reverse pass stores the tape of every step).
