+++
id = "WP-005"
status = "withdrawn"
goals = ["A1", "A3", "A4", "A7", "A8"]
depends_on = ["WP-004"]
adds_dependencies = []

[checked]
A1 = "69f5bec32fe5d073e16e7811300bcb1ef5092a73"
A3 = "2e44fafc2ffe3634ff3dd6e84bb90825e76c585e"
A4 = "8c6d1910f937970d7f14e1a844ed3ff49f67802a"
A7 = "25b4f7853f52a9efd57ebd467f84844c7b105f70"
A8 = "ca00019a5a518e1747bce0643e09c87eb11494b2"
+++
# WP-005: engine core, particles: species, carry, loading, exchange, sort and compaction

Withdrawn: replaced by WP-011, the engine/module split written from the spike's code (`reports/spike-electrostatic-1d.md`), which owns every contract this package proposed. The text below is kept as the record of the proposal.

The particle side of the engine in `lib/ParticleMeshCore`. Contracts: `docs/plan/architecture-draft.md` §0.7 items 4 (exchange and status), 5 (buffers, attributes, count, position advance, momentum, weights, loading), 6 (sorting), 13 and 17; §0.4 rules 1 and 2.

## Scope

Fixed-capacity SoA species with the core attribute set, module-declared extras and per-species constants; the device-side count as a high-water mark and an overflow counter; the position-advance device function with its floor carry and a generic `drift!`; `exchange_particles!(species, grid)` on periodic axes with the `UInt8` status for bounded faces; counter-based Philox4x32-10 loading with quiet start, weights from the density and Box–Muller momenta; in-place `sort_by_cell!` with the total-order key and segment offsets, and `compact!`.

## Deliverables

1. **Species type and accessors first.** Push the species type, its accessors (cell, offset, momentum, weight, status, extras, constants) and its constructor early in the PR, or as a first small PR, because the transfers package (WP-006) starts from them.
2. **Position advance** `advance(cell, offset, Δ)` in cell units, keeping the offset in [0, 1) including the Float32 round-up-to-1 case, and `drift!` for tests and the step package's mock module.
3. **`exchange_particles!`**: wraps any displacement on periodic axes; on bounded faces sets status 1 + k (face numbering from WP-004) and leaves the particle in place for the module's `boundary!`.
4. **Loading.** In-house Philox4x32-10 device function; key = (seed, stream id), counter = (global cell per axis zero-padded to three words, slot in the high 16 bits and draw index in the low 16 bits of the fourth word), with an error when ppc or the draws exceed 16 bits; stream id = (species index, purpose). Quiet start on a per-cell lattice; weight n(xₚ)/ppc in units of n_c from an isbits density callable n(x, p) with parameters passed explicitly (A3); Gaussian momenta by Box–Muller with Base functions only. Injection writes into `count+1:capacity`, clamps at capacity and adds the excess to the overflow counter.
5. **Sort and compaction.** `sort_by_cell!` in place with key = cell · capacity + slot (`Int64`), cell the column-major `LinearIndices` over interior local cells, dead particles keyed at prod(n) + 1; one permutation applied to every SoA array; segment offsets into a caller-owned buffer; the key helper exported for WP-006. `compact!` as a stable partition. All scratch in a workspace allocated once (item 17). Check AcceleratedKernels' sort against the total order on the CPU; if it is not deterministic across thread counts, use a different sort and say why.
6. Tests in `lib/ParticleMeshCore/test/particles.jl`, its table in `tolerances.toml`.

The package is delivered in two pull requests, because it is large: first the species type, the position advance, `exchange_particles!`, sort and compaction (deliverables 1, 2, 3, 5); then loading (deliverable 4: Philox with its known-answer vectors, quiet start, Box–Muller, injection). The species type lands in the first one, which unblocks WP-006. Random123.jl comes into the depot as a dependency of KernelAbstractions 0.10, so its tests are a citable source for T-040's known-answer vectors (whether they contain them is unverified).

## Non-goals

No shapes, gather or deposition (WP-006). No pusher: modules own pushers; only `advance` and `drift!` here. No moving-window refill, only the counter layout that makes it reproducible. No Reactant (A9). No SpecialFunctions. Parameters never enter through positions (A3).

## Acceptance

Test ids T-040 to T-059.

- T-040: the Philox known-answer vectors match, taken from the Random123 distribution's `kat_vectors` file or Random123.jl's tests, with the source path and version cited in the test; never written from memory. If neither is reachable from the sandbox, write `questions/philox-kat-vectors.md` and keep the test as `@test_broken` naming the missing source.
- T-041: for every `T` of the A8 matrix, the same seed gives bitwise identical loads on reruns and with `JULIA_NUM_THREADS` = 1 and many (A4: bitwise holds per element type and backend, not across element types); a different stream id gives different draws.
- T-042: free streaming with `drift!` over N steps, for every `T` of the A8 matrix: with a dyadic displacement (for example 1/8 cell) the result is bitwise equal to the analytic position; with a general displacement it is within `C·eps(T)·N` of the Float64 analytic position, with the measured maximum and the proposed `C` recorded in `tolerances.toml`.
- T-043: `exchange_particles!` on periodic axes conserves the count exactly; bounded faces set the documented status.
- T-044: the loaded weights equal n(xₚ)/ppc at the lattice positions bitwise for Float32 and Float64, kernel against a host evaluation on the same backend (the CPU test matrix, A8); for `Dual` parameters the value part agrees with the Float32 evaluation within a proposed `C·eps(Float32)·|w|`, relative to the weight (A4: the compiler may fuse the kernel's operations differently on the `Dual` path, observed as 1 ulp in the spike's loader), with the measured maximum recorded. Their sum equals the same quadrature of n computed independently in Float64 to a proposed `C·eps(T)·count` bound; parameter tangents of the weights match the exact derivative of n in p for `Dual` parameters (A3).
- T-045: `sort_by_cell!` returns a bitwise permutation of the input, stable within cells, whose segment lengths sum to the alive count, with the last offset equal to it; `compact!` preserves order; both reset `count` to the alive count.
- T-046: bitwise identical sort results across `JULIA_NUM_THREADS` = 1 and many.
- T-047: injection beyond capacity writes what fits and increments the overflow counter; no host-side read inside the call.
- T-048: allocation does not grow with particle count for `drift!`, `exchange_particles!` and `sort_by_cell!` (§0.7 item 13).

## Depends on

WP-004 (grid, `GridState`, test scaffolding).

## Budgets

CPU timings per particle for sort and exchange recorded without a threshold (A7's throughput budget is on the GPU); proposed `C` values for T-042 and T-044 recorded in `tolerances.toml` marked "proposed". No new dependencies, so not tier 2 unless the sort needs one.
