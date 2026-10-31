# Architecture and work package review

Review date: 2026-10-31. Written by Codex. Scope: the goals, architecture, and WP-003 through WP-009 as present at review time. This is advisory feedback, not a change to the binding goals or work-package contracts.

**The architecture is broadly reasonable. The first delivery is somewhat overcomplicated, especially by specifying the framework before testing it with real physics. Keep the overall structure and change the implementation order.**

The simulation package is still an empty module, so this is a design review; performance and backend compatibility remain unproven.

What looks sound:

- **The core / electromagnetic PIC / Wakefields split.** Particle storage and transfers, Maxwell–particle coupling, and laser-wakefield application features are useful boundaries.
- **The data layout decisions.** SoA storage, cell-plus-offset positions, explicit workspaces, and separation of configuration from evolving state address concrete requirements.
- **A deterministic reference path alongside a fast path.** That is justified by the gradient-checking goal.
- **The numerical acceptance tests.** Partition of unity, gather/deposit adjointness, continuity, and finite-difference scans are much stronger than testing only agreement with another simulator.
- **Optional symbolic and compiler integrations.** Keeping these out of the core dependency chain is the right direction.

The main recommended changes, in priority order:

**1. Test the stepping interface with real electrostatic and electromagnetic schemes before freezing it.**

[WP-007](../../trace/wp/WP-007-core-step.md) specifies one fixed sequence: deposit → solve → gather → push → exchange. But [WP-009](../../trace/wp/WP-009-core-flux.md) needs old and new particle positions for current deposition.

That is not necessarily incompatible: a carefully defined staggered state can carry the previous trajectory. However, the contract does not establish the time levels, initialization, treatment of old positions across periodic wrapping, or placement of field substeps. Those details determine whether the composition is correct. Existing electromagnetic PIC schemes explicitly specify this staggering and splitting; see the [JAX-in-Cell explicit scheme](https://jax-in-cell.readthedocs.io/en/latest/numerics/explicit.html).

A mock module can pass every composition test while missing that problem.

**Recommendation:** let the module or a concrete stepper own the numerical ordering, with shared core operations underneath. Implement tiny electrostatic and electromagnetic examples before declaring a universal step contract. Avoid building a scheduling framework to solve this prematurely.

**2. Move a meaningful gradient experiment earlier.**

The transfer tests mostly differentiate **weights**, where deposition is linear. That is useful coverage, but the scientific problem also needs this chain:

`parameter → fields → particle motion → new deposition → later fields → objective`

Even when parameters enter only through initial weights, evolved positions depend on them. Cell crossings, sorting, and the cell/offset representation therefore matter.

[WP-007](../../trace/wp/WP-007-core-step.md) includes a mock gradient test, but it does not explicitly require those difficult cases.

**Recommendation:** add an early, short, self-consistent simulation with a smooth scalar diagnostic and a Float64 finite-difference scan. Include trajectories that cross cells and a periodic seam. The final optimization objective need not be settled to test this.

That experiment would reduce more architectural uncertainty than implementing the SciML wrapper first.

**3. Specify how deterministic deposition combines contributions from neighboring cells.**

The plan repeatedly says “sort plus segmented reduction.” Sorting particles by their containing cell establishes a useful ordering, but neighboring particle-cell segments can still contribute to the same mesh node.

[WP-006](../../trace/wp/WP-006-core-transfers.md) needs an explicit ownership and reduction strategy for those overlapping contributions. Otherwise two implementations could both follow the prose and have different determinism or scratch-memory properties.

**Recommendation:** make this a small implementation experiment with measured scratch usage and thread-count reproducibility. Keep the existing acceptance tests; resolve the algorithm before relying on it throughout the engine.

**4. Reduce the machinery required before the first useful simulation.**

Some requirements are justified eventually but need not all be mandatory now:

- `DiffCache` for every numeric workspace, even though the state is already constructed with its execution element type.
- A custom tangent-bundle layout before measuring the simple typed-array reference.
- All dimensions, multiple shape orders, and optional number types across the entire initial implementation.
- Mandatory Molly interface research despite Molly compatibility being explicitly secondary.
- A source-text test banning “physics names.” Dependency direction and working modules test the boundary more meaningfully.

In particular, [WP-008](../../trace/wp/WP-008-core-harness.md) acknowledges that the SciML wrapper’s memory multiplier is unknown. Keep SciML integration, but make it an adapter around a working simulation rather than a prerequisite for running one.

Retain dimension-generic interfaces while staging validation and optimization. That preserves the intended direction without requiring every combination before learning whether the design works.

**5. The work packages are good task descriptions, but they need an earlier runnable milestone.**

Their scopes, non-goals, dependencies, and acceptance criteria are unusually explicit. The weakness is that the current batch can finish a considerable amount of infrastructure without producing a physical simulation.

Suggested sequencing:

| Work | Suggested change |
|---|---|
| WP-004 mesh | Keep broadly as written. |
| WP-005 particles | Keep the split; make the species/sort prerequisite explicit in the dependency graph. |
| WP-006 transfers | Resolve deterministic reduction and scratch cost early. |
| WP-007 step | Develop alongside a minimal electrostatic module; keep its interface provisional. |
| WP-009 flux | Develop alongside a minimal electromagnetic step with explicit time levels. |
| WP-008 harness | Follow the first working physics and gradient demonstration. |
| WP-003 cross-code | Separate case definitions and deck preparation from the blocked container runs. |

There is also a small contract mismatch: WP-006 makes cubic shapes optional, while WP-009 requires orders 1–3. This is the sort of drift that duplicated specifications encourage.

For the factory itself, **the extra process is understandable because experimentation with autonomous development is one of the stated goals**. Keep work-package acceptance criteria, provenance, and independent review. But goal hashes only detect changes to goals; they do not detect changes to the architecture contracts that those packages also depend on. Treat trace coverage as navigation, not evidence that the specifications remain consistent.

Keep the architecture’s major boundaries and make the next milestone: **a small periodic PIC simulation, one physical benchmark, and one checked gradient**. Let that result determine the remaining interface details. There is enough design to start; the greatest value now comes from making the abstractions encounter real numerical constraints.
