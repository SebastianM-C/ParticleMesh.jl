# Design record

The design derived from the owner's aims and decisions in `../goals/` (A0 to A10). These documents describe the current design; git is their history (`../goals/README.md`). They change through ordinary pull requests, except `factory-and-observability.md`, which holds the factory settings F1 to F4 and is the owner's.

| File | What it holds | When to read it |
|---|---|---|
| `overview.md` | the picture in one pass, pointers to what is decided, the design decisions still open | first |
| `architecture-draft.md` | §0: the engine/module split, the Reactant rules and the engine contracts (§0.7); then layers L0 to L4, core numerics, parallel layout, differentiation, symbolic front-end, hardware mapping, order of work | when a package touches the engine |
| `open-items.md` | thin spots (B1 to B11), what can wait (C), dependencies and risks (D) | when a package or a question needs it |
| `factory-and-observability.md` | traceability, compute topology, platform facts (§3, with their dates), visualization, billing, the overseer, the human interface, the factory shape, the factory settings F1 to F4 (§5) | for factory work and for F1 to F4 |
| `landscape-and-requirements.md` | background: what exists (Julia, JAX, production PIC, BO practice, reduced models), the MethodOfLines v1 measurements, requirements R1 to R13, risks, sources | when a package points there |
| `sciml-reuse.md` | background: what to reuse from SciML per layer, with what was verified in the package sources | when a package points there |
| `pushers-literature.md` | background for WP-015 and WP-016: the relativistic pushers, the claims of each paper, the test problems | for those packages |
| `workflows-primer.md` | background: how saved Claude Code workflows work, and the factory's workflows | when writing or running a workflow |
