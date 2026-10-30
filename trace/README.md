# Traceability records

Content-addressed records for requirements (R-), decisions (D-), interfaces (I-), work packages (WP-), tests (T-), runs (RUN-) and sessions (S-), each linking upstream by hash so that a change upstream marks everything downstream stale. The format, the builder and the first records are the "trace MVP" work package; until it lands, `docs/plan/factory-and-observability.md` section 1 is the specification. `trace/build/` is generated and ignored by git.
