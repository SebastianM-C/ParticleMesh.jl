# ParticleMesh as a software-factory experiment: agents, loops, observability

Companion to `open-items.md`. Status: draft; the agent-platform facts are from the Claude API reference bundled with Claude Code (2026-10) and the Claude Code docs as cited; verify prices and limits against the current platform documentation before relying on them.

Premise (open-items F): a significant part of the implementation is done by autonomous agent teams and loops; a human reviews; the process itself is an experiment. Two things are therefore first-class deliverables next to the code: a traceability layer that shows how requirements propagate into decisions, work, tests and runs, and an observability layer over the agents that do the work.

## 1. Traceability: requirements as content-addressed records

The idea is the one EDM's run manifests already use for simulation runs (run records with identities and commits, campaigns as data, derived results pointing at their parents), applied to planning artifacts.

Node types, one file each under `trace/` in the repository:

| Node | Prefix | Holds |
|---|---|---|
| Requirement | `R-` | statement, rationale, status (proposed, accepted, superseded), `derived_from` |
| Decision (ADR) | `D-` | the choice, alternatives rejected, `derived_from` requirements, consequences |
| Interface contract | `I-` | a function or data-layout contract between layers (e.g. `deposit_current!` signature and semantics), `derived_from` decisions |
| Work package | `WP-` | scope, non-goals, `implements` interfaces, `satisfies` requirements, acceptance tests and budgets |
| Test | `T-` | what it checks, tolerance, `verifies` requirements, where it lives (Julia testset name) |
| Run record | `RUN-` | a validation or benchmark run (existing EDM run-manifest format), `verifies` requirements, inputs, commit |
| Agent session | `S-` | an autonomous run: agent/platform, WPs touched, start and end commit or PR, tokens and cost, outcome grade, review status |

Every node stores a content hash of itself and the hashes of the upstream nodes it was derived from at the time it was written. A builder script recomputes hashes and marks any node whose recorded upstream hash no longer matches as **stale**; staleness propagates downstream. That is the "cascading requirements" view: edit R-012 and the builder lists every decision, interface, work package, test and run that must be re-reviewed, and nothing else. A node leaves the stale state only when someone (human or agent, recorded as an `S-` node) re-reviews it and rewrites its upstream hashes.

Hooks into the code:
- Julia testsets carry their trace ids in the name (`@testset "T-031 Gauss law residual [R-012]"`); a small macro can do the same with less typing. CI exports test results (JUnit XML or a custom logger) and the builder attaches pass/fail per commit to the `T-` nodes.
- Run manifests gain a `requirements` field; the EDM manifest tooling already records commit, config, hardware and outputs.
- Agent sessions are written by the agent harness itself: Claude Code hooks at session or subagent stop, Managed Agents `session.usage` events and webhooks (facts in section 3).

Outputs of the builder: `trace/graph.json` (nodes, edges, status, hashes, cost), a coverage matrix (requirements versus tests and runs), the stale frontier after any change, cost per work package, and drill-down links to the session transcript or Console session view. The viewer can reuse the dependency-free static-page pattern of EDM's static results dashboard, with a graph view added.

CI gates, kept minimal at first: fail on an untraced test, on a merge into code owned by a work package whose accepted requirements are stale, and on a run record without requirement links.

## 2. Compute topology for the agents

The hard constraint is where the hardware is. Cloud-hosted agents can do CPU-backend development, tests, documentation, trace maintenance and reviews. Anything with CUDA, the Smilei and PIConGPU comparisons, and campaigns needs the local 4080 SUPER or a rented GPU VM. Two ways to connect hosted agents to that hardware exist (section 3); the fallback is Claude Code sessions running on the machine itself.

Suggested split:
- Cloud, autonomous: core numerics on the KA CPU backend, unit and conservation tests, documentation, requirement-graph maintenance, first-pass reviews, nightly trace rebuilds.
- GPU machines, autonomous but budgeted: CUDA backend work, kernel benchmarks, FP32-versus-FP64 drift tests, cross-code validation, campaigns.
- Human: requirement changes, decision records, merge approval for anything touching an interface contract, final review of gradient claims.

## 3. Platform facts (verified from the bundled Claude API reference, 2026-10)

### Managed Agents (Claude API, beta)
- Agent configs are persisted and versioned; sessions pin to a version. Sessions stream events; a Console session viewer shows transcript, per-tool statistics, tokens in and out and cost per session with no setup.
- **Self-hosted sandboxes**: `config: {type: "self_hosted"}` keeps the agent loop on Anthropic's side while bash, file operations and code run in a container you control through an outbound-polling worker (`EnvironmentWorker.run()` in the SDK or `ant beta:worker poll`). This is the route to GPU hardware: the worker runs on the GPU machine and only needs outbound connectivity. Verified details: the worker is `EnvironmentWorker` in the Python, TypeScript or Go SDK (always-on `run()`, or `handle_item()` for one claimed work item), or `ant beta:worker poll` with the fixed built-in toolset (bash, read, write, edit, glob, grep); a per-session container is spawned from the poller's `--on-work` script (the reference shows `docker run` with the forwarded `ANTHROPIC_*` variables); a webhook on `session.status_run_started` can wake the poller instead of keeping it always on; the host needs `/bin/bash` and a POSIX filesystem. The reference does not mention GPUs at all: GPU access is a property of the container you spawn (GPU passthrough in the run script), not something the platform provides or blocks. Sessions in a self-hosted environment cannot attach file or GitHub-repository resources (cloud environments only); the repository must be present in the container.
- **Session budgets**: a hard cap in list cost set at session creation; the session pauses at the cap and resumes only when the budget is changed or removed. List cost counts model tokens at list price, web searches, and running time at $0.08 per hour.
- **Outcomes**: a session can be started with a rubric; a separate grader iterates the agent against it until it passes or hits a maximum number of iterations. Work-package acceptance criteria map directly onto such rubrics.
- **Scheduled deployments**: cron-scheduled sessions with per-run records and budgets, for nightly validation or trace rebuilds.
- **Multi-agent sessions**: a coordinator with a roster of subagents sharing one container, plus dynamic workflow runs for fan-out work.
- **Observability**: `session.usage` events with cumulative tokens and list cost; `span.model_request_end` events with per-request token usage; webhooks on state transitions; OpenTelemetry export both from the SDK (client side) and from the API (server side, configured by an org admin through the Admin API or Console) into any OTLP backend. The API-side spans cover each model request and each Managed Agents turn; SDK-side spans cover client tools and tools in a self-hosted sandbox.
- Billing: sessions are priced at API rates plus runtime (the reference only states list-price mechanics).

### Claude Code side (from code.claude.com docs, 2026-10-30)

- **Agent teams** (experimental, `CLAUDE_CODE_EXPERIMENTAL_AGENT_TEAMS=1`): a lead plus teammates, each a full session with its own context, sharing a task list and mailbox; spawned only from interactive sessions, never from `claude -p` or the Agent SDK; one team per session, no nesting, all teammates on the lead's machine (team state under `~/.claude/teams/`); hooks `TeammateIdle`, `TaskCreated`, `TaskCompleted`. Docs: https://code.claude.com/docs/en/agent-teams.md
- **Workflows** (`Workflow` tool, `/workflows`): JavaScript scripts orchestrating subagents in the background with `agent()`, `pipeline()`, `parallel()`, `phase()`; resumable with cached agent results; saved to `.claude/workflows/` and run as `/<name>` with `args`; 16 concurrent agents by default, 1,000 per run; counts against subscription usage with no per-run dollar cap; the `ultracode` keyword is ignored from `-p` and scheduled tasks, but a saved workflow can run unattended from `claude -p` when `Workflow(<name>)` is allowed and the subagents' tools have allow rules; subagents cannot launch workflows. Docs: https://code.claude.com/docs/en/workflows.md
- **Scheduled tasks** (`/schedule`, research preview): run on an Anthropic cloud VM (4 vCPU, 16 GB RAM, 30 GB disk, no GPU, no Julia preinstalled) or, for Team and Enterprise plans only, a self-hosted environment; triggers are cron (hourly at most), an API fire endpoint, and GitHub events; results are cloud sessions with transcript, diff and PR; counts against subscription usage. Local alternative: Desktop scheduled tasks on your own machine. Docs: https://code.claude.com/docs/en/routines.md
- **Remote Control**: the local `claude` process does the work on your machine (so on the GPU boxes), reachable from web and mobile through outbound HTTPS polling only; `claude remote-control` server mode keeps many sessions (`--capacity`, `--spawn worktree`); needs a claude.ai login, not an API key or setup token, which matters for headless boxes (log in once interactively, keep it in tmux); server mode gives up after about ten minutes of network outage. Cross-session messaging reaches sessions on other machines through it. Docs: https://code.claude.com/docs/en/remote-control.md , https://code.claude.com/docs/en/cross-session-messaging.md
- **Cloud sessions** (`claude --cloud`): Anthropic VM, cannot drive local machines; `--teleport` pulls a session back. Docs: https://code.claude.com/docs/en/claude-code-on-the-web.md
- **Telemetry**: `CLAUDE_CODE_ENABLE_TELEMETRY=1` with OTLP metrics and logs exporters (set in the shell or `~/.claude/settings.json`, not project settings); metrics for sessions, cost (estimate), tokens by type, tool decisions, active time, with attributes `agent.name`, `skill.name`, `model`, `effort`, `query_source`; per-tool events `tool_decision` and `tool_result` with duration and error type; beta traces (`CLAUDE_CODE_ENHANCED_TELEMETRY_BETA=1`) with spans for interactions, LLM requests and tools, and `agent_id`, `parent_agent_id`, `workflow.run_id` identifiers; prompts and tool parameters redacted unless opted in. Docs: https://code.claude.com/docs/en/monitoring-usage.md
- **Hooks**: 33 events including `SessionStart`, `SessionEnd`, `PreToolUse`, `PostToolUse`, `Stop`, `SubagentStart`, `SubagentStop` (with the last assistant message), `TaskCompleted`, `TeammateIdle`; handler types `command`, `http` (POST JSON), `mcp_tool`, `prompt`, `agent`; inputs carry `session_id`, `transcript_path`, `agent_id` (whether workflow agents fire hooks: open-items B11). Subagent transcripts live under `~/.claude/projects/<project>/<session>/subagents/` with a 30-day default cleanup. Docs: https://code.claude.com/docs/en/hooks.md
- **Reproducible roles**: `.claude/agents/*.md` frontmatter pins `model`, `effort`, `tools` allowlist, `disallowedTools`, `permissionMode`, `maxTurns`, `skills`, `hooks`, `isolation: worktree`; a teammate spawned from such a definition takes its tools, model and effort; `CLAUDE_CODE_EFFORT_LEVEL` overrides everything; 20 concurrent subagents and depth 3 by default. Docs: https://code.claude.com/docs/en/sub-agents.md
- **Managed Agents cloud sandbox** (for comparison): Ubuntu 24.04, up to 8 GB RAM and 10 GB disk, no GPU, Julia not preinstalled; self-hosted sandboxes accept only memory-store resources, so the repository must already be in the container. Billing is API billing: tokens at list rates, $0.08 per session-hour while running, web search per request; whether self-hosted sandboxes incur the runtime charge is not stated. Docs: https://platform.claude.com/docs/en/managed-agents/cloud-sandboxes-reference.md , https://platform.claude.com/docs/en/managed-agents/self-hosted-sandboxes.md , https://platform.claude.com/docs/en/about-claude/pricing.md

Two constraints follow for the factory. No hosted surface has a GPU or Julia; every GPU task and, in practice, every task that must compile the Julia stack runs on machines we control, reached through Remote Control (subscription) or a Managed Agents self-hosted worker (API billing). And the "autonomous team" primitives differ in how unattended they can be: agent teams need an interactive lead; saved workflows run unattended from `claude -p` with allow rules; Managed Agents sessions and deployments are unattended by construction and are the only ones with hard dollar caps.

## 3b. Visualization as the inspection channel

Design premise: much of the inspection is plot-driven. Three audiences read them: the human reviewer, the agents doing the work (Claude reads images, so a plot is an input an agent can act on), and the dashboard. Consequences for the design:

- **A standard plot set per run, generated from the run record, not ad hoc.** Fixed names, fixed layouts, fixed axes conventions, so that two runs or two codes are comparable at a glance and an agent can ask for "the standard set" and read it. Physics set: field slices (E_z, E_y or laser envelope, density), longitudinal lineouts of the wake, phase space (x, p_x) of plasma and bunch, energy spectrum and cut-off, conservation residuals versus time (energy, Gauss's law), and for AD runs the tangent fields and the gradient-versus-finite-difference plot. Cross-code set: the same panels overlaid for ParticleMesh, Smilei and PIConGPU on the matched case. Performance set: particle-steps per second versus particle count and GPU, roofline-style points.
- **Numbers next to every plot.** Agents reason better with a figure plus the few numbers it encodes (peak field, cut-off energy, residual norms), so each standard plot is emitted together with a small JSON or TOML sidecar of the plotted scalars, and the trace builder reads the sidecar, not the image.
- **Plots are derived records.** The EDM pattern applies directly: a plot is a `derived` sidecar that depends on a run id, with the script, commit and parameters recorded, so a plot can always be regenerated and the dashboard can group them into sweeps. Plot generation is a package extension (CairoMakie for headless PNG and SVG; GLMakie or WGLMakie for interactive sessions), with recipes on the solution and diagnostic types so the same call works for a CPU test and a GPU campaign.
- **Factory views are plots too.** The trace graph (nodes coloured by status, stale frontier highlighted), the requirement-coverage matrix, test status over commits, and cost and tokens per work package over time belong in the same dashboard, generated by the same builder, so that "what changed and what does it affect" is answered by looking rather than by reading records.
- **Agent review loops can be plot-driven.** A work package's rubric can require specific standard plots and state what they must show (for example: the two-stream growth-rate fit overlays the analytic slope; the Gauss residual stays flat at roundoff). The grader or reviewer agent reads the plot and the sidecar numbers; the human sees the same artifacts in the dashboard.
- **Plots and reports (A13).** CairoMakie from `scripts/`, every figure a PNG with a JSON sidecar of its scalar values; Markdown reports in `reports/` with the figures committed and a record header (work package, commit, run ids, input hashes); the current-state page is static HTML the trace builder generates from `trace/graph.json`, the latest digest and the report index, published to GitHub Pages on merge, with an optional Claude-artifact copy for commenting.


## 3c. Billing surfaces and where each kind of work runs

Verified billing facts (code.claude.com, platform.claude.com, 2026-10-30):
- Cloud sessions, cloud scheduled tasks, Projects threads and ultrareview run on Anthropic VMs and "count toward your plan's usage limits ... there's no separate charge for the cloud VM".
- The Agent SDK (Claude Code as a Python or TypeScript library, same tools, hooks, subagents, sessions and `.claude/` loading) must use API-key authentication; claude.ai login is not allowed for SDK agents. Managed Agents is API billing too.
- Cloud VM: 4 vCPU, 16 GB RAM, 30 GB disk, Docker available, no GPU. Julia is not preinstalled and `julialang.org` is not on the Trusted allowlist (`*.amazonaws.com` and GitHub release assets are, so juliaup binaries pass but `pkg.julialang.org` does not). A Custom network list with `julialang.org`, `*.julialang.org` and `pkg.julialang.org` fixes that. A setup script runs once per environment and its filesystem snapshot is cached only if it finishes in roughly five minutes, for about seven days; Docker images it pulls are kept in the cache.
- Projects (public beta, gradual rollout): one coordinating conversation starts a thread per task, each a cloud session on its own branch with a pull request and auto-fix, sharing project instructions and memory; threads get each repository's `CLAUDE.md` but not its skills, settings or MCP config; thread model and effort are project settings (default Opus at high effort); a thread can run on your own computer through Remote Control when it needs local hardware.

### First batch: CPU-only work packages in the cloud

Prerequisites: the public repository with the planning documents under `docs/plan/`, a `CLAUDE.md` carrying the decisions and conventions, `.claude/agents/` role definitions, and a cloud environment with the Custom network list and a setup script that installs juliaup plus Julia 1.13 and instantiates the core package's dependencies inside the five-minute cache window (KernelAbstractions and a handful of deps fit; the SciML stack for the reduced models does not, so either precompile it in a SessionStart hook per session or pull a prebuilt depot image with Docker in the setup script). Install the Claude GitHub App on the repository.

Then run these as parallel cloud sessions, or as Project threads where available:
1. Trace layer MVP: record schema, hash check, builder to `trace/graph.json`, session-record hook.
2. Core package skeleton on the KA CPU backend: `Grid{N}`, SoA species, Yee, Boris, Esirkepov, gather, generic `T`, unit and conservation tests.
3. 1D3V benchmark suite with expected numbers (Landau damping, two-stream, Weibel) plus the standard plot set and sidecars with CairoMakie.
4. Level-0 reduced model: 1D nonlinear wake plus test electron, SciMLSensitivity gradient, Optimization.jl, comparison against the Valenta analytic maximum-energy expressions. First answer to "is the gradient of electron energy informative".
5. The `plan-review` workflow on the planning documents, and `/code-review ultra` on the first pull requests, both of which run in cloud sandboxes.

Start with three or four sessions, check usage after the first day, and scale. Rate limits are shared across all sessions, so parallelism costs proportionally.

### API-billed work: the GPU side

- An Agent SDK script on the 4080 box under an API key, run on demand: validation and benchmark campaigns, FP32-versus-FP64 drift tests, cross-code comparisons against Smilei and PIConGPU, using the repository's role definitions and writing session records through hooks. This is the API-billed autonomous GPU loop; a truly unattended version waits for rented GPU VMs.
- Managed Agents with a self-hosted worker on the same box for a few well-specified GPU work packages, to try budget-capped, rubric-graded sessions.
- Cheap plumbing: Batch API grading of plot sets and sidecars at half price, API-side OpenTelemetry tracing.

API-billed sessions are sized for nightly loops, not for bulk implementation.

## 3d. The overseer: one Claude with the overall view (no GPU needed)

Projects is in gradual rollout (2026-10-30), so the coordinating view is built independently of it. Its job is to read, compare and report, never to compute:

- Inputs, all in the repository or published by it: the planning documents and decision records, the trace graph (or the records it is built from), open branches and pull requests with their CI results and test reports, the standard plots with their scalar sidecars, run records from GPU campaigns, agent session records with cost. GPU work produces artifacts that get committed or published (the EDM pattern: run manifests, plots, dashboard); the overseer reads artifacts, so it needs no GPU. If it concludes something must be re-run, it files a task for the GPU loop instead of running it.
- Checks: stale frontier after document edits; each branch or PR against its work package (interface implemented, tests tagged with requirement ids, plot set present, budgets met); drift between decisions and code (normalized units, no float literals, no hostnames, generic element type) via scripts; progress and cost per work package from session records; requirement-level questions that need a human.
- Outputs: a dated report in the repository (and the dashboard), issues or PR comments for concrete gaps, and a list of proposed next sessions as ready-to-paste `claude --cloud "..."` commands, which is the part of Projects worth replacing by hand.
- Tooling constraint: write the trace builder and the checks in Python or shell, or have CI emit `trace/graph.json`, so the overseer runs in any sandbox without installing Julia.

Two ways to host it, same prompt and tools:
1. **Claude Code cloud scheduled task** (now): daily cron plus GitHub-event triggers on pull requests, the same cloud environment as the implementation sessions, repository access through the GitHub App, memory = the records it commits. Zero plumbing; subscription usage.
2. **Managed Agents scheduled deployment** (later): a versioned agent with a GitHub repository resource and a vault credential for pushing, a memory store for what it flagged before, a hard budget per run, webhooks for notifications, Console cost per session. API-billed. The cloud sandbox (8 GB, 10 GB) is enough for reading and running Python checks.

Recommendation: start as a scheduled task, migrate the same role to a Managed Agents deployment once the GPU loop exists.

## 3e. Human interface: steering without becoming the bottleneck

Premise: the human reviewer cannot read every line, and frequent approval requests cause approval fatigue. Design goals: review artifacts rather than code, replace approvals with gates and budgets, batch the decisions that remain, and loosen gates as the track record justifies.

**Three levels of human attention**
- Requirements and decisions (rare, highest leverage): editing `R-` and `D-` records is the steering wheel; the trace layer propagates the change and the overseer lists what it affects.
- Acceptance evidence per work package: the standard plots with sidecars, conservation residuals, benchmark agreement, gradient checks, the two agent reviews and the overseer's verdict, all on one page. The human reads evidence, not diffs.
- Diffs: only for interface contracts (`I-` records) and anything that supports a scientific claim leaving the machine (independent review before a claim leaves the repository; see `CLAUDE.md`).

**Risk tiers decide what needs a human, assigned from file ownership (A14)**
- Tier 0 (docs, tests, plots, trace records, benchmark data) and tier 1 (implementation inside an existing interface): merge unattended once the required checks and the automated review pass. The review runs two reviewer lenses with distinct briefs (numerics, code) on every non-draft PR and approves or requests changes as `github-actions[bot]`; the PR author enables auto-merge and GitHub merges.
- Tier 2 (interface contracts, scheme changes, decision records, the contract and the roles, CI, anything a claim depends on, new dependencies): the paths in `.github/CODEOWNERS`; the owner's approval is required in addition, given from any device, in the daily batch.
- Enforcement is GitHub's, not the agents': the ruleset on `main` (pull requests only, squash, one approval, code owners, required checks, no bypass for agents). Local factory sessions act as the GitHub App `particlemesh-factory[bot]`, so the owner's comments and reviews are distinguishable from agents'; cloud sessions act as the owner's account.

**No blocking questions.** Agents never stop to ask. They state an assumption in the PR description and in `questions/<id>.md` with the options and the default they took, and continue. The overseer folds open questions into the digest; answering one is editing the file or a one-line reply, which becomes a decision record. Workflow agents cannot ask anyway, cloud sessions run in auto mode, and local unattended runs use allow rules per role in the project settings.

**Three inboxes instead of outbound issues.** Agents may not open issues on, or comment in, other repositories; the factory's GitHub App is installed on this repository only, and the cloud proxy serves only the attached repository. A suspected bug in a dependency goes to a fresh `upstream-reporter` subagent that reduces it to a minimal working example and posts one comment on the pinned issue "Potential upstream issues" (label `upstream-triage`). Problems with the setup or the process go as one comment per problem on the pinned issue "Factory issues" (label `factory`). Dependency needs go as one comment on the pinned issue "Package requests" (label `pkg-request`), answered by the `package-manager` role with a PR (A12). Humans and sessions outside the factory triage both: confirm and file upstream, or change the rule, role or machine setup, and reply with what changed. The overseer's digest lists new comments on all three.

**Surfaces, in order of frequency**
- A daily digest from the overseer, 15 to 20 minutes to read: what landed, what is waiting (tier-2 items and open questions, each with a default), what the nightly GPU loop found, cost and progress per work package, proposed next sessions as ready-to-paste commands. Delivered to the dashboard and, optionally, exported as a PDF for annotation.
- The dashboard: trace graph with the stale frontier, work-package status, the plot sets, review precision and defect statistics.
- One steering session: a local Claude Code session (reachable remotely through Remote Control) where directions are given in natural language and turned into decision records, questions answered, sessions launched.
- PR comments for the rare line-level remark; tier-2 approvals from the phone as they come; a weekly hour for requirements and decisions.

**Trust calibration.** The dashboard tracks agent-review precision (findings confirmed versus refuted), defects found after merge by CI or the nightly loop, and reverted PRs, per tier. Tiers loosen on evidence (a package type moves from tier 1 to tier 0 after N clean merges) and tighten on a defect. Start strict.

**Brakes.** One switch pauses everything: scheduled tasks paused, workflows stopped, Managed Agents budgets set to the consumed amount. Budgets per session and per work package cap the damage of a runaway loop without anyone watching.

## 4. Proposed factory shape (to discuss)

1. The documents in `docs/plan/` become the `trace/` seed: requirements, decisions, interfaces, work packages with rubrics.
2. Each work package runs as one budgeted session: cloud environment for CPU-only packages, self-hosted sandbox on a GPU machine for the rest, started with the package's rubric as the outcome. The session record is written automatically.
3. A scheduled job rebuilds the trace graph, runs the CPU test suite, and opens re-validation tasks for stale nodes.
4. Human gates: requirement edits, interface contracts, merges, and anything that makes a scientific claim.
5. Cost and progress are read from the same graph: tokens and dollars per work package next to test status.

### The split (`overview.md` §3)

- **Bulk implementation: Claude Code saved workflows run on the GPU machine.** One workflow per work-package type (implement → test → plot-inspect → review → report), stored in `.claude/workflows/`, started either from an interactive lead session or unattended with `claude -p "/<workflow> WP-xx"` under explicit allow rules, on the local 4080 SUPER box with a warm Julia depot and the GPU (rented GPU VMs are the scale-up). A primer and a first read-only workflow are in `workflows-primer.md`. This spends subscription usage, keeps Julia compile times out of cloud sandboxes, and gives the trace layer `workflow.run_id` and `agent_id` for free through telemetry.
- **Roles as subagent definitions** in `.claude/agents/`: implementer (Opus, xhigh, edit and run tools), numerics reviewer (no edit tools, reads plots and sidecars), plot inspector (read-only plus the plot scripts), trace maintainer (Haiku or Sonnet, low effort), each with a pinned tool allowlist for reproducibility. Two service roles complete the roster: `upstream-reporter` (minimal examples for the upstream inbox) and `package-manager` (the only editor of dependency files, A12).
- **Loops**: desktop scheduled tasks or the Agent SDK script on the 4080 for validation and benchmark campaigns; Claude Code scheduled tasks in the cloud for the overseer and CPU-light chores such as trace rebuilds and documentation checks.
- **Managed Agents as a bounded second track**, not the backbone: rubric-graded outcomes and hard session budgets for a few well-specified work packages, and cron deployments with a self-hosted worker on a rented GPU VM for unattended GPU validation. It is the only surface with dollar caps and per-session cost records, so it is also the cheapest way to measure what a work package costs. Prerequisites: a worker container with Julia, the depot and GPU passthrough.
- **Observability pipeline**: OpenTelemetry metrics, logs and beta traces from Claude Code and, if used, API tracing from Managed Agents, into one OTLP backend; hooks (`SubagentStop`, `TaskCompleted`, `SessionEnd`) POST a session record into `trace/` so the graph has cost, commits and WP links even without the backend; the dashboard reads `trace/graph.json` plus the standard plots.
- **Human gates** unchanged: requirement edits, interface contracts, merges, scientific claims.

Open questions: whether to keep the Managed Agents track at all for a single-maintainer project (cost and plumbing versus the budget and outcome features), the OTLP backend (a self-hosted Grafana stack versus a hosted one), and how much of the trace builder to write before the first work package starts (recommendation: the schema, the hash check and the session-record hook; the graph view after the first packages exist).
