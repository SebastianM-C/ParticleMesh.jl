# Saved Claude Code workflows: a primer for ParticleMesh

Source: the Claude Code workflow docs (https://code.claude.com/docs/en/workflows.md) and the bundled authoring reference. Purpose: enough to read, write and run the workflows the factory will use.

## What a workflow is

A workflow is a short JavaScript script that orchestrates subagents. The script owns the control flow (fan-out, pipelines, loops, conditionals); the agents do the work (read, edit, run, judge). Intermediate results live in script variables, not in any model's context, so the orchestration costs no relay tokens and the same script runs the same way again. Runs happen in the background, show up in `/workflows` with phases and per-agent progress, and can be paused, stopped, resumed and saved from there.

Contrast with the two other delegation mechanisms: a subagent spawned from a session is model-driven delegation whose report lands back in the parent's context; several named sessions coordinate by messages. A workflow is code-driven, resumable and repeatable.

## Mechanics

- **Script shape.** The file starts with `export const meta = {name, description, phases: [...]}` as a pure literal, then the body. The body is plain JavaScript (not TypeScript) in an async context: `await` directly, standard built-ins available, no filesystem, no Node APIs, no `Date.now()` or `Math.random()` (they would break resume; pass timestamps through `args`).
- **Primitives.** `agent(prompt, opts)` spawns one subagent and returns its final text, or, with `opts.schema` (a JSON Schema), a validated object. `pipeline(items, stage1, stage2, ...)` runs every item through the stages independently with no barrier between stages; this is the default for multi-stage work. `parallel([thunks])` runs tasks concurrently and waits for all of them; use it only when a stage needs all previous results together (dedup, early exit, cross-item comparison). `phase(title)` groups the following agents in the progress display; `log(msg)` emits a narrator line; `args` carries what the caller passed; `workflow(name, args)` runs another saved workflow one level deep; `budget` exposes a token ceiling when the user set one.
- **Per-agent options.** `label`, `phase`, `schema`, `model` (omit to inherit the session model), `effort` (`low` for mechanical stages, higher for judging), `isolation: 'worktree'` (only when agents edit files in parallel), `agentType` (a named definition from `.claude/agents/`, which is how the factory's pinned roles enter a workflow).
- **Failure semantics.** An agent that is skipped or dies returns `null`; a stage that throws drops that item to `null` and skips its later stages; `parallel` never rejects. Filter with `.filter(Boolean)`.
- **Limits.** About 16 concurrent agents by default (queue beyond that), 1,000 agents per run, 4,096 items per pipeline or parallel call. Runs count against subscription usage; there is no dollar cap, so size the fan-out deliberately and `log()` anything the script truncates.
- **Running.** Type `ultracode` in a prompt (or `/effort ultracode` for the session) and describe the task, or ask for a workflow in your own words; Claude authors the script and runs it. Saved workflows run as `/<name> <args>`; a saved workflow can also run unattended with `claude -p "/<name> ..."` when `Workflow(<name>)` and the agents' tools are allowed by permission rules.
- **Observing.** `/workflows` lists runs; select one for phases and agents; keys: `p` pause or resume, `x` stop, `r` restart an agent, `s` save the script, `f` filter. Subagent transcripts are under `~/.claude/projects/<project>/<session>/subagents/`. With telemetry on, events carry `workflow.run_id` and `workflow.name`.
- **Saving and editing.** Press `s` in `/workflows` to save to `.claude/workflows/` (project, shared through git) or `~/.claude/workflows/` (personal). Edit the file, then `/reload-skills`. Every run also persists its script under the session directory; iterate by editing that file and relaunching with its path.
- **Resuming.** Relaunch the same script with the previous run id: unchanged agent calls return cached results instantly, the first changed call and everything after it runs live. The run's `journal.jsonl` records every agent's actual return value; read it before concluding a run returned nothing.

## Quality patterns the reference recommends

Adversarial verification (several independent skeptics per finding, keep what survives a majority), perspective-diverse verifiers (distinct lenses rather than identical refuters), judge panels over independent attempts, loop-until-dry for discovery of unknown size, a completeness critic at the end, and never a silent cap.

## A first workflow for this project: review the plan

Read-only, useful now, and it exercises every primitive. Five reviewers read the planning documents through different lenses, each finding is adversarially checked by two refuters, and the survivors come back as a structured list. Arguments: the document paths and a date stamp.

The script is `.claude/workflows/plan-review.js`.

How it is used: run `/plan-review` with the plan documents as `args` after each substantial edit. The same skeleton, with an implementer stage in a worktree and a test-runner stage instead of the reviewers, becomes the work-package workflow.

## What the factory's workflows would look like

- `plan-review` (above): read-only review of the planning documents.
- `plan-packages`, two modes: `cut` takes the next work packages from a named slice of the goals and the plan, `rederive` re-derives the packages a goal edit made stale. The stages, the models per agent and the caller's steps are in `.claude/workflows/plan-packages.js`. In `cut` the verifiers read the whole set rather than one package each, because the defects that matter sit between packages. Its PR merges on the automated review, and the digest lists the packages for the owner to veto.
- `wp-implement`: for one work package, an implementer agent in a worktree (role from `.claude/agents/`), then a test-runner stage, then a plot-inspector stage reading the standard plots and their sidecars, then two reviewers with distinct lenses, then a report; the script writes nothing itself, the agents commit on a branch.
- `cross-code-validate`: run the matched case in ParticleMesh, Smilei and PIConGPU on the local GPU, produce the overlay plots, judge agreement against the tolerances of A7.

Each is a saved file in the repository, versioned with the code; its cost and usage come from telemetry.
