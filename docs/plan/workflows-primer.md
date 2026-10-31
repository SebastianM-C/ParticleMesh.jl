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

```js
export const meta = {
  name: 'plan-review',
  description: 'Review the ParticleMesh planning documents through five lenses and verify each finding',
  whenToUse: 'Before handing the plan to the implementation team, or after a substantial edit of the docs',
  phases: [
    { title: 'Review', detail: 'one reviewer per lens over all documents' },
    { title: 'Verify', detail: 'two refuters per finding' },
    { title: 'Synthesize', detail: 'ranked list of surviving findings' },
  ],
}

const DOCS = args.docs            // e.g. ["landscape-and-requirements.md", "architecture-draft.md", ...]
const DATE = args.date            // stamped by the caller; Date.now() is unavailable here

const LENSES = [
  { key: 'physics',   prompt: 'LWFA physics: wrong or missing scalings, regimes, injection and objective assumptions' },
  { key: 'numerics',  prompt: 'PIC numerics: scheme choices, conservation, resolution, boundary and window handling, FP32 risks' },
  { key: 'ad',        prompt: 'differentiability: where gradients are undefined, noisy or mis-specified; tangent layout; atomics' },
  { key: 'software',  prompt: 'software and infrastructure: package layout, GPU portability, dependencies, CI, reproducibility' },
  { key: 'process',   prompt: 'process: requirements that are untestable, decisions without rationale, work that is unowned or unsequenced' },
]

const FINDINGS = {
  type: 'object',
  properties: {
    findings: { type: 'array', items: {
      type: 'object',
      properties: {
        doc: { type: 'string' }, section: { type: 'string' },
        claim: { type: 'string' }, why: { type: 'string' },
        severity: { type: 'string', enum: ['blocking', 'major', 'minor'] },
      },
      required: ['doc', 'section', 'claim', 'why', 'severity'],
    }},
  },
  required: ['findings'],
}

const VERDICT = {
  type: 'object',
  properties: { refuted: { type: 'boolean' }, reason: { type: 'string' } },
  required: ['refuted', 'reason'],
}

// Review → Verify as a pipeline: a lens whose review finishes early has its findings verified
// while slower lenses are still reading. No barrier until the synthesis needs everything.
const verified = await pipeline(
  LENSES,
  lens => agent(
    `Read these planning documents in ${args.dir}: ${DOCS.join(', ')}. ` +
    `Review them through this lens only: ${lens.prompt}. ` +
    `Report concrete findings with the document and section they refer to. Do not edit anything.`,
    { label: `review:${lens.key}`, phase: 'Review', schema: FINDINGS }),
  (review, lens) => review ? parallel(review.findings.map(f => () =>
    parallel([0, 1].map(i => () => agent(
      `A reviewer claims, about ${f.doc} (${f.section}): "${f.claim}" because "${f.why}". ` +
      `Read that document in ${args.dir} and try to refute the claim. Default to refuted=true if uncertain.`,
      { label: `verify:${lens.key}:${i}`, phase: 'Verify', schema: VERDICT })))
      .then(votes => ({ ...f, lens: lens.key,
                        survives: votes.filter(Boolean).filter(v => !v.refuted).length === 2 })))) : [],
)

const survivors = verified.filter(Boolean).flat().filter(f => f.survives)
log(`${survivors.length} findings survived verification`)

phase('Synthesize')
const report = await agent(
  `Here are verified review findings on the ParticleMesh plan (${DATE}): ${JSON.stringify(survivors)}. ` +
  `Rank them by severity and by how early they bite, merge duplicates, and write a short markdown report ` +
  `with one line per finding: document, section, claim, suggested change.`,
  { label: 'synthesize', effort: 'high' })

return { date: DATE, survivors, report }
```

How it is used: run it once with the plan documents as `args`, save it with `s`, and from then on run `/plan-review` after each substantial edit. The same skeleton, with an implementer stage in a worktree and a test-runner stage instead of the reviewers, becomes the work-package workflow.

## What the factory's workflows would look like

- `plan-review` (above): read-only review of the planning documents.
- `plan-packages`: cuts the next work packages from a named slice of the plan. Three planners (interface-first, validation-first, risk-first) propose, one judge merges and orders, two refuters per package try to reject it for scope and for plan compliance, and a writer saves the survivors as records under `trace/wp/`. The caller passes the slice, the first free package number and the date, reads the files, and opens the PR; the owner approves it, since `trace/wp/` is a code-owner path.
- `wp-implement`: for one work package, an implementer agent in a worktree (role from `.claude/agents/`), then a test-runner stage, then a plot-inspector stage reading the standard plots and their sidecars, then two reviewers with distinct lenses, then a report; the script writes nothing itself, the agents commit on a branch.
- `trace-rebuild`: regenerate the trace graph, list stale nodes, open one re-validation task per stale node.
- `cross-code-validate`: run the matched case in ParticleMesh, Smilei and PIConGPU on the local GPU, produce the overlay plots, judge agreement against the tolerances in the requirement.

Each is a saved file in the repository, versioned with the code, and each run is a session record in the trace layer.
