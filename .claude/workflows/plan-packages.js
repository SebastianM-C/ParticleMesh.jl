export const meta = {
  name: 'plan-packages',
  description: 'Cut the next work packages from the ParticleMesh goals and plan, or re-derive the packages a goal edit made stale',
  whenToUse: 'mode "cut" when no dispatchable package is left; mode "rederive" after a goal in docs/goals/ changed (scripts/trace/build.jl lists the stale packages). Its PR merges on the automated review; the digest lists the packages for the owner to veto',
  phases: [
    { title: 'Propose', detail: 'cut: three planners, each from a different angle' },
    { title: 'Merge', detail: 'cut: one judge merges the proposals into an ordered list' },
    { title: 'Verify', detail: 'cut: two refuters per package, distinct lenses' },
    { title: 'Write', detail: 'cut: surviving packages as files under trace/wp/' },
    { title: 'Rederive', detail: 'rederive: one agent per stale package, then one refuter' },
  ],
}

// Inputs come from args; Date.now() and the filesystem are unavailable in a workflow script.
//   mode      "cut" (default) or "rederive"
//   date      stamped by the caller
// cut:
//   scope     one sentence naming the slice to cut, e.g. "the engine core on the CPU backend"
//   next_id   the first free package number; one past the highest in trace/wp
//   max       upper bound on packages per run (default 6)
// rederive:
//   stale     [{id, file, goals: ["A2", ...]}], from the stale list scripts/trace/build.jl prints
// The caller then runs `scripts/trace/build.jl --refresh <ids> --check`, commits and opens the PR.
const MODE = args.mode || 'cut'
const DATE = args.date
const MAX = args.max || 6

const CONTEXT =
  `The owner's aims and binding decisions are docs/goals/A0 to A10 (one file each); they are the root and ` +
  `must not be contradicted or edited. Derived plan text is in docs/plan/ (start with overview.md; interfaces ` +
  `in architecture-draft.md section 0; thin spots in open-items.md B; factory settings F1 to F4 in ` +
  `factory-and-observability.md section 5). Work packages are trace/wp/WP-<nnn>-<slug>.md with TOML front ` +
  `matter as trace/README.md specifies; the existing ones show structure and tone. A package is one pull ` +
  `request by one session: it implements one interface or one validation step, names the goals it serves, ` +
  `the tests that prove it (with trace tags such as [A2, WP-004]), the budgets of A7 where they apply, the ` +
  `Julia packages it adds, what it must not do, and the packages it depends on. The rules in CLAUDE.md hold.`

const PACKAGE = {
  type: 'object',
  properties: {
    title: { type: 'string' }, slug: { type: 'string' }, scope: { type: 'string' },
    goals: { type: 'array', items: { type: 'string' } },
    deliverables: { type: 'array', items: { type: 'string' } },
    non_goals: { type: 'array', items: { type: 'string' } },
    acceptance: { type: 'array', items: { type: 'string' } },
    depends_on: { type: 'array', items: { type: 'string' } },
    adds_dependencies: { type: 'array', items: { type: 'string' } },
    budgets: { type: 'string' },
    rationale: { type: 'string' },
  },
  required: ['title', 'slug', 'scope', 'goals', 'deliverables', 'non_goals', 'acceptance', 'depends_on', 'adds_dependencies', 'budgets', 'rationale'],
}
const PROPOSAL = { type: 'object', properties: { packages: { type: 'array', items: PACKAGE } }, required: ['packages'] }
const VERDICT = {
  type: 'object',
  properties: { refuted: { type: 'boolean' }, reason: { type: 'string' }, fixes: { type: 'array', items: { type: 'string' } } },
  required: ['refuted', 'reason', 'fixes'],
}
const WRITTEN = { type: 'object', properties: { files: { type: 'array', items: { type: 'string' } } }, required: ['files'] }
const REDERIVED = {
  type: 'object',
  properties: {
    id: { type: 'string' },
    action: { type: 'string', enum: ['confirm', 'edit', 'split', 'withdraw'] },
    summary: { type: 'string' },
    files: { type: 'array', items: { type: 'string' } },
  },
  required: ['id', 'action', 'summary', 'files'],
}

async function cut() {
  const SCOPE = args.scope
  const NEXT = args.next_id
  if (!SCOPE || !NEXT) throw new Error('plan-packages cut needs args.scope and args.next_id')
  const ctx = `${CONTEXT} Slice to cut: ${SCOPE}. Do not edit anything.`
  const ANGLES = [
    { key: 'interface',  prompt: 'interface-first: each package implements one interface contract of architecture-draft.md section 0 and nothing else' },
    { key: 'validation', prompt: 'validation-first: each package is defined by the test or benchmark that proves it, from the goals and the budgets of A7' },
    { key: 'risk',       prompt: 'risk-first: the packages that retire the thin spots B and the spikes (Reactant, symbolic, FP32) earliest, smallest first' },
  ]

  // Propose: a barrier is right here, the merge needs all three proposals at once.
  const proposals = (await parallel(ANGLES.map(a => () => agent(
    `${ctx} Propose at most ${MAX} packages from this angle: ${a.prompt}. ` +
    `Skip anything an existing package already covers. Keep each package small enough for one session.`,
    { label: `propose:${a.key}`, phase: 'Propose', schema: PROPOSAL })))).filter(Boolean)
  log(`${proposals.length} proposals, ${proposals.reduce((n, p) => n + p.packages.length, 0)} candidate packages`)

  const merged = await agent(
    `${ctx} Three planners proposed packages from different angles: ${JSON.stringify(proposals)}. ` +
    `Merge them into one list of at most ${MAX} packages, ordered so that every dependency comes before its ` +
    `dependents and packages that add Julia dependencies do not run at the same time as each other; merge ` +
    `duplicates, drop anything outside the slice, keep every package's structure complete.`,
    { label: 'merge', phase: 'Merge', schema: PROPOSAL, effort: 'high' })
  const candidates = merged ? merged.packages : []
  const titles = candidates.map(p => p.title)

  // Verify: each package through two lenses, as a pipeline so early packages are judged while later ones wait.
  const LENSES = [
    { key: 'fit',  prompt: 'scope: it overlaps an existing package or another proposed package, exceeds one session, or has no clear interface' },
    { key: 'plan', prompt: 'goal compliance: it contradicts a goal in docs/goals/ or a rule of CLAUDE.md, names no goal it serves, its acceptance is not testable, a dependency is missing or circular, or budgets are absent where A7 applies' },
  ]
  const verified = await pipeline(
    candidates,
    pkg => parallel(LENSES.map(l => () => agent(
      `${ctx} A planner proposes this package: ${JSON.stringify(pkg)}. The other proposed packages are: ` +
      `${JSON.stringify(titles)}. Try to refute it through this lens only: ${l.prompt}. ` +
      `Default to refuted=true if uncertain; list concrete fixes when a small change would save it.`,
      { label: `verify:${l.key}:${pkg.slug}`, phase: 'Verify', schema: VERDICT })))
      .then(votes => {
        const v = votes.filter(Boolean)
        // Fail closed: a package survives only when both refuters answered and neither refuted it.
        return { ...pkg, votes: v, survives: v.length === LENSES.length && v.every(x => !x.refuted) }
      }),
  )

  const survivors = verified.filter(Boolean).filter(p => p.survives)
  const dropped = verified.filter(Boolean).filter(p => !p.survives)
    .map(p => ({ title: p.title, reasons: p.votes.length < LENSES.length
      ? ['a verifier returned no verdict'].concat(p.votes.filter(v => v.refuted).map(v => v.reason))
      : p.votes.filter(v => v.refuted).map(v => v.reason) }))
  log(`${survivors.length} packages survived, ${dropped.length} dropped`)

  phase('Write')
  const written = survivors.length ? await agent(
    `Write these work packages as files in trace/wp/, one file each, named WP-<nnn>-<slug>.md and numbered ` +
    `from ${NEXT} in this order: ${JSON.stringify(survivors.map(p => ({ ...p, votes: undefined, survives: undefined })))}. ` +
    `Apply the fixes the verifiers suggested: ${JSON.stringify(survivors.map(p => ({ slug: p.slug, fixes: p.votes.flatMap(v => v.fixes) })))}. ` +
    `Each file starts with the TOML front matter of trace/README.md (id, status "proposed", goals, depends_on ` +
    `with the ids you assign, adds_dependencies; leave out [checked], the caller fills it), then the sections ` +
    `the existing packages use (Scope, Deliverables, Non-goals, Acceptance, Depends on, Budgets). ` +
    `Write nothing else and do not commit.`,
    { label: 'write', phase: 'Write', schema: WRITTEN, effort: 'high' }) : { files: [] }

  return { mode: 'cut', date: DATE, scope: SCOPE, files: written ? written.files : [], dropped }
}

async function rederive() {
  const STALE = args.stale
  if (!Array.isArray(STALE) || STALE.length === 0) throw new Error('plan-packages rederive needs args.stale')
  phase('Rederive')
  const results = await pipeline(
    STALE,
    s => agent(
      `${CONTEXT} The goals ${s.goals.join(', ')} changed since work package ${s.id} (trace/wp/${s.file}) was ` +
      `checked against them. For each goal, the front matter's [checked] entry is the git blob id of the old ` +
      `version: see the change with \`git diff <old blob id> <new blob id>\`, the new one from ` +
      `\`git hash-object docs/goals/<file>\`. Decide whether the package still holds: confirm it unchanged, ` +
      `edit it, split it (new packages numbered one past the highest in trace/wp/), or withdraw it ` +
      `(status "withdrawn", one line saying why). Make the edits in trace/wp/, and update any text in ` +
      `docs/plan/ that cites the changed goals and no longer matches them. Never edit docs/goals/. ` +
      `Do not touch [checked] and do not commit. Report the files you changed.`,
      { label: `rederive:${s.id}`, phase: 'Rederive', schema: REDERIVED, effort: 'high' }),
    (r, s) => r ? agent(
      `${CONTEXT} After the goals ${s.goals.join(', ')} changed, a planner re-derived work package ${s.id}: ` +
      `${JSON.stringify(r)}. Read the changed goals (git diff in the working tree) and the files it changed, ` +
      `and try to refute the result: the package or the edited plan text still contradicts a changed goal, or ` +
      `the change goes beyond what the goal edit requires. Default to refuted=true if uncertain.`,
      { label: `refute:${s.id}`, phase: 'Rederive', schema: VERDICT })
      .then(v => ({ ...r, verdict: v, holds: !!v && !v.refuted })) : null,
  )
  const done = results.filter(Boolean)
  const missing = STALE.filter(s => !done.some(r => r.id === s.id)).map(s => s.id)
  return {
    mode: 'rederive', date: DATE,
    refresh: done.filter(r => r.holds && r.action !== 'withdraw').map(r => r.id),
    results: done,
    unresolved: missing.concat(done.filter(r => !r.holds).map(r => r.id)),
  }
}

return MODE === 'rederive' ? await rederive() : await cut()
