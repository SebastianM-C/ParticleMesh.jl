export const meta = {
  name: 'plan-packages',
  description: 'Cut the next work packages from the ParticleMesh goals and plan, or re-derive the packages a goal edit made stale',
  whenToUse: 'mode "cut" when no dispatchable package is left; mode "rederive" after a goal in docs/goals/ changed (scripts/trace/build.jl lists the stale packages). Its PR merges on the automated review; the digest lists the packages for the owner to veto',
  phases: [
    { title: 'Readiness', detail: 'cut: one agent checks the plan settles what the slice needs; stops with the gaps if not' },
    { title: 'Propose', detail: 'cut: two planners, interface-first and validation-first' },
    { title: 'Merge', detail: 'cut: one judge merges the proposals into an ordered list' },
    { title: 'Verify', detail: 'cut: two verifiers over the whole set, ownership and compliance, findings graded blocking or fixable' },
    { title: 'Write', detail: 'cut: one agent applies the fixable findings, drops only blocked packages, writes trace/wp/' },
    { title: 'Rederive', detail: 'rederive: one agent per stale package, then one refuter' },
  ],
}

// Inputs come from args; Date.now() and the filesystem are unavailable in a workflow script.
// Pass args as a JSON object, not a string.
//   mode      "cut" (default) or "rederive"
//   date      stamped by the caller
// cut:
//   scope     one sentence naming the slice to cut, e.g. "the engine core on the CPU backend"
//   next_id   the first free package number; one past the highest ever used, deleted packages
//             included (trace/README.md)
//   max       upper bound on packages per run (default 6)
//   skip_readiness  true to skip the readiness check, after the caller fixed the gaps it reported
//   gaps      with skip_readiness: the non-blocking gaps of an earlier readiness check, passed on
//             to the planners as resolutions to follow (same shape as the check returns)
// rederive:
//   stale     [{id, file, goals: ["A2", ...]}], from the stale list scripts/trace/build.jl prints.
//             Pass only the packages worth a re-derive: leave out parked ones that the next cut
//             replaces anyway (refresh their [checked] entries by hand after reading the goal diff),
//             and done ones unless their text now contradicts a goal. Each entry costs two agents.
// Models: planners, the judge and the writer run on Opus; readiness, verifiers, re-derives and
// refuters on Sonnet, all at high effort. Without these the agents inherit the caller's model, which
// made one re-derive of eight packages cost sixteen agents on the most expensive tier.
// The caller then runs `scripts/trace/build.jl --refresh <ids> --check`, commits and opens the PR.
// A cut that returns ready=false wrote nothing: fix the reported gaps in docs/plan/ (or write
// questions/ for the ones that need the owner) and run it again.
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
  `the tests that prove it (with trace tags such as [A2, WP-009]), the budgets of A7 where they apply, the ` +
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
const READINESS = {
  type: 'object',
  properties: {
    ready: { type: 'boolean' },
    gaps: { type: 'array', items: { type: 'object', properties: {
      topic: { type: 'string' }, where: { type: 'string' }, resolution: { type: 'string' }, blocking: { type: 'boolean' },
    }, required: ['topic', 'where', 'resolution', 'blocking'] } },
  },
  required: ['ready', 'gaps'],
}
const FINDINGS = {
  type: 'object',
  properties: { findings: { type: 'array', items: { type: 'object', properties: {
    slug: { type: 'string' }, issue: { type: 'string' }, fix: { type: 'string' },
    severity: { type: 'string', enum: ['fixable', 'blocking'] },
  }, required: ['slug', 'issue', 'fix', 'severity'] } } },
  required: ['findings'],
}
const VERDICT = {
  type: 'object',
  properties: { refuted: { type: 'boolean' }, reason: { type: 'string' }, fixes: { type: 'array', items: { type: 'string' } } },
  required: ['refuted', 'reason', 'fixes'],
}
const WRITTEN = {
  type: 'object',
  properties: {
    files: { type: 'array', items: { type: 'string' } },
    applied: { type: 'array', items: { type: 'object', properties: {
      slug: { type: 'string' }, fixes: { type: 'array', items: { type: 'string' } },
    }, required: ['slug', 'fixes'] } },
  },
  required: ['files', 'applied'],
}
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
  if (!SCOPE || !NEXT) throw new Error('plan-packages cut needs args.scope and args.next_id (args is a JSON object)')
  const ctx = `${CONTEXT} Slice to cut: ${SCOPE}. Do not edit anything.`
  const empty = extra => ({ mode: 'cut', date: DATE, scope: SCOPE, files: [], applied: [], findings: [], dropped: [], ...extra })

  // Readiness: where the plan leaves an interface, a layout or an owner of the slice open, planners
  // invent incompatible answers and every package fails on the same seams. One agent finds that first.
  phase('Readiness')
  if (args.skip_readiness && args.gaps !== undefined && !Array.isArray(args.gaps))
    throw new Error('plan-packages: args.gaps must be an array of gaps, not a JSON-encoded string')
  const ready = args.skip_readiness ? { ready: !(args.gaps || []).some(g => g.blocking), gaps: args.gaps || [] } : await agent(
    `${ctx} Before anyone cuts packages: list what the slice needs the plan to settle (the owner of each ` +
    `interface, data layout, configuration, boundary and communication point it touches, and the open ` +
    `items B it depends on), and check whether docs/plan/ settles each one without contradicting itself or ` +
    `a goal. Report each gap: the topic, where the plan is silent or inconsistent, and a proposed ` +
    `resolution. A gap is blocking when two reasonable planners would cut incompatible packages because of ` +
    `it. Set ready=false if any gap is blocking.`,
    { label: 'readiness', phase: 'Readiness', schema: READINESS, model: 'sonnet', effort: 'high' })
  if (!ready) throw new Error('the readiness check returned nothing')
  if (!ready.ready) {
    log(`plan not ready for this slice: ${ready.gaps.filter(g => g.blocking).length} blocking gaps`)
    return empty({ ready: false, gaps: ready.gaps })
  }
  const gapNote = ready.gaps.length
    ? ` Non-blocking plan gaps; resolve them as proposed: ${JSON.stringify(ready.gaps)}.` : ''

  const ANGLES = [
    { key: 'interface',  prompt: 'interface-first: each package implements one interface contract of architecture-draft.md section 0 and nothing else' },
    { key: 'validation', prompt: 'validation-first: each package is defined by the test that proves it, preferring analytic solutions and conservation laws held close to roundoff (A7)' },
  ]
  // Propose: a barrier is right here, the merge needs both proposals at once.
  const proposals = (await parallel(ANGLES.map(a => () => agent(
    `${ctx}${gapNote} Propose at most ${MAX} packages from this angle: ${a.prompt}. Order by risk: packages ` +
    `that retire thin spots and spikes come early. Skip anything an existing package already covers. Keep ` +
    `each package small enough for one session, and give every interface it touches exactly one owning package.`,
    { label: `propose:${a.key}`, phase: 'Propose', schema: PROPOSAL, model: 'opus', effort: 'high' })))).filter(Boolean)
  log(`${proposals.length} proposals, ${proposals.reduce((n, p) => n + p.packages.length, 0)} candidate packages`)

  const merged = await agent(
    `${ctx}${gapNote} Two planners proposed packages from different angles: ${JSON.stringify(proposals)}. ` +
    `Merge them into one list of at most ${MAX} packages, ordered so that every dependency comes before its ` +
    `dependents and packages that add Julia dependencies do not run at the same time as each other; merge ` +
    `duplicates, drop anything outside the slice, give each shared interface one owner, and keep every ` +
    `package's structure complete.`,
    { label: 'merge', phase: 'Merge', schema: PROPOSAL, model: 'opus', effort: 'high' })
  const candidates = merged ? merged.packages : []
  if (!candidates.length) return empty({ ready: true, gaps: ready.gaps })

  // Verify: two lenses over the whole set, because the defects that matter sit between packages.
  // Findings are graded; only a blocking one drops a package, fixable ones are applied when writing.
  const LENSES = [
    { key: 'ownership',  prompt: 'scope and ownership: two packages own the same interface or none does, a package exceeds one session or has no clear interface, a dependency is missing, circular or out of order' },
    { key: 'compliance', prompt: 'goal compliance: a package contradicts a goal in docs/goals/ or a rule of CLAUDE.md, names no goal it serves or misses one it relies on, its acceptance is not testable, or A7 budgets are absent where they apply' },
  ]
  const verdicts = (await parallel(LENSES.map(l => () => agent(
    `${ctx} The proposed packages, in order: ${JSON.stringify(candidates)}. Check the whole set through ` +
    `this lens only: ${l.prompt}. For each defect give the package slug, the issue, a concrete fix and a ` +
    `severity: "fixable" when a change within this package set resolves it, "blocking" only when it cannot ` +
    `be resolved without an owner decision or a plan change outside the set. Leave out anything you are not ` +
    `sure is a defect.`,
    { label: `verify:${l.key}`, phase: 'Verify', schema: FINDINGS, model: 'sonnet', effort: 'high' })))).filter(Boolean)
  if (verdicts.length < LENSES.length) throw new Error('a verifier returned no verdict')
  const findings = verdicts.flatMap(v => v.findings)
  const blocked = new Set(findings.filter(f => f.severity === 'blocking').map(f => f.slug))
  const keep = candidates.filter(p => !blocked.has(p.slug))
  const dropped = candidates.filter(p => blocked.has(p.slug)).map(p => ({
    title: p.title,
    reasons: findings.filter(f => f.slug === p.slug && f.severity === 'blocking').map(f => f.issue),
  }))
  log(`${findings.length} findings, ${dropped.length} packages blocked, ${keep.length} to write`)

  phase('Write')
  const written = keep.length ? await agent(
    `Write these work packages as files in trace/wp/, one file each, named WP-<nnn>-<slug>.md and numbered ` +
    `from ${NEXT} in this order: ${JSON.stringify(keep)}. Apply every fixable finding of the verifiers: ` +
    `${JSON.stringify(findings.filter(f => f.severity === 'fixable' && !blocked.has(f.slug)))}. These packages ` +
    `were dropped as blocked, and no written package may depend on them: ${JSON.stringify(dropped.map(d => d.title))}. ` +
    `Each file starts with the TOML front matter of trace/README.md (id, status "proposed", goals, depends_on ` +
    `with the ids you assign, adds_dependencies; leave out [checked], the caller fills it), then the sections ` +
    `the existing packages use (Scope, Deliverables, Non-goals, Acceptance, Depends on, Budgets). ` +
    `Write nothing else and do not commit. Report each file and, per package, the fixes you applied.`,
    { label: 'write', phase: 'Write', schema: WRITTEN, model: 'opus', effort: 'high' }) : { files: [], applied: [] }

  return {
    mode: 'cut', date: DATE, scope: SCOPE, ready: true, gaps: ready.gaps,
    files: written ? written.files : [], applied: written ? written.applied : [], findings, dropped,
  }
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
      `edit it, split it (new packages numbered one past the highest id ever used, deleted ones included, ` +
      `trace/README.md), or withdraw it (delete its file and say why in the summary, for the commit ` +
      `message). Make the edits in trace/wp/, and update any text in ` +
      `docs/plan/ that cites the changed goals and no longer matches them. Never edit docs/goals/. ` +
      `Do not touch [checked] and do not commit. Report the files you changed.`,
      { label: `rederive:${s.id}`, phase: 'Rederive', schema: REDERIVED, model: 'sonnet', effort: 'high' }),
    (r, s) => r ? agent(
      `${CONTEXT} After the goals ${s.goals.join(', ')} changed, a planner re-derived work package ${s.id}: ` +
      `${JSON.stringify(r)}. Read the changed goals (git diff in the working tree) and the files it changed, ` +
      `and try to refute the result: the package or the edited plan text still contradicts a changed goal, or ` +
      `the change goes beyond what the goal edit requires. Default to refuted=true if uncertain.`,
      { label: `refute:${s.id}`, phase: 'Rederive', schema: VERDICT, model: 'sonnet', effort: 'high' })
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
