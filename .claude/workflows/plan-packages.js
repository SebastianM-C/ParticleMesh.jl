export const meta = {
  name: 'plan-packages',
  description: 'Cut the next work packages from the ParticleMesh plan, verify each against the plan, and write them as records under trace/wp/',
  whenToUse: 'When no dispatchable package is left, or after the plan or a decision changed; the records it writes go into one PR that merges on the automated review, and the digest lists them',
  phases: [
    { title: 'Propose', detail: 'three planners, each from a different angle' },
    { title: 'Merge', detail: 'one judge merges the proposals into an ordered list' },
    { title: 'Verify', detail: 'two refuters per package, distinct lenses' },
    { title: 'Write', detail: 'surviving packages as records in the existing format' },
  ],
}

// Inputs come from args; Date.now() and the filesystem are unavailable in a workflow script.
//   dir       plan directory (default docs/plan)
//   scope     one sentence naming the slice of the plan to cut, e.g. "the engine core on the CPU backend"
//   existing  directory of the existing records (default trace/wp)
//   next_id   the first free package number, e.g. 3 for WP-003; the caller reads trace/wp before invoking
//   max       upper bound on packages per run (default 6)
//   date      stamped by the caller
const DIR = args.dir || 'docs/plan'
const SCOPE = args.scope
const EXISTING = args.existing || 'trace/wp'
const NEXT = args.next_id
const MAX = args.max || 6
const DATE = args.date
if (!SCOPE || !NEXT) throw new Error('plan-packages needs args.scope and args.next_id')

const CONTEXT =
  `Planning documents are in ${DIR}; start with overview.md and open-items.md, where decisions A1 to A14 are binding. ` +
  `Existing work-package records are in ${EXISTING}; WP-001 there is the template for structure and tone. ` +
  `Slice to cut: ${SCOPE}. A package is one pull request by one session: it implements one interface or one ` +
  `validation step, names the tests that prove it (with trace tags), the budgets from A7 where they apply, what it ` +
  `must not do, and the decisions and packages it depends on. Do not edit anything.`

const ANGLES = [
  { key: 'interface',  prompt: 'interface-first: each package implements one interface contract of architecture-draft.md section 0 and nothing else' },
  { key: 'validation', prompt: 'validation-first: each package is defined by the test or benchmark that proves it, from the requirements of landscape-and-requirements.md section 6 and the budgets of A7' },
  { key: 'risk',       prompt: 'risk-first: the packages that retire the thin spots B and the spikes (Reactant, symbolic, FP32) earliest, smallest first' },
]

const PACKAGE = {
  type: 'object',
  properties: {
    title: { type: 'string' }, slug: { type: 'string' }, scope: { type: 'string' },
    deliverables: { type: 'array', items: { type: 'string' } },
    non_goals: { type: 'array', items: { type: 'string' } },
    acceptance: { type: 'array', items: { type: 'string' } },
    depends_on: { type: 'array', items: { type: 'string' } },
    budgets: { type: 'string' },
    tier: { type: 'string', enum: ['0', '1', '2'] },
    rationale: { type: 'string' },
  },
  required: ['title', 'slug', 'scope', 'deliverables', 'non_goals', 'acceptance', 'depends_on', 'budgets', 'tier', 'rationale'],
}
const PROPOSAL = { type: 'object', properties: { packages: { type: 'array', items: PACKAGE } }, required: ['packages'] }
const VERDICT = {
  type: 'object',
  properties: { refuted: { type: 'boolean' }, reason: { type: 'string' }, fixes: { type: 'array', items: { type: 'string' } } },
  required: ['refuted', 'reason', 'fixes'],
}
const WRITTEN = { type: 'object', properties: { files: { type: 'array', items: { type: 'string' } } }, required: ['files'] }

// Propose: a barrier is right here, the merge needs all three proposals at once.
const proposals = (await parallel(ANGLES.map(a => () => agent(
  `${CONTEXT} Propose at most ${MAX} packages from this angle: ${a.prompt}. ` +
  `Skip anything an existing record already covers. Keep each package small enough for one session.`,
  { label: `propose:${a.key}`, phase: 'Propose', schema: PROPOSAL })))).filter(Boolean)
log(`${proposals.length} proposals, ${proposals.reduce((n, p) => n + p.packages.length, 0)} candidate packages`)

const merged = await agent(
  `${CONTEXT} Three planners proposed packages from different angles: ${JSON.stringify(proposals)}. ` +
  `Merge them into one list of at most ${MAX} packages, ordered so that every dependency comes before its ` +
  `dependents; merge duplicates, drop anything outside the slice, keep every package's structure complete.`,
  { label: 'merge', phase: 'Merge', schema: PROPOSAL, effort: 'high' })
const candidates = merged ? merged.packages : []
const titles = candidates.map(p => p.title)

// Verify: each package through two lenses, as a pipeline so early packages are judged while later ones wait.
const LENSES = [
  { key: 'fit',  prompt: 'scope: it overlaps an existing record or another proposed package, exceeds one session, or has no clear interface' },
  { key: 'plan', prompt: 'plan compliance: it contradicts a decision A1 to A14 or a rule of CLAUDE.md, its acceptance is not testable, a dependency is missing or circular, or budgets are absent where A7 applies' },
]
const verified = await pipeline(
  candidates,
  pkg => parallel(LENSES.map(l => () => agent(
    `${CONTEXT} A planner proposes this package: ${JSON.stringify(pkg)}. The other proposed packages are: ` +
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
  `Write these work packages as records in ${EXISTING}, one file each, in the format and naming the existing ` +
  `records there use (Markdown WP-<nnn>-<slug>.md until WP-001 lands, then the TOML record format it defines), ` +
  `numbered from ${NEXT} in this order: ` +
  `${JSON.stringify(survivors.map(p => ({ ...p, votes: undefined, survives: undefined })))}. ` +
  `Apply the fixes the verifiers suggested: ${JSON.stringify(survivors.map(p => ({ slug: p.slug, fixes: p.votes.flatMap(v => v.fixes) })))}. ` +
  `Follow the structure of the existing WP-001 record exactly (Status, Scope, Deliverables, Non-goals, ` +
  `Acceptance, Depends on, Budgets), status proposed, and use the ids you assign in the depends-on lists. ` +
  `Write nothing else and do not commit.`,
  { label: 'write', phase: 'Write', schema: WRITTEN, effort: 'high' }) : { files: [] }

return { date: DATE, scope: SCOPE, files: written ? written.files : [], dropped }
