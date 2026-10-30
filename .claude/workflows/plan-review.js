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
