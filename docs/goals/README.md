# Goals and decisions

The owner's input to the project: the aims (A0) and the binding decisions A1 to A10, one file each. Everything else (requirements, interfaces, work packages, tests, plans) is derived from these files by the factory and must not contradict them.

- Only the owner changes these files, or approves a pull request that does (tier 2, `.github/CODEOWNERS`). An agent that finds a decision wrong or blocking writes `questions/<id>.md` and proceeds on a stated assumption; it may also open a pull request that proposes the edit and says why.
- A file states the decision as it stands; git is its history. Never keep superseded text next to the new one.
- Each fact has one home; derived text cites it by id, it does not copy it (the map of homes is in `CLAUDE.md`, "Working here").
- Work packages name the goals they serve (`goals = ["A3"]` in their front matter) and record the version of each goal they were checked against. Editing a goal makes those packages stale; the factory re-derives them (`trace/README.md`).
- Ids are never reused. A11 to A14 were process settings and are now the factory settings F1 to F4 in `docs/plan/factory-and-observability.md`.
