# Questions and assumptions

Agents do not ask blocking questions. When something is undecided, the agent writes `questions/<id>.md` with the question, the options, the assumption it proceeded with, and the work packages or files affected, says so in its PR description, and continues. The owner answers by editing the file, or by editing a goal in `docs/goals/` when the answer changes one; the planner then re-derives what depends on it. Problems with the setup, bugs and upstream suspicions are issues, not questions (`CLAUDE.md`, "Issues").
