+++
id = "WP-057"
status = "done"
goals = ["A2", "A3", "A4", "A9", "A0"]
depends_on = ["WP-055", "WP-056", "WP-022", "WP-026", "WP-027", "WP-028", "WP-034"]
adds_dependencies = []
+++
# WP-057: one field solve over a tuple of field modifiers, and one step body over N

Make the mesh solve of the electromagnetic module one function that walks a tuple of field modifiers held by the state (an open axis' absorbing layer or PML, the PML of the moving window's back edge, the laser antennas, the window's edges), each with its arrays and coefficients in the state and what it reads of the parameters taken from `EMParams` at the step, and make `initialize!` and `step!` one method each over N with the 1D case as the case of one axis and a dispatch on N only where the two dimensions give different bits or the engine's conventions differ, without changing a result. It works when the fixed set of short 1D and 2D runs of WP-055 gives the same bits before and after per element type and deposition mode, a modifier defined by a test acts at its place among the module's at no allocation, and the solve with no modifiers is the periodic solve bit for bit. Outcome: the description of the pull request that adds this file.
