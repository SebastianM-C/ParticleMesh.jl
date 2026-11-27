+++
id = "WP-086"
status = "active"
goals = ["A0", "A4", "A7", "A8", "A9"]
depends_on = ["WP-080", "WP-084"]
adds_dependencies = []
+++
# WP-086: the 3D reverse pass at λ₀/32 in 32 GB, without the script patches the engine can carry

Give Checkpointing.jl's reverse sweep a working copy that holds only the evolved arrays (a `StepBody` loop body whose `checkpoint_alloc` is a state view over copies of them), and move into the engine what WP-084's runs patched in from scripts: `current!`'s species loops as `map`s (#384), the flag counts through an integer count that Enzyme takes as inactive (#551), and the density ramp through `sin` (#545). It works when the working copy continues the run bit for bit with the memory of one checkpoint, Enzyme's gradient through it equals the plain reverse pass's, and on the RTX 5090 the reverse pass of WP-084's 3D trapped-beam case fits at λ₀/32 and agrees with Dual{1} runs by the chain rule, with only KernelAbstractions' Active-argument patch (#546) left as a script.
