+++
id = "WP-089"
status = "proposed"
goals = ["A0", "A4", "A7", "A9"]
depends_on = ["WP-086"]
adds_dependencies = []
+++
# WP-089: the push's gather through the engine's gather rule, so that the 3D reverse pass fits at λ₀/32

Gather E and B at the particles through the engine's `gather!` and its reverse rule, which tapes the fields once per step, and push from the gathered values, so that Enzyme's tape of the push no longer holds the 162 field values per particle slot (688 bytes per slot, 11.8 GiB for the plasma at λ₀/32; `reports/wp-086-reverse-3d-memory.md`). It works when the step keeps its bits on every backend and element type, and on the RTX 5090 the reverse pass of the 3D trapped-beam case fits at λ₀/32 and agrees with Dual{1} runs.
