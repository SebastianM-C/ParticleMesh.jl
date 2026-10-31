# A2. Baseline numerical scheme, written down as a decision set

Yee; Boris with half-step momentum centering; Esirkepov current; quadratic shape functions (order 1 only in tests); binomial current filter with a documented pass count; transverse open boundaries (simple absorbing layer first, PML later); laser by antenna plane at the window front; immobile ions; quiet-start loading. The AD-smoothness rule decides ties: prefer the smoother variant. Document each choice with its validation test.
