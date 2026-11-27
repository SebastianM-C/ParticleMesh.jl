+++
id = "WP-088"
status = "proposed"
goals = ["A0", "A7", "A8"]
depends_on = ["WP-078", "WP-080", "WP-083"]
+++
# WP-088: the trapped-beam step's reverse pass without its open disagreements

Find and remove the causes of the three disagreements left in Enzyme's reverse pass through the trapped-beam step on CUDA: the drift of the checkpointed gradient once a reversed segment holds more than 20 steps (#548), the window refill's `sum!` that frees its flags array and keeps the flag-sum workaround in place (#565), and the per-piece dot-product mismatches of the births and `current!` (#566). It works when each piece's dot-product test and the whole run's reverse gradient at 40 steps per segment agree with forward-mode duals to roundoff, without the flag-sum workaround.
