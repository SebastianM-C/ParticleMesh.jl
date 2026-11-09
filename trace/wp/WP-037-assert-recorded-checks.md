+++
id = "WP-037"
status = "active"
goals = ["A4", "A7"]
depends_on = ["WP-018", "WP-032"]
+++
# WP-037: assert what the tests only record

Turn the tests that compute a number and pass whatever it is into checks that fail when it goes wrong: 2D Reference digests against recorded values, Gauss's law in the Weibel case, Fast against Reference histories, each pusher's reproducibility and order, a CPU–CUDA bound, a Landau rate bound from WP-032's ensemble, and refusal of mismatched or unsorted input to the 2D flux deposit. It works when each new check fails on a planted defect and passes on `main`.
