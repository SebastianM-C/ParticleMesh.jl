+++
id = "WP-025"
status = "done"
goals = ["A5", "A7"]
depends_on = ["WP-017", "WP-019"]
adds_dependencies = []
+++
# WP-025: the oblique 2D Landau ensembles over more seeds

Run the oblique 2D Landau case over enough seeds in Smilei and in PIConGPU (FP64) that the 1.5-error difference of their four-seed ensemble damping rates is either resolved as a code difference or shown to be noise, so that A7's 2D thresholds rest on a measured spread. The check is the two codes' ensemble γ against each other and against linear theory, with the seed count chosen from the measured four-seed spreads before the runs. Outcome: `reports/crosscode-picongpu/` ("over 32 seeds").
