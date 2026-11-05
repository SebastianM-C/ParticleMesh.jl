+++
id = "WP-038"
status = "active"
goals = ["A2", "A5", "A7"]
depends_on = ["WP-029", "WP-017", "WP-019"]
+++
# WP-038: the binomial current filter in the cross-code generators

The Smilei and PIConGPU generators emit a case's `filter_passes` (Smilei's `CurrentFilter` for any count on its CPU build, PIConGPU's binomial current interpolation for one pass), refusing what a code cannot match. It works when a filtered oblique Landau case keeps the filtered Gauss law mode by mode, Δr̂(m) = (H(m) − 1)Δρ̂(m), to roundoff in all three codes and misses it for the neighbouring pass counts.
