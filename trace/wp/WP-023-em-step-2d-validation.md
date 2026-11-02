+++
id = "WP-023"
status = "done"
goals = ["A1", "A2", "A4", "A5", "A7", "A8"]
depends_on = ["WP-019", "WP-020", "WP-022", "WP-024"]
adds_dependencies = []
+++
# WP-023: the 2D electromagnetic step, validated

Join the 2D current deposition (WP-020), the field solver (WP-022) and the 2D state and push (WP-024) into a periodic 2D3V electromagnetic step, in both deposition modes. It works when Gauss's law holds to roundoff with moving particles and the oblique Landau and Weibel cases of WP-019 agree with linear theory and with the committed Smilei runs. Outcome: `reports/wp-023-em-step-2d.md`.
