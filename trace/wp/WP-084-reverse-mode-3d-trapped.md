+++
id = "WP-084"
status = "active"
goals = ["A0", "A3", "A4", "A7", "A8"]
depends_on = ["WP-065", "WP-072", "WP-073", "WP-074", "WP-075", "WP-076", "WP-078", "WP-080", "WP-083"]
adds_dependencies = []
+++
# WP-084: Enzyme's reverse mode through the 3D trapped-beam case

Differentiate the objective of the 3D trapped-beam case (WP-065's case on a 3D box, with the antenna, the moving window, the PMLs, field ionization and the compensated gather) with respect to its four free parameters in Enzyme's reverse mode over the whole run, checkpointed with Checkpointing.jl's Revolve and `checkpoint_storage` (WP-080), on a GPU. It works when the reverse gradient agrees with forward-mode duals of one partial per parameter by the chain rule, Revolve's checkpoints over the case continue to the run's end bit for bit, and the reverse pass's time and device memory against the plain run and the duals are measured at λ₀/16 and λ₀/32 on the RTX 5090.
