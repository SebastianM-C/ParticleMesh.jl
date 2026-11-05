+++
id = "WP-040"
status = "proposed"
goals = ["A4", "A8"]
depends_on = ["WP-013", "WP-021"]
+++
# WP-040: periodic spectral solve on CUDA, and bound NTuple signatures

Run the periodic spectral solve's tests on CUDA (bits across launch shapes, a measured CPU–CUDA tolerance, no allocation growth), and bind the type parameters of the seven `ParticleMeshCore` NTuple signatures that Aqua flags. It works when the CUDA tests pass, Aqua's unbound-parameter check passes without `broken`, and the CPU digests keep their bits.
