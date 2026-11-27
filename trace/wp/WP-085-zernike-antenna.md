+++
id = "WP-085"
status = "done"
goals = ["A2", "A3", "A7", "A8"]
depends_on = ["WP-066", "WP-073", "WP-083"]
+++
# WP-085: a Zernike pupil beam on the 3D antenna

Launch through the 3D antenna the focus of a round pupil with a near field and a wavefront of Zernike modes (OSA/ANSI indexing), its plane's profile and derivatives tabulated once per parameter point by a pupil-sum kernel that the step's antenna reads. It works when the profile matches its closed forms (the Gaussian pupil's round Gaussian beam, the Airy pattern and its zeros, Maréchal's Strehl ratio, astigmatism's two line foci), the launched beam is the Yee propagation of its incident field carrying the energy the plane drives, a phase-only wavefront keeps the energy, and the coefficients' gradients by duals agree with finite differences and by Enzyme on CUDA with ForwardDiff.
