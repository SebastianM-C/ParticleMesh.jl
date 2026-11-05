+++
id = "WP-034"
status = "active"
goals = ["A2", "A4", "A7", "A8", "A10"]
depends_on = ["WP-027", "WP-028"]
adds_dependencies = []
+++
# WP-034: a convolutional perfectly matched layer

A convolutional PML (graded σ, κ and frequency shift α as case parameters, PIConGPU's profile by default and Smilei's expressible) on the open y axis next to the matched layer (A2: PML later), and on the moving window's back edge, its convolution fields shifted with the window. It works when the reflection and transmission of WP-027's slab packet match the scheme's exact discrete theory of the PML to roundoff over thickness and angle, a long run stays bounded, WP-028's vacuum twin loses its back-edge reflection down to the layer's own, and Gauss's law holds to roundoff off the layer.
