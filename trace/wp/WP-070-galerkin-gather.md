+++
id = "WP-070"
status = "active"
goals = ["A2", "A7", "A8"]
depends_on = ["WP-067"]
+++
# WP-070: a Galerkin gather, the transpose of the charge-conserving current

Let the particles read each field component one shape order lower along the axes on which it is staggered, as a numerics option next to the compensated gather that can also be combined with it. It works when E's weights are the current deposition's for a vanishing move and B's follow Faraday's law with the gathered E, both to roundoff, and the total energy of the 1D test cases (quadratic shape) is then kept to the time integrator's error; the trapped-beam objective of WP-065 is compared with the compensated gather under refinement.
