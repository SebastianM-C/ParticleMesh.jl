+++
id = "WP-077"
status = "active"
goals = ["A0", "A4", "A7", "A8"]
depends_on = ["WP-020", "WP-041", "WP-050"]
adds_dependencies = ["EnzymeCore", "Enzyme"]
+++
# WP-077: hand-written reverse rules for the transfers and the sort, registered with Enzyme

Give `deposit!`, `gather!`, `deposit_flux!` (N = 1 and 2) and the sort hand-written pullbacks, registered as Enzyme's reverse rules through EnzymeRules, so that Enzyme keeps no tape of those kernels and the Reference mode's sort differentiates as the inverse permutation. It works when each pullback and Enzyme's reverse pass through the rules agree with forward-mode duals by the chain rule to roundoff in both deposition modes, and when on a GPU the reverse pass of the 2D step with the rules agrees with Enzyme's own, with the cost of each measured.
