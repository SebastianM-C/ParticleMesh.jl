+++
id = "WP-083"
status = "active"
goals = ["A0", "A3", "A4", "A7", "A8"]
depends_on = ["WP-026", "WP-066", "WP-073", "WP-077"]
+++
# WP-083: the laser antenna under Enzyme's reverse mode

Let Enzyme's reverse mode differentiate the laser antenna's sources on a GPU without a hand-written rule, by calling `sin` and `cos` where the laser's device functions called `sincos`, whose tape KernelAbstractions cannot allocate on the device (#547). It works when the change leaves the CPU's bits unchanged, when Enzyme's reverse pass through the antenna's incident field on CUDA agrees with ForwardDiff to roundoff, and when the reverse gradient through `step!` of a case with the antenna agrees with duals without the rule.
