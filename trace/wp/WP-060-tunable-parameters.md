+++
id = "WP-059"
status = "done"
goals = ["A3", "A8", "A9"]
depends_on = ["WP-055", "WP-008"]
adds_dependencies = ["SciMLStructures"]
+++
# WP-059: the differentiable parameters as one vector, and as a SciML structure

Read the numbers of an `EMProblem` and of an `EMParams` that A3 lets a derivative be taken with respect to into one named vector and write a vector back, in any element type, and make that vector the `Tunable` portion of both through SciMLStructures, so that SciML tools reach the parameters of the harness's `DiscreteProblem` without code of the user's. It works when the round trip through the vector gives the description back field for field per element type, and the gradient of WP-043's wake amplitude taken through `solve` with the five parameters found by name in the Tunable portion has the bits of the hand-built dual run. Outcome: T-770 to T-773 in `lib/ParticleMeshElectromagnetic/test/tunables.jl`.
