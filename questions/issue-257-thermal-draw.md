# The thermal draw of the quiet start passes c (issue #257)

## Question

`load_quiet!` draws each velocity component as `drift + thermal·z` with `z` standard normal, in velocity. The electromagnetic module turns velocities into momenta with γ = 1/√(1 − v²), which is not finite for |v| ≥ 1, and the Gaussian tail reaches |v| ≥ 1 with a probability that grows with the thermal speed (the WP-023 plasma `plasma_step_2d` with the electron v_th at 0.3 draws |v| = 1.39). Should a relativistic species instead be drawn in momentum (`u = drift + thermal·z`, always finite) or from a Maxwell–Jüttner distribution?

Either change alters the distribution behind the committed cross-code results (the case files' `thermal_velocity` enters as a velocity-space spread), so it is the owner's decision.

## Assumption proceeded on

The draw stays in velocity. `initialize!` refuses, at N = 1 and 2, a load in which a drawn speed of an alive particle reaches c (or is NaN): one reduction over the device arrays, then one host comparison, before the velocities become momenta and outside `step!`. The `ArgumentError` names the species, the largest drawn |v| and that the thermal spread is drawn in velocity (T-480, T-481). Every load whose speeds stay below c loads with the same bits as before.
