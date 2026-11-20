| gradient | checkpoints | every C steps | device heap (GiB) | dJ/dτ | dJ/dw | dJ/dfocus | dJ/dn | largest relative difference from Dual{4} | primal with checkpoints (s) | reverse, no compilation (s) | reverse per step (ms) | primal + reverse (s) | start tangent, first call (s) | card memory in use, peak (GiB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Dual{4} | | | | -0.00616894 | 0.000767155 | 0.00203787 | 1.65191 | | | | | 16.5 | | |
| reverse | device | 20 | 1 | -0.00616924 | 0.000767048 | 0.00203783 | 1.65187 | 1.4e-04 | 6.5 | 20.1 | 11.2 | 26.6 | 5.1 | 31.4 |
| reverse | device | 40 | 1 | -0.00607272 | 0.000846864 | 0.00203008 | 1.64241 | 1.0e-01 | 6.5 | 22.5 | 12.5 | 29.0 | 5.1 | 31.4 |
| reverse | host | 5 | 1 | -0.00616924 | 0.000767041 | 0.00203783 | 1.65187 | 1.5e-04 | 22.8 | 17.0 | 9.4 | 39.7 | 9.9 | 11.7 |
| reverse | host | 10 | 1 | -0.00616924 | 0.000767036 | 0.00203783 | 1.65187 | 1.6e-04 | 14.2 | 16.6 | 9.2 | 30.8 | 5.1 | 16.6 |
| reverse | host | 20 | 4 | -0.00616923 | 0.000767046 | 0.00203783 | 1.65187 | 1.4e-04 | 10.6 | 19.5 | 10.9 | 30.1 | 5.0 | 24.5 |
| reverse | host | 30 | 1 | -0.00617198 | 0.000766831 | 0.00203768 | 1.65219 | 4.9e-04 | 9.5 | 18.8 | 10.5 | 28.3 | 5.2 | 22.4 |
| reverse | host | 40 | 1 | -0.00607272 | 0.000846864 | 0.00203008 | 1.64241 | 1.0e-01 | 8.7 | 19.2 | 10.7 | 27.9 | 5.1 | 28.5 |
| reverse | host | 40 | 4 | -0.00607272 | 0.000846866 | 0.00203008 | 1.64241 | 1.0e-01 | 8.6 | 23.3 | 13.0 | 32.0 | 5.3 | 31.4 |

| run | t | steps | primal (ms per step) | reverse (ms per step) | reverse / primal |
|---|---|---|---|---|---|
| enzyme-trapped-steps-20261120T163522 | 180 | 1 | 1.038 | 11.39 | 11.0 |
| enzyme-trapped-steps-20261120T163522 | 180 | 4 | 0.959 | 9.03 | 9.4 |
| enzyme-trapped-steps-20261120T163522 | 180 | 16 | 0.917 | 9.31 | 10.2 |
| enzyme-trapped-steps-20261120T164202 | 180 | 1 | 1.023 | 10.24 | 10.0 |
| enzyme-trapped-steps-20261120T164202 | 180 | 4 | 0.956 | 8.45 | 8.8 |
| enzyme-trapped-steps-20261120T164202 | 180 | 16 | 0.916 | 9.42 | 10.3 |

Optimization run wp078-opt-born-reverse-5090, 87 gradients: median per gradient 40.0 s (primal with checkpoints 5.76 s, reverse 33.0 s, start tangent 0.007 s); the window is 3 λ₀ longer than the gradient table's case (`cap=cetal25`).
