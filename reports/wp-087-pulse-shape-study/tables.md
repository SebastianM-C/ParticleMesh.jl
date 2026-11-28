Run `optim-pulse2d-opt-4080` at commit `1a1a158048b18f3d77a280cf014f723c12f87556`: ppl 32, ny 2, Float32, cuda, compensated gather, maxiters 25, pgtol 0.015625 of the reference gradient, seed 87, na 0.325556.

Reference (unaberrated, transform-limited, focus 80): J = 0.02501, dJ/dz = 0.1501, 0.003179, -0.09698, 0.02084, 0.0009695 (focus, aberration3, aberration4, gdd, tod), a₀ = 1.898, 113.0 s.

| start | start x | optimum x | J | J/J_ref | Strehl | h | on bounds | retcode | iterations | points run | max proj. grad. (z) | seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| r1 | 77.7, -0.0366, -0.0597, 2.81, 8.85 | 110.0, -0.005805, -0.1185, 10.48, -30.0 | 0.2263 | 9.05 | 0.955 | 8.37 | focus=hi, tod=lo | MaxIters | 25 | 46 | 0.00841 | 4595 |
| r2 | 72.2, 0.162, 0.0393, 5.35, 15.6 | 110.0, 0.05972, -0.1169, 10.29, -27.4 | 0.2199 | 8.79 | 0.952 | 8.05 | focus=hi | MaxIters | 25 | 46 | 0.0292 | 4577 |
| r3 | 96.9, -0.0441, 0.216, 5.45, 15.0 | 110.0, 0.04774, -0.1354, 6.537, -1.337 | 0.1973 | 7.89 | 0.939 | 6.38 | none | MaxIters | 25 | 64 | 0.0621 | 6312 |
| r4 | 75.7, -0.123, 0.0208, -1.5, -20.5 | 110.0, 0.01263, -0.1178, 9.753, -28.78 | 0.2252 | 9.0 | 0.956 | 8.09 | focus=hi | MaxIters | 25 | 42 | 0.0284 | 4186 |
| r5 | 80.5, -0.079, 0.0912, -0.523, 1.33 | 110.0, 0.0299, -0.1168, 9.376, -22.19 | 0.2214 | 8.85 | 0.955 | 7.29 | focus=hi | MaxIters | 25 | 43 | 0.0395 | 4395 |

260 points run with duals in five directions, median 99.5 s each; lowest launched share in the interior rows over all points 0.973682.

Check at the best point, Float64 at ppl 32, ny 2, central differences at 2^-10 of each scale (J = 0.226297):

| direction | dJ/dz central differences | dJ/dz ForwardDiff | relative difference |
|---|---|---|---|
| focus | 0.070628 | 0.0509593 | 0.39 |
| aberration3 | -0.0030173 | -0.00408066 | 0.26 |
| aberration4 | -0.00278529 | -0.00115683 | 1.4 |
| gdd | -0.000800382 | -0.00842086 | 0.9 |
| tod | 0.000213483 | -0.00608476 | 1.0 |

Check at the best point, Float64 at ppl 32, ny 2, central differences at 2^-13 of each scale (J = 0.226297):

| direction | dJ/dz central differences | dJ/dz ForwardDiff | relative difference |
|---|---|---|---|
| focus | 0.0485772 | 0.0509593 | 0.047 |
| aberration3 | -0.0278006 | -0.00408066 | 5.8 |
| aberration4 | -0.00158495 | -0.00115683 | 0.37 |
| gdd | -0.0309144 | -0.00842086 | 2.7 |
| tod | -0.0122385 | -0.00608476 | 1.0 |

Check at the reference point, Float64 at ppl 32, ny 2, central differences at 2^-13 of each scale (J = 0.0250134):

| direction | dJ/dz central differences | dJ/dz ForwardDiff | relative difference |
|---|---|---|---|
| focus | 0.14988 | 0.15011 | 0.0015 |
| aberration3 | 0.00317854 | 0.00317861 | 2.2e-5 |
| aberration4 | -0.0969838 | -0.0969849 | 1.1e-5 |
| gdd | 0.0208353 | 0.020843 | 0.00037 |
| tod | 0.000969737 | 0.000969516 | 0.00023 |

Check at the best point, Float64 at ppl 32, ny 2, central differences at 2^-16 of each scale (J = 0.226297):

| direction | dJ/dz central differences | dJ/dz ForwardDiff | relative difference |
|---|---|---|---|
| focus | 0.0435921 | 0.0509593 | 0.14 |
| aberration3 | -0.00421897 | -0.00408066 | 0.034 |
| aberration4 | -0.00114722 | -0.00115683 | 0.0083 |
| gdd | -0.00795245 | -0.00842086 | 0.056 |
| tod | -0.00614013 | -0.00608476 | 0.0091 |

