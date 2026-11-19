Run `wp068-born48`: λ₀/48, Δy = π/12, Float32, cuda, gather compensated, budget of a₀ = 1.9 at the base pulse; L-BFGS-B with maxiters = 12; objective born (J below is that objective); one start from the best optimum of wp068-born; peak power cap cetal25: τ ≥ 2.455·n^-0.5, the duration between its floor and 12.0, the window 3 λ₀ longer behind, so its dark current column counts a longer region than a run without the cap does.

| start | from (τ, w, focus, n) | a₀ | J there | to (τ, w, focus, n) | a₀ | J there | J ratio | on the box's bounds | stop | iterations | points run | largest dark current weight on the path | largest J_plasma/J on the path |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| from | 7.762, 7.348, 110.0, 0.1 | 1.945 | 0.2217 | 7.762, 7.348, 110.0, 0.1 | 1.945 | 0.2217 | 1.0 | τ: floor→12=lo, focus=hi, density=hi | Success | 0 | 0 | 0.2712 | 0.5042 |

Start `from`, every point the solver reported:

| point | τ | w | focus | n | a₀ | J | J_plasma/J | dark weight | d ln J/dz (τ, w, focus, n) | seconds of the run |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 7.762 | 7.348 | 110.0 | 0.1 | 1.945 | 0.2217 | 0.5042 | 0.2712 | -1.79, -0.0146, 2.76, 1.06 | 68.4 |

Cost: 1 runs with duals in the four free directions, 68.35 s in all, median 68.35 s per run (first run, with compilation: 68.35 s).

The check at the optimum of start `from` (7.762, 7.348, 110.0, 0.1):

| run | J | d ln J/dz (τ, w, focus, n) | dark weight | J_plasma/J | seconds |
|---|---|---|---|---|---|
| optimization: λ₀/48, Float32, duals | 0.221658 | -1.79, -0.0146, 2.76, 1.06 | 0.2712 | 0.5042 | 68.4 |
| λ₀/48, Float64, duals | 0.221718 | -1.79, -0.0157, 2.75, 1.07 | 0.2652 | 0.5048 | 134.0 |
| λ₀/64, Float32, plain | 0.221573 | – | 0.213 | 0.5227 | 38.4 |
| λ₀/64, Float32, duals | 0.221573 | -1.78, -0.0254, 2.7, 1.09 | 0.2116 | 0.5231 | 105.0 |

| direction | ForwardDiff d ln J/dz, Float64 | central difference at 2^-13, Float64 | difference |
|---|---|---|---|
| duration | -1.794 | -1.813 | 0.019 |
| waist | -0.01568 | -0.01054 | 0.0051 |
| focus | 2.748 | 2.675 | 0.073 |
| density | 1.071 | 1.187 | 0.12 |
