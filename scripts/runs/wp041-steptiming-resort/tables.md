Seconds per run of 512 steps (16 threads, AMD EPYC 9655 96-Core Processor):

| element type | mode | `run!` | push | sort | deposit | window | fields | other |
|---|---|---|---|---|---|---|---|---|
| `Float32` | Reference | 0.9 | 0.18 | 0.25 | 0.19 | 0.1 | 0.03 | 0.04 |
| `Float64` | Reference | 0.97 | 0.2 | 0.28 | 0.23 | 0.11 | 0.03 | 0.05 |
| `Dual{Float32,2}` | Reference | 2.34 | 0.71 | 0.36 | 0.89 | 0.15 | 0.06 | 0.14 |
| `Dual{Float64,2}` | Reference | 3.15 | 0.62 | 0.6 | 1.25 | 0.28 | 0.07 | 0.15 |

Milliseconds per call on the last state:

| element type | `deposit_flux!` | patch kernel | reduce kernel | `sort_by_cell!` (sorted input) |
|---|---|---|---|---|
| `Float32` | 0.57 | 0.26 | 0.1 | 0.26 |
| `Float64` | 0.55 | 0.3 | 0.11 | 0.28 |
| `Dual{Float32,2}` | 1.68 | 1.52 | 0.17 | 0.35 |
| `Dual{Float64,2}` | 2.06 | 1.71 | 0.35 | 0.39 |

Milliseconds per call of the patch kernel per work-group:

| element type | `nothing` | `1` | `8` | `64` | `512` |
|---|---|---|---|---|---|
| `Float32` | 0.47 | 0.29 | 0.51 | 0.51 | 0.56 |
| `Float64` | 0.51 | 0.34 | 0.55 | 0.56 | 0.62 |
| `Dual{Float32,2}` | 9.11 | 1.46 | 4.5 | 4.78 | 34.9 |
| `Dual{Float64,2}` | 18.5 | 1.68 | 4.71 | 9.21 | 59.0 |

Milliseconds per call of the deposition per patch layout:

| element type | `deposit_flux!` StencilMajor | patch StencilMajor | reduce StencilMajor | `deposit_flux!` CellMajor | patch CellMajor | reduce CellMajor |
|---|---|---|---|---|---|---|
| `Float32` | 0.58 | 0.26 | 0.1 | 1.39 | 1.19 | 0.17 |
| `Float64` | 0.49 | 0.3 | 0.1 | 1.7 | 1.39 | 0.18 |
| `Dual{Float32,2}` | 1.62 | 1.43 | 0.16 | 2.68 | 2.39 | 0.21 |
| `Dual{Float64,2}` | 2.03 | 1.68 | 0.31 | 3.08 | 2.66 | 0.31 |
