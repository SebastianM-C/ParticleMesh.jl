Seconds per run of 512 steps (16 threads, AMD EPYC 9655 96-Core Processor):

| element type | mode | `run!` | push | sort | deposit | window | fields | other |
|---|---|---|---|---|---|---|---|---|
| `Float32` | Reference | 1.0 | 0.19 | 0.25 | 0.19 | 0.11 | 0.02 | 0.04 |
| `Float64` | Reference | 1.0 | 0.2 | 0.32 | 0.22 | 0.12 | 0.03 | 0.05 |
| `Dual{Float32,2}` | Reference | 2.51 | 0.89 | 0.45 | 0.89 | 0.16 | 0.05 | 0.14 |
| `Dual{Float64,2}` | Reference | 3.2 | 0.69 | 0.59 | 1.23 | 0.26 | 0.07 | 0.22 |

Milliseconds per call on the last state:

| element type | `deposit_flux!` | patch kernel | reduce kernel | `sort_by_cell!` (sorted input) |
|---|---|---|---|---|
| `Float32` | 0.45 | 0.26 | 0.1 | 0.36 |
| `Float64` | 0.46 | 0.3 | 0.1 | 0.46 |
| `Dual{Float32,2}` | 1.63 | 1.44 | 0.16 | 0.59 |
| `Dual{Float64,2}` | 2.04 | 1.66 | 0.34 | 0.87 |

Milliseconds per call of the patch kernel per work-group:

| element type | `nothing` | `1` | `8` | `64` | `512` |
|---|---|---|---|---|---|
| `Float32` | 0.45 | 0.28 | 0.43 | 0.5 | 0.44 |
| `Float64` | 0.44 | 0.32 | 0.58 | 0.58 | 0.56 |
| `Dual{Float32,2}` | 9.25 | 1.47 | 4.42 | 4.77 | 36.4 |
| `Dual{Float64,2}` | 19.2 | 1.69 | 4.71 | 9.4 | 59.0 |

Milliseconds per call of the deposition per patch layout:

| element type | `deposit_flux!` StencilMajor | patch StencilMajor | reduce StencilMajor | `deposit_flux!` CellMajor | patch CellMajor | reduce CellMajor |
|---|---|---|---|---|---|---|
| `Float32` | 0.38 | 0.26 | 0.09 | 1.3 | 1.09 | 0.17 |
| `Float64` | 0.43 | 0.31 | 0.1 | 1.44 | 1.2 | 0.18 |
| `Dual{Float32,2}` | 1.65 | 1.49 | 0.16 | 2.6 | 2.3 | 0.2 |
| `Dual{Float64,2}` | 2.01 | 1.65 | 0.3 | 3.13 | 2.65 | 0.31 |
