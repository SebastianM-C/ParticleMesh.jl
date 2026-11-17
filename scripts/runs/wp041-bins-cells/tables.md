Seconds per run of 512 steps (16 threads, AMD EPYC 9655 96-Core Processor):

| element type | mode | `run!` | push | sort | deposit | window | fields | other |
|---|---|---|---|---|---|---|---|---|
| `Float32` | Reference | 0.93 | 0.19 | 0.28 | 0.19 | 0.11 | 0.02 | 0.04 |
| `Float32` | Fast | 1.14 | 0.18 | 0.0 | 0.67 | 0.11 | 0.09 | 0.05 |
| `Float64` | Reference | 0.99 | 0.2 | 0.31 | 0.22 | 0.11 | 0.03 | 0.05 |
| `Float64` | Fast | 1.11 | 0.19 | 0.0 | 0.62 | 0.22 | 0.03 | 0.05 |
| `Dual{Float32,2}` | Reference | 2.55 | 0.66 | 0.4 | 0.89 | 0.17 | 0.05 | 0.14 |
| `Dual{Float64,2}` | Reference | 3.34 | 0.85 | 0.63 | 1.31 | 0.27 | 0.07 | 0.17 |

Milliseconds per call on the last state:

| element type | `deposit_flux!` | patch kernel | reduce kernel | `sort_by_cell!` (sorted input) |
|---|---|---|---|---|
| `Float32` | 0.38 | 0.25 | 0.1 | 0.35 |
| `Float64` | 0.72 | 0.32 | 0.11 | 0.5 |
| `Dual{Float32,2}` | 1.65 | 1.46 | 0.16 | 0.61 |
| `Dual{Float64,2}` | 2.11 | 1.66 | 0.32 | 0.89 |

Milliseconds per call of the patch kernel per work-group:

| element type | `nothing` | `1` | `8` | `64` | `512` |
|---|---|---|---|---|---|
| `Float32` | 0.37 | 0.29 | 0.36 | 0.36 | 0.34 |
| `Float64` | 0.47 | 0.31 | 0.49 | 0.57 | 0.54 |
| `Dual{Float32,2}` | 9.18 | 1.45 | 4.34 | 4.76 | 35.1 |
| `Dual{Float64,2}` | 18.7 | 1.7 | 4.62 | 9.22 | 60.5 |

Milliseconds per call of the deposition per patch layout:

| element type | `deposit_flux!` StencilMajor | patch StencilMajor | reduce StencilMajor | `deposit_flux!` CellMajor | patch CellMajor | reduce CellMajor |
|---|---|---|---|---|---|---|
| `Float32` | 0.38 | 0.26 | 0.1 | 1.27 | 1.07 | 0.17 |
| `Float64` | 0.44 | 0.31 | 0.1 | 1.5 | 1.31 | 0.18 |
| `Dual{Float32,2}` | 1.62 | 1.45 | 0.16 | 2.6 | 2.3 | 0.21 |
| `Dual{Float64,2}` | 2.0 | 1.66 | 0.31 | 3.06 | 2.63 | 0.32 |
