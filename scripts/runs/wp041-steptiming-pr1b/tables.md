Seconds per run of 512 steps (16 threads, AMD EPYC 9655 96-Core Processor):

| element type | mode | `run!` | push | sort | deposit | window | fields | other |
|---|---|---|---|---|---|---|---|---|
| `Float32` | Reference | 1.06 | 0.19 | 0.36 | 0.19 | 0.13 | 0.02 | 0.04 |
| `Float32` | Fast | 1.21 | 0.19 | 0.0 | 0.68 | 0.26 | 0.03 | 0.05 |
| `Float64` | Reference | 1.12 | 0.2 | 0.37 | 0.22 | 0.14 | 0.03 | 0.04 |
| `Float64` | Fast | 1.23 | 0.19 | 0.0 | 0.64 | 0.21 | 0.03 | 0.05 |
| `Dual{Float32,2}` | Reference | 2.76 | 0.69 | 0.5 | 0.89 | 0.19 | 0.05 | 0.13 |
| `Dual{Float64,2}` | Reference | 3.23 | 0.69 | 0.79 | 1.25 | 0.32 | 0.07 | 0.16 |

Milliseconds per call on the last state:

| element type | `deposit_flux!` | patch kernel | reduce kernel | `sort_by_cell!` (sorted input) |
|---|---|---|---|---|
| `Float32` | 0.4 | 0.27 | 0.1 | 0.38 |
| `Float64` | 0.41 | 0.29 | 0.1 | 0.54 |
| `Dual{Float32,2}` | 1.65 | 1.46 | 0.16 | 0.65 |
| `Dual{Float64,2}` | 2.08 | 1.69 | 0.32 | 0.92 |

Milliseconds per call of the patch kernel per work-group:

| element type | `nothing` | `1` | `8` | `64` | `512` |
|---|---|---|---|---|---|
| `Float32` | 0.31 | 0.26 | 0.29 | 0.32 | 0.34 |
| `Float64` | 0.33 | 0.32 | 0.39 | 0.36 | 0.36 |
| `Dual{Float32,2}` | 9.16 | 1.47 | 4.35 | 4.53 | 34.1 |
| `Dual{Float64,2}` | 19.4 | 1.71 | 4.62 | 9.26 | 64.8 |

Milliseconds per call of the deposition per patch layout:

| element type | `deposit_flux!` StencilMajor | patch StencilMajor | reduce StencilMajor | `deposit_flux!` CellMajor | patch CellMajor | reduce CellMajor |
|---|---|---|---|---|---|---|
| `Float32` | 0.38 | 0.26 | 0.1 | 1.32 | 1.14 | 0.16 |
| `Float64` | 0.4 | 0.3 | 0.1 | 1.54 | 1.31 | 0.16 |
| `Dual{Float32,2}` | 1.57 | 1.43 | 0.16 | 2.58 | 2.31 | 0.2 |
| `Dual{Float64,2}` | 2.06 | 1.68 | 0.32 | 3.11 | 2.68 | 0.3 |
