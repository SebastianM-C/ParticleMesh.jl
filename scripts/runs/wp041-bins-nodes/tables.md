Seconds per run of 512 steps (16 threads, AMD EPYC 9655 96-Core Processor):

| element type | mode | `run!` | push | sort | deposit | window | fields | other |
|---|---|---|---|---|---|---|---|---|
| `Float32` | Reference | 0.88 | 0.19 | 0.36 | 0.15 | 0.11 | 0.03 | 0.04 |
| `Float32` | Fast | 1.1 | 0.18 | 0.0 | 0.65 | 0.17 | 0.02 | 0.04 |
| `Float64` | Reference | 0.98 | 0.19 | 0.43 | 0.17 | 0.14 | 0.03 | 0.05 |
| `Float64` | Fast | 1.14 | 0.18 | 0.0 | 0.63 | 0.14 | 0.04 | 0.05 |
| `Dual{Float32,2}` | Reference | 2.13 | 0.82 | 0.47 | 0.59 | 0.2 | 0.05 | 0.14 |
| `Dual{Float64,2}` | Reference | 2.65 | 0.62 | 0.71 | 0.79 | 0.28 | 0.07 | 0.14 |

Milliseconds per call on the last state:

| element type | `deposit_flux!` | patch kernel | reduce kernel | `sort_by_cell!` (sorted input) |
|---|---|---|---|---|
| `Float32` | 0.3 | 0.21 | 0.08 | 0.42 |
| `Float64` | 0.38 | 0.27 | 0.1 | 0.55 |
| `Dual{Float32,2}` | 1.32 | 0.97 | 0.12 | 0.66 |
| `Dual{Float64,2}` | 1.29 | 1.06 | 0.16 | 0.92 |

Milliseconds per call of the patch kernel per work-group:

| element type | `nothing` | `1` | `8` | `64` | `512` |
|---|---|---|---|---|---|
| `Float32` | 0.24 | 0.21 | 0.29 | 0.33 | 0.37 |
| `Float64` | 0.29 | 0.25 | 0.46 | 0.28 | 0.32 |
| `Dual{Float32,2}` | 6.16 | 0.98 | 2.76 | 3.08 | 21.6 |
| `Dual{Float64,2}` | 12.2 | 1.18 | 2.92 | 6.12 | 40.6 |

Milliseconds per call of the deposition per patch layout:

| element type | `deposit_flux!` StencilMajor | patch StencilMajor | reduce StencilMajor | `deposit_flux!` CellMajor | patch CellMajor | reduce CellMajor |
|---|---|---|---|---|---|---|
| `Float32` | 0.45 | 0.24 | 0.09 | 0.83 | 0.7 | 0.12 |
| `Float64` | 0.32 | 0.23 | 0.08 | 0.96 | 0.77 | 0.11 |
| `Dual{Float32,2}` | 1.2 | 0.99 | 0.13 | 1.89 | 1.57 | 0.14 |
| `Dual{Float64,2}` | 1.34 | 1.05 | 0.16 | 2.02 | 1.7 | 0.18 |
