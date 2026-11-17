Seconds per run of 512 steps (16 threads, AMD EPYC 9655 96-Core Processor):

| element type | mode | `run!` | push | sort | deposit | window | fields | other |
|---|---|---|---|---|---|---|---|---|
| `Float32` | Reference | 1.04 | 0.17 | 0.29 | 0.26 | 0.12 | 0.04 | 0.04 |
| `Float32` | Fast | 1.36 | 0.16 | 0.0 | 0.84 | 0.14 | 0.05 | 0.05 |
| `Float64` | Reference | 1.78 | 0.18 | 0.67 | 0.33 | 0.29 | 0.07 | 0.06 |
| `Float64` | Fast | 1.43 | 0.17 | 0.0 | 0.74 | 0.27 | 0.07 | 0.06 |
| `Dual{Float32,2}` | Reference | 3.49 | 0.65 | 1.02 | 1.08 | 0.36 | 0.11 | 0.13 |
| `Dual{Float64,2}` | Reference | 4.78 | 0.62 | 1.45 | 1.18 | 0.61 | 0.13 | 0.13 |

Milliseconds per call on the last state:

| element type | `deposit_flux!` | patch kernel | reduce kernel | `sort_by_cell!` (sorted input) |
|---|---|---|---|---|
| `Float32` | 0.49 | 0.24 | 0.09 | 0.47 |
| `Float64` | 0.6 | 0.3 | 0.1 | 1.14 |
| `Dual{Float32,2}` | 2.03 | 1.27 | 0.14 | 1.45 |
| `Dual{Float64,2}` | 2.29 | 1.38 | 0.17 | 2.12 |

Milliseconds per call of the patch kernel per work-group:

| element type | `nothing` | `1` | `8` | `64` | `512` |
|---|---|---|---|---|---|
| `Float32` | 0.27 | 0.25 | 0.28 | 0.28 | 0.29 |
| `Float64` | 0.33 | 0.32 | 0.33 | 0.32 | 0.38 |
| `Dual{Float32,2}` | 8.89 | 1.27 | 3.73 | 3.99 | 30.3 |
| `Dual{Float64,2}` | 15.7 | 1.39 | 3.81 | 7.96 | 53.6 |

Milliseconds per call of the deposition per patch layout:

| element type | `deposit_flux!` StencilMajor | patch StencilMajor | reduce StencilMajor | `deposit_flux!` CellMajor | patch CellMajor | reduce CellMajor |
|---|---|---|---|---|---|---|
| `Float32` | 0.49 | 0.24 | 0.08 | 1.16 | 0.91 | 0.14 |
| `Float64` | 0.6 | 0.29 | 0.09 | 1.3 | 0.98 | 0.14 |
| `Dual{Float32,2}` | 2.04 | 1.27 | 0.13 | 2.57 | 2.08 | 0.16 |
| `Dual{Float64,2}` | 2.3 | 1.39 | 0.15 | 2.96 | 2.27 | 0.18 |
