Seconds per run of 512 steps (16 threads, AMD EPYC 9655 96-Core Processor):

| element type | mode | `run!` | push | sort | deposit | window | fields | other |
|---|---|---|---|---|---|---|---|---|
| `Float32` | Reference | 1.17 | 0.17 | 0.36 | 0.22 | 0.14 | 0.04 | 0.05 |
| `Float32` | Fast | 1.32 | 0.16 | 0.0 | 0.78 | 0.17 | 0.05 | 0.05 |
| `Float64` | Reference | 1.56 | 0.18 | 0.68 | 0.28 | 0.28 | 0.07 | 0.06 |
| `Float64` | Fast | 1.6 | 0.17 | 0.0 | 0.79 | 0.23 | 0.07 | 0.06 |
| `Dual{Float32,2}` | Reference | 3.28 | 0.59 | 0.92 | 0.95 | 0.41 | 0.11 | 0.14 |
| `Dual{Float64,2}` | Reference | 4.79 | 0.61 | 1.34 | 1.17 | 0.58 | 0.13 | 0.15 |

Milliseconds per call on the last state:

| element type | `deposit_flux!` | patch kernel | reduce kernel | `sort_by_cell!` (sorted input) |
|---|---|---|---|---|
| `Float32` | 0.42 | 0.23 | 0.08 | 0.65 |
| `Float64` | 0.54 | 0.22 | 0.09 | 0.94 |
| `Dual{Float32,2}` | 1.82 | 1.08 | 0.11 | 1.47 |
| `Dual{Float64,2}` | 2.11 | 1.18 | 0.14 | 2.13 |

Milliseconds per call of the patch kernel per work-group:

| element type | `nothing` | `1` | `8` | `64` | `512` |
|---|---|---|---|---|---|
| `Float32` | 0.23 | 0.23 | 0.23 | 0.23 | 0.24 |
| `Float64` | 0.29 | 0.23 | 0.29 | 0.29 | 0.31 |
| `Dual{Float32,2}` | 6.62 | 1.08 | 3.05 | 3.18 | 22.4 |
| `Dual{Float64,2}` | 12.6 | 1.19 | 3.1 | 6.39 | 40.9 |

Milliseconds per call of the deposition per patch layout:

| element type | `deposit_flux!` StencilMajor | patch StencilMajor | reduce StencilMajor | `deposit_flux!` CellMajor | patch CellMajor | reduce CellMajor |
|---|---|---|---|---|---|---|
| `Float32` | 0.43 | 0.23 | 0.08 | 0.96 | 0.76 | 0.12 |
| `Float64` | 0.51 | 0.23 | 0.09 | 1.07 | 0.83 | 0.12 |
| `Dual{Float32,2}` | 1.79 | 1.09 | 0.11 | 2.17 | 1.75 | 0.15 |
| `Dual{Float64,2}` | 2.08 | 1.17 | 0.14 | 2.58 | 1.97 | 0.16 |
