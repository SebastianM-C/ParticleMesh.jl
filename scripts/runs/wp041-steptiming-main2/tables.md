Seconds per run of 512 steps (16 threads, AMD EPYC 9655 96-Core Processor):

| element type | mode | `run!` | push | sort | deposit | window | fields | other |
|---|---|---|---|---|---|---|---|---|
| `Float32` | Reference | 1.1 | 0.2 | 0.31 | 0.22 | 0.12 | 0.02 | 0.04 |
| `Float32` | Fast | 2.32 | 0.19 | 0.0 | 1.78 | 0.14 | 0.03 | 0.05 |
| `Float64` | Reference | 1.12 | 0.2 | 0.38 | 0.21 | 0.16 | 0.03 | 0.05 |
| `Float64` | Fast | 2.21 | 0.19 | 0.0 | 1.64 | 0.2 | 0.03 | 0.04 |
| `Dual{Float32,2}` | Reference | 2.89 | 0.8 | 0.54 | 1.17 | 0.2 | 0.05 | 0.14 |
| `Dual{Float64,2}` | Reference | 3.59 | 0.65 | 0.77 | 1.51 | 0.37 | 0.07 | 0.16 |

Milliseconds per call on the last state:

| element type | `deposit_flux!` | patch kernel | reduce kernel | `sort_by_cell!` (sorted input) |
|---|---|---|---|---|
| `Float32` | 0.4 | 0.29 | 0.1 | 0.39 |
| `Float64` | 0.4 | 0.28 | 0.1 | 0.5 |
| `Dual{Float32,2}` | 2.14 | 1.94 | 0.16 | 0.63 |
| `Dual{Float64,2}` | 2.48 | 2.13 | 0.33 | 0.9 |

Milliseconds per call of the patch kernel per work-group:

| element type | `nothing` | `1` | `8` | `64` | `512` |
|---|---|---|---|---|---|
| `Float32` | 0.4 | 0.36 | 0.46 | 0.48 | 0.48 |
| `Float64` | 0.47 | 0.3 | 0.47 | 0.35 | 0.36 |
| `Dual{Float32,2}` | 10.7 | 2.0 | 4.84 | 5.0 | 39.0 |
| `Dual{Float64,2}` | 18.7 | 2.16 | 5.02 | 9.08 | 66.9 |

Milliseconds per call of the deposition per patch layout:

| element type | `deposit_flux!` StencilMajor | patch StencilMajor | reduce StencilMajor | `deposit_flux!` CellMajor | patch CellMajor | reduce CellMajor |
|---|---|---|---|---|---|---|
| `Float32` | 0.57 | 0.33 | 0.12 | 1.31 | 1.13 | 0.18 |
| `Float64` | 0.39 | 0.28 | 0.11 | 1.41 | 1.24 | 0.17 |
| `Dual{Float32,2}` | 2.11 | 1.96 | 0.16 | 3.24 | 2.99 | 0.2 |
| `Dual{Float64,2}` | 2.42 | 2.09 | 0.32 | 3.49 | 3.01 | 0.31 |
