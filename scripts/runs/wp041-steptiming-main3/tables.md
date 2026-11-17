Seconds per run of 512 steps (16 threads, AMD EPYC 9655 96-Core Processor):

| element type | mode | `run!` | push | sort | deposit | window | fields | other |
|---|---|---|---|---|---|---|---|---|
| `Float32` | Reference | 0.98 | 0.18 | 0.34 | 0.19 | 0.11 | 0.02 | 0.04 |
| `Float64` | Reference | 1.07 | 0.19 | 0.34 | 0.22 | 0.13 | 0.03 | 0.04 |
| `Dual{Float32,2}` | Reference | 2.55 | 0.63 | 0.49 | 0.86 | 0.18 | 0.05 | 0.13 |
| `Dual{Float64,2}` | Reference | 3.12 | 1.01 | 0.74 | 1.25 | 0.29 | 0.07 | 0.14 |

Milliseconds per call on the last state:

| element type | `deposit_flux!` | patch kernel | reduce kernel | `sort_by_cell!` (sorted input) |
|---|---|---|---|---|
| `Float32` | 0.55 | 0.39 | 0.16 | 0.36 |
| `Float64` | 0.45 | 0.29 | 0.1 | 0.52 |
| `Dual{Float32,2}` | 1.6 | 1.43 | 0.16 | 0.61 |
| `Dual{Float64,2}` | 1.97 | 1.61 | 0.31 | 0.86 |

Milliseconds per call of the patch kernel per work-group:

| element type | `nothing` | `1` | `8` | `64` | `512` |
|---|---|---|---|---|---|
| `Float32` | 0.32 | 0.28 | 0.49 | 0.34 | 0.46 |
| `Float64` | 0.42 | 0.32 | 0.42 | 0.52 | 0.5 |
| `Dual{Float32,2}` | 9.04 | 1.44 | 4.45 | 4.75 | 38.5 |
| `Dual{Float64,2}` | 18.5 | 1.68 | 4.86 | 9.19 | 60.6 |

Milliseconds per call of the deposition per patch layout:

| element type | `deposit_flux!` StencilMajor | patch StencilMajor | reduce StencilMajor | `deposit_flux!` CellMajor | patch CellMajor | reduce CellMajor |
|---|---|---|---|---|---|---|
| `Float32` | 0.57 | 0.26 | 0.1 | 1.25 | 1.06 | 0.18 |
| `Float64` | 0.42 | 0.3 | 0.1 | 1.43 | 1.19 | 0.18 |
| `Dual{Float32,2}` | 1.62 | 1.44 | 0.15 | 2.62 | 2.39 | 0.21 |
| `Dual{Float64,2}` | 2.01 | 1.66 | 0.33 | 3.09 | 2.65 | 0.3 |
