Seconds per run of 512 steps (4 threads, Intel(R) Xeon(R) Processor @ 2.10GHz):

| element type | mode | `run!` | push | sort | deposit | window | fields | other |
|---|---|---|---|---|---|---|---|---|
| `Float32` | Reference | 4.29 | 0.9 | 1.16 | 0.96 | 0.67 | 0.06 | 0.15 |
| `Float32` | Fast | 10.5 | 1.02 | 0.0 | 8.71 | 0.75 | 0.07 | 0.3 |
| `Float64` | Reference | 4.03 | 0.86 | 1.36 | 0.99 | 0.78 | 0.07 | 0.12 |
| `Float64` | Fast | 10.5 | 0.81 | 0.0 | 7.92 | 0.76 | 0.08 | 0.16 |
| `Dual{Float32,2}` | Reference | 11.4 | 3.97 | 1.69 | 4.99 | 0.76 | 0.12 | 0.51 |
| `Dual{Float64,2}` | Reference | 12.0 | 1.94 | 2.42 | 5.06 | 1.0 | 0.14 | 0.31 |

Milliseconds per call on the last state:

| element type | `deposit_flux!` | patch kernel | reduce kernel | `sort_by_cell!` (sorted input) |
|---|---|---|---|---|
| `Float32` | 1.65 | 1.29 | 0.27 | 0.99 |
| `Float64` | 1.35 | 0.98 | 0.33 | 1.24 |
| `Dual{Float32,2}` | 8.43 | 6.28 | 0.6 | 1.78 |
| `Dual{Float64,2}` | 8.1 | 8.23 | 0.78 | 2.52 |

Milliseconds per call of the patch kernel per work-group:

| element type | `nothing` | `1` | `8` | `64` | `512` |
|---|---|---|---|---|---|
| `Float32` | 1.25 | 1.06 | 1.62 | 1.4 | 1.29 |
| `Float64` | 1.53 | 1.11 | 2.06 | 1.58 | 1.79 |
| `Dual{Float32,2}` | 61.2 | 5.53 | 34.8 | 44.8 | 60.8 |
| `Dual{Float64,2}` | 63.5 | 7.85 | 36.8 | 44.4 | 103.1 |

Milliseconds per call of the deposition per patch layout:

| element type | `deposit_flux!` StencilMajor | patch StencilMajor | reduce StencilMajor | `deposit_flux!` CellMajor | patch CellMajor | reduce CellMajor |
|---|---|---|---|---|---|---|
| `Float32` | 1.44 | 1.08 | 0.26 | 2.48 | 1.89 | 0.46 |
| `Float64` | 1.9 | 1.17 | 0.36 | 3.86 | 3.12 | 0.74 |
| `Dual{Float32,2}` | 6.25 | 5.52 | 0.73 | 9.67 | 9.47 | 0.91 |
| `Dual{Float64,2}` | 9.57 | 8.18 | 0.88 | 12.2 | 11.1 | 0.98 |
