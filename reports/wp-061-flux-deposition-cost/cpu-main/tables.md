Seconds per run of 512 steps (4 threads, Intel(R) Xeon(R) Processor @ 2.10GHz):

| element type | mode | `run!` | push | sort | deposit | window | fields | other |
|---|---|---|---|---|---|---|---|---|
| `Float32` | Reference | 5.12 | 0.92 | 1.22 | 1.81 | 0.6 | 0.06 | 0.15 |
| `Float32` | Fast | 10.1 | 0.92 | 0.0 | 7.84 | 0.68 | 0.07 | 0.26 |
| `Float64` | Reference | 5.79 | 0.87 | 1.55 | 2.39 | 0.8 | 0.07 | 0.14 |
| `Float64` | Fast | 9.94 | 0.78 | 0.0 | 7.73 | 0.72 | 0.07 | 0.18 |
| `Dual{Float32,2}` | Reference | 46.7 | 3.83 | 1.7 | 32.1 | 0.9 | 0.14 | 0.55 |
| `Dual{Float64,2}` | Reference | 40.1 | 2.08 | 2.49 | 32.7 | 1.14 | 0.17 | 0.39 |

Milliseconds per call on the last state:

| element type | `deposit_flux!` | patch kernel | reduce kernel | `sort_by_cell!` (sorted input) |
|---|---|---|---|---|
| `Float32` | 2.83 | 2.43 | 0.57 | 1.0 |
| `Float64` | 4.03 | 3.13 | 0.6 | 1.4 |
| `Dual{Float32,2}` | 71.6 | 62.6 | 1.27 | 1.93 |
| `Dual{Float64,2}` | 54.1 | 51.4 | 0.95 | 2.66 |

Milliseconds per call of the patch kernel per work-group:

| element type | `nothing` | `1` | `8` | `64` | `512` |
|---|---|---|---|---|---|
| `Float32` | 1.97 | 2.38 | 2.6 | 2.02 | 2.53 |
| `Float64` | 3.15 | 3.11 | 2.89 | 2.99 | 3.47 |
| `Dual{Float32,2}` | 70.4 | 10.6 | 46.5 | 62.9 | 85.8 |
| `Dual{Float64,2}` | 51.4 | 13.5 | 40.0 | 52.7 | 107.2 |
