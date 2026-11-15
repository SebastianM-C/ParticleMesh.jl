Seconds per run of 512 steps (4 threads, Intel(R) Xeon(R) Processor @ 2.10GHz):

| element type | mode | `run!` | push | sort | deposit | window | fields | other |
|---|---|---|---|---|---|---|---|---|
| `Float32` | Reference | 4.73 | 0.91 | 1.26 | 1.67 | 0.69 | 0.06 | 0.14 |
| `Float32` | Fast | 10.2 | 0.94 | 0.0 | 8.02 | 0.7 | 0.06 | 0.23 |
| `Float64` | Reference | 5.49 | 0.83 | 1.47 | 2.17 | 0.63 | 0.07 | 0.12 |
| `Float64` | Fast | 9.56 | 0.75 | 0.0 | 7.47 | 0.73 | 0.07 | 0.13 |
| `Dual{Float32,2}` | Reference | 37.1 | 3.71 | 1.72 | 30.1 | 0.83 | 0.13 | 0.51 |
| `Dual{Float64,2}` | Reference | 38.6 | 2.01 | 2.58 | 31.6 | 1.27 | 0.17 | 0.36 |

Milliseconds per call on the last state:

| element type | `deposit_flux!` | patch kernel | reduce kernel | `sort_by_cell!` (sorted input) |
|---|---|---|---|---|
| `Float32` | 2.82 | 1.91 | 0.46 | 1.02 |
| `Float64` | 3.93 | 3.05 | 0.64 | 1.27 |
| `Dual{Float32,2}` | 53.2 | 51.8 | 0.81 | 1.53 |
| `Dual{Float64,2}` | 54.2 | 51.5 | 1.02 | 2.83 |
