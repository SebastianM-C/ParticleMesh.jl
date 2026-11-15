Seconds per run of 512 steps (4 threads, Intel(R) Xeon(R) Processor @ 2.10GHz):

| element type | mode | `run!` | push | sort | deposit | window | fields | other |
|---|---|---|---|---|---|---|---|---|
| `Float32` | Reference | 3.62 | 0.81 | 1.11 | 0.8 | 0.54 | 0.05 | 0.14 |
| `Float32` | Fast | 10.1 | 0.92 | 0.0 | 7.73 | 0.65 | 0.06 | 0.24 |
| `Float64` | Reference | 3.62 | 0.77 | 1.41 | 0.82 | 0.68 | 0.07 | 0.11 |
| `Float64` | Fast | 9.4 | 0.78 | 0.0 | 7.61 | 0.72 | 0.07 | 0.18 |
| `Dual{Float32,2}` | Reference | 11.3 | 3.61 | 1.53 | 4.16 | 0.69 | 0.11 | 0.46 |
| `Dual{Float64,2}` | Reference | 10.9 | 1.96 | 2.23 | 5.09 | 1.1 | 0.14 | 0.31 |

Milliseconds per call on the last state:

| element type | `deposit_flux!` | patch kernel | reduce kernel | `sort_by_cell!` (sorted input) |
|---|---|---|---|---|
| `Float32` | 1.39 | 1.05 | 0.27 | 0.96 |
| `Float64` | 1.36 | 0.96 | 0.31 | 1.23 |
| `Dual{Float32,2}` | 6.99 | 5.58 | 0.5 | 1.55 |
| `Dual{Float64,2}` | 7.83 | 7.29 | 0.75 | 2.68 |
