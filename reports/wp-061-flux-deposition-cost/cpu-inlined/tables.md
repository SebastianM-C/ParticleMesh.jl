Seconds per run of 512 steps (4 threads, Intel(R) Xeon(R) Processor @ 2.10GHz):

| element type | mode | `run!` | push | sort | deposit | window | fields | other |
|---|---|---|---|---|---|---|---|---|
| `Float32` | Reference | 4.6 | 0.86 | 1.12 | 1.54 | 0.67 | 0.06 | 0.14 |
| `Float32` | Fast | 9.69 | 0.9 | 0.0 | 7.72 | 0.7 | 0.06 | 0.25 |
| `Float64` | Reference | 5.0 | 0.77 | 1.29 | 1.9 | 0.78 | 0.06 | 0.11 |
| `Float64` | Fast | 9.65 | 0.77 | 0.0 | 7.53 | 0.86 | 0.07 | 0.14 |
| `Dual{Float32,2}` | Reference | 13.0 | 3.62 | 1.59 | 5.94 | 0.81 | 0.11 | 0.48 |
| `Dual{Float64,2}` | Reference | 14.5 | 2.01 | 2.32 | 7.82 | 1.12 | 0.14 | 0.32 |

Milliseconds per call on the last state:

| element type | `deposit_flux!` | patch kernel | reduce kernel | `sort_by_cell!` (sorted input) |
|---|---|---|---|---|
| `Float32` | 2.61 | 1.92 | 0.46 | 0.97 |
| `Float64` | 3.4 | 2.65 | 0.62 | 1.22 |
| `Dual{Float32,2}` | 7.97 | 7.38 | 0.8 | 1.52 |
| `Dual{Float64,2}` | 10.8 | 8.62 | 0.9 | 2.58 |
