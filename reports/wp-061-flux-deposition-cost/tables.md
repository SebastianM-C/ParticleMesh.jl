Seconds per run of 512 steps and in its deposit phase (4 threads, Intel(R) Xeon(R) Processor @ 2.10GHz); the runs: main = `optim-step-timing-cpu-20261115T045557` at `e141e09`, branch = `optim-step-timing-cpu-20261115T050338` at `382894a`.

| element type | mode | `run!` main | `run!` branch | deposit main | deposit branch |
|---|---|---|---|---|---|
| `Float32` | Reference | 5.12 | 4.29 | 1.81 | 0.96 |
| `Float32` | Fast | 10.1 | 10.5 | 7.84 | 8.71 |
| `Float64` | Reference | 5.79 | 4.03 | 2.39 | 0.99 |
| `Float64` | Fast | 9.94 | 10.5 | 7.73 | 7.92 |
| `Dual{Float32,2}` | Reference | 46.7 | 11.4 | 32.1 | 4.99 |
| `Dual{Float64,2}` | Reference | 40.1 | 12.0 | 32.7 | 5.06 |

Milliseconds per call of the deposition's kernels on the last state:

| element type | patch main | patch branch | reduce main | reduce branch |
|---|---|---|---|---|
| `Float32` | 2.43 | 1.29 | 0.57 | 0.27 |
| `Float64` | 3.13 | 0.98 | 0.6 | 0.33 |
| `Dual{Float32,2}` | 62.6 | 6.28 | 1.27 | 0.6 |
| `Dual{Float64,2}` | 51.4 | 8.23 | 0.95 | 0.78 |
