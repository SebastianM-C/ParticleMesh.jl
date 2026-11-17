# Deposition study: prototypes and their measurements

Prototype kernels for a study of the sort and the flux deposition (what could replace or
complement the engine's mechanisms in 2D, and what a 3D deposition could be), with the checks and
the benchmarks behind the study's numbers. None of this is engine code: nothing in `lib/` uses it,
and the engine's tests do not cover it.

| File | What it is |
|---|---|
| `proto.jl` | The kernels, generic in the element type: the incremental (neighbour-bounded) sort in a gather and a scatter form; the flux of one particle in the engine's 6-wide cell frame or in its own 4-wide frame, in 2D and 3D; atomics per particle; per-tile buffers with a per-node reduction; coloured tiles; three variants of the engine's per-cell patch; and a 3D per-cell patch, one component per launch |
| `common.jl` | Synthetic states on N axes, host-side references (ρ, fold, continuity residual), the timer |
| `check.jl` | Every variant against the engine (2D, to roundoff; the incremental sorts bitwise, at both N) and against discrete continuity (2D and 3D), in Float32, Float64 and `Dual{Float32,3}` on the CPU backend |
| `bench2d.jl` | The engine and the variants on states of the WP-045 case at three sizes, on the CPU or CUDA |
| `bench3d.jl` | The sorts and the 3D variants on a synthetic 3D state, on the CPU or CUDA |
| `phases2d.jl` | The engine's step per phase (push, sort, deposit, window, fields) on the sizes of `bench2d.jl`, on the CPU or CUDA |

```
julia +1.13 -t 4 --project=scripts scripts/deposition/check.jl
julia +1.13 -t 4 --project=scripts scripts/deposition/bench2d.jl cpu S,M f32,f64,d32,d64
julia +1.13 -t 4 --project=scripts scripts/deposition/bench3d.jl cpu 48x48x48 8 f32,f64,d32
```

Each benchmark writes its numbers (`bench2d.json`, `bench3d.json` or `phases2d.json`), a table
(`tables.md`) and a manifest into `scripts/runs/deposition-<name>-<time>/`. The numbers are single-machine
measurements and have not been independently reviewed.
