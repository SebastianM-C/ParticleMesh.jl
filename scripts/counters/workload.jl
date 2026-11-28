# The workload of the hardware-counter study (WP-091): one case of the trapped-beam optimization
# on CUDA, run unprofiled to a step inside its run, then a few steps timed, then a few steps
# inside GPUDiagnostics' profiling window (`hw_counter_window`), which an Nsight Compute started
# with `profile_from_start = false` profiles and nothing else does. `study.jl` starts it under ncu;
# without ncu it is a plain run that prints its timing.
#
#   julia +1.13 --project=scripts scripts/counters/workload.jl <case> <eltype> <out.toml> [key=value ...]
#
# <case>    2d: WP-068's case (`trapped2d_case.jl`) at λ₀/32, Δy = π/8, the compensated gather;
#           3d: WP-084's case (`trapped3d_case.jl`) at λ₀/16, Δy = Δz = π/4, its GPU defaults.
# <eltype>  f32 (Float32) or dual4 (Dual{Float32,4}: duals in the four free directions, as the
#           optimization's gradient runs).
# keys      at = the window's first step as a fraction of the run's steps (default 0.6), steps =
#           steps in the window (2), tsteps = timed steps before it (20), ppl, ny (2D), nt (3D),
#           point = born (WP-068's born-only optimum, default) or base (TCASE.x0).
#
# The window starts after the timed steps. <out.toml> gets the case, the steps, the seconds per
# step of the timed steps and the device's name.
include(joinpath(@__DIR__, "..", "optim", "trapped3d_case.jl"))
using CUDA
using GPUDiagnostics: hw_counter_window
using ForwardDiff: ForwardDiff
using TOML

# `PM_COUNTERS_BACKEND=cpu` runs the same steps on the CPU without the window, to check the script.
const ON_CPU = get(ENV, "PM_COUNTERS_BACKEND", "cuda") == "cpu"
const BE = ON_CPU ? CPU() : CUDABackend()

const POINTS = (born = (7.762, 7.348, 110.0, 0.1), base = (TCASE.x0[1], TCASE.x0[2], TCASE.x0[3], TCASE.x0[5]))

"The key=value arguments after the first three as a Dict of strings."
parse_keys(args) = Dict(String(first(split(a, "="; limit = 2))) => String(last(split(a, "="; limit = 2))) for a in args)

"The four free parameters in the element type: plain Float32, or Float32 duals in the four free directions."
function free_args(point, eltype)
    eltype == "f32" && return Float32.(point)
    eltype == "dual4" && return ntuple(k -> ForwardDiff.Dual{Nothing}(Float32(point[k]), ntuple(j -> Float32(j == k), 4)), 4)
    error("workload.jl: the element type is f32 or dual4, got $eltype")
end

"The state at its start, its parameters and the problem of the case."
function case_start(case, eltype, keys)
    free = free_args(getfield(POINTS, Symbol(get(keys, "point", "born"))), eltype)
    if case == "2d"
        ppl = parse(Int, get(keys, "ppl", "32"))
        ny = parse(Int, get(keys, "ny", "2"))
        x = trapped_point(free)
        prob = trapped_problem(x...; energy = Float32(budget()), ppl, ny, float = Float32)
        return start_of(prob; ppl, ny, backend = BE, gather = CompensatedGather())
    elseif case == "3d"
        ppl = parse(Int, get(keys, "ppl", "16"))
        nt = parse(Int, get(keys, "nt", "1"))
        return trapped3d_start(free; backend = BE, ppl, nt)
    end
    error("workload.jl: the case is 2d or 3d, got $case")
end

function main(args)
    length(args) >= 3 || error("usage: workload.jl <case> <eltype> <out.toml> [key=value ...]")
    case, eltype, out = args[1], args[2], args[3]
    keys = parse_keys(args[4:end])
    st, p, prob = case_start(case, eltype, keys)
    at = parse(Float64, get(keys, "at", "0.6"))
    steps = parse(Int, get(keys, "steps", "2"))
    tsteps = parse(Int, get(keys, "tsteps", "20"))
    k0 = max(1, round(Int, at * prob.nsteps))
    k0 + tsteps + steps - 1 <= prob.nsteps || error("workload.jl: the window ends after the run's $(prob.nsteps) steps")
    run!(st, p, k0 - 1)
    KernelAbstractions.synchronize(BE)
    t = @elapsed begin
        run!(st, p, tsteps; first = k0)
        KernelAbstractions.synchronize(BE)
    end
    kw = k0 + tsteps
    if ON_CPU
        run!(st, p, steps; first = kw)
    else
        hw_counter_window(BE) do
            run!(st, p, steps; first = kw)
        end
    end
    ionization_lost(st) == 0 || error("$(ionization_lost(st)) births found no free slot")
    alive = sum(s -> count(==(ALIVE), Array(s.status)), (st.species..., ionization_electrons(st)))
    info = Dict{String,Any}("case" => case, "eltype" => eltype, "nsteps" => prob.nsteps, "first_timed_step" => k0, "timed_steps" => tsteps,
                            "seconds_per_step" => t / tsteps, "window_first_step" => kw, "window_steps" => steps,
                            "cells" => collect(st.grid.n), "particles_alive" => alive, "gpu" => ON_CPU ? "none" : CUDA.name(CUDA.device()),
                            "keys" => keys)
    mkpath(dirname(abspath(out)))
    open(io -> TOML.print(io, info), out, "w")
    println("workload.jl: ", case, " ", eltype, ": ", round(1e3 * t / tsteps; digits = 3), " ms per step at step ", k0, " of ", prob.nsteps,
            "; window at steps ", kw, " to ", kw + steps - 1)
    return info
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
