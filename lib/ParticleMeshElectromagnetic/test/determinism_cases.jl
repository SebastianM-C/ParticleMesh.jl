# The short runs of T-218, shared by the test and its subprocess worker: every case, every
# element type of the A8 matrix, Reference mode, 64 particles per cell, one time unit (93 to 131
# steps, enough for particles to cross cells and the seam).

const DETERMINISM_T_END = 1

function determinism_digests(; workgroupsize = nothing, be = CPU())
    out = Tuple{String,DataType,UInt64}[]
    for name in ("landau", "two-stream", "weibel"), T in TYPES
        st, p, prob, _ = case_state(name, T; mode = Reference(), workgroupsize, ppc = 64, t_end = DETERMINISM_T_END, be)
        run!(st, p, prob.nsteps)
        push!(out, (name, T, state_digest(st)))
    end
    return out
end

"The thread counts of the worker runs; the first request starts one worker for each, side by side."
const DETERMINISM_THREADS = (1, 4)
const DETERMINISM_RUNS = Dict{Int,Tuple{Base.Process,String}}()
const DETERMINISM_WORKER_OUTPUT = Dict{Int,String}()

"Start determinism_worker.jl at `nt` threads, its output going to a file (not a pipe nobody reads yet)."
function start_determinism_worker(nt)
    get!(DETERMINISM_RUNS, nt) do
        worker = joinpath(@__DIR__, "determinism_worker.jl")
        out = tempname()
        cmd = `$(Base.julia_cmd()) --startup-file=no --threads=$nt --project=$(dirname(Base.active_project())) $worker`
        (run(pipeline(cmd; stdout = out, stderr = stderr); wait = false), out)
    end
end

"The output of determinism_worker.jl at `nt` threads, run once per suite (T-218, T-355 and the 2D files read it)."
worker_output(nt) = get!(DETERMINISM_WORKER_OUTPUT, nt) do
    foreach(start_determinism_worker, nt in DETERMINISM_THREADS ? DETERMINISM_THREADS : (nt,))
    proc, out = DETERMINISM_RUNS[nt]
    wait(proc)
    success(proc) || error("determinism_worker.jl with $nt threads failed: $proc")
    s = read(out, String)
    rm(out; force = true)
    s
end
