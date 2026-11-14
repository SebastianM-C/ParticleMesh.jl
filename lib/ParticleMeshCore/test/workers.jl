# The subprocess runs of `threads_worker.jl` at 1 and 4 threads, which engine.jl, flux.jl, engine2d.jl,
# flux2d.jl, spectral.jl and momenta.jl compare against their in-process runs, and the helpers that
# read their lines. Shared by every test group (groups.toml), so a group runs on its own.

"The thread counts the files compare at; the first request starts a worker for each, side by side."
const WORKER_THREADS = (1, 4)
const WORKER_RUNS = Dict{Int,Tuple{Base.Process,String}}()
const WORKER_OUTPUT = Dict{Int,Vector{String}}()

"Start `threads_worker.jl` with `nt` threads, its output going to a file (not a pipe nobody reads yet)."
function start_worker(nt)
    get!(WORKER_RUNS, nt) do
        worker = joinpath(@__DIR__, "threads_worker.jl")
        project = dirname(Base.active_project())
        out = tempname()
        cmd = `$(Base.julia_cmd()) --startup-file=no --threads=$nt --project=$project $worker`
        (run(pipeline(cmd; stdout = out, stderr = stderr); wait = false), out)
    end
end

"The output lines of `threads_worker.jl` run with `nt` threads (cached: one subprocess per thread count)."
function worker_lines(nt)
    get!(WORKER_OUTPUT, nt) do
        foreach(start_worker, nt in WORKER_THREADS ? WORKER_THREADS : (nt,))
        proc, out = WORKER_RUNS[nt]
        wait(proc)
        success(proc) || error("threads_worker.jl with $nt threads failed: $proc")
        lines = split(read(out, String), '\n'; keepempty = false)
        rm(out; force = true)
        lines
    end
end
worker_digests(nt) = [chopprefix(l, "DIGEST ") for l in worker_lines(nt) if startswith(l, "DIGEST ")]
"The digest lines of one kind (`scan`, `sort`, `sortmatch`, `deposit`)."
of_kind(lines, kind) = filter(l -> startswith(l, kind * "|"), lines)
"Lines grouped by everything except the launch shape (the fourth field)."
function by_case(lines)
    groups = Dict{String,Vector{String}}()
    for l in lines
        f = split(l, '|')
        push!(get!(groups, join(f[[1, 2, 3]], '|'), String[]), f[5])
    end
    return groups
end
