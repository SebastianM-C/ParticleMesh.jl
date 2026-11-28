# The hardware-counter study of WP-091: which kernels of a step hold the GPU's time, and what
# limits each, from Nsight Compute counters collected through GPUDiagnostics 0.6 around a few
# steady-state steps of the optimization's cases (`workload.jl`).
#
#   julia +1.13 --project=scripts scripts/counters/study.jl probe <out dir>
#   julia +1.13 --project=scripts scripts/counters/study.jl collect <out dir> [cell ...]
#   julia +1.13 --project=scripts scripts/counters/study.jl reduce <out dir>
#
# `probe` checks that counters can be collected at all: the tool, the driver's profiling setting,
# GPUDiagnostics' known-count kernel (131072 DFMA thread instructions per launch, `probe_kernel.jl`)
# and one step of a small 2D case through the whole pipeline; a few minutes. `collect` runs each
# cell (all of `CELLS` by default) twice under ncu, both times inside the workload's profiling
# window only (`profile_from_start = false`):
#
# - `time`: one metric, gpu__time_duration.sum, with no cache flush and no clock lock, so the
#   kernels run once each, at the clocks the card holds: their durations are close to an
#   unprofiled run's and give the time shares;
# - `detail`: ncu's `detailed` section set plus WarpStateStats and SchedulerStats (speed of light,
#   occupancy, memory tables, launch statistics, every stall reason), replayed per kernel with the
#   caches flushed, at the clocks the card holds.
#
# `reduce` turns each cell's two collections into `<cell>/kernels.json`: one row per kernel group
# (`kernel_name_stem`), with the `time` share, the `detail` collection's `hw_counter_by_kernel`
# columns and limiter, and the duration-weighted mean of every raw counter of the detail export.
# `report.jl` makes the figures and tables from those files. The run's manifest (F3) is written by
# `collect` into the out dir.
include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
using GPUDiagnostics
using GPUDiagnostics: NsightCompute

"The study's cells: name => (case, element type, workload keys). `steps` is the profiled window."
const CELLS = [
    "2d-f32" => ("2d", "f32", ["steps=2"]),
    "2d-dual4" => ("2d", "dual4", ["steps=2"]),
    "3d-f32" => ("3d", "f32", ["steps=1", "tsteps=10"]),
    "3d-dual4" => ("3d", "dual4", ["steps=1", "tsteps=5"]),
]
"The probe's small cell: the 2D case at λ₀/16, one step in the window."
const PROBE_CELL = "2d-small" => ("2d", "f32", ["ppl=16", "ny=1", "tsteps=2", "steps=1"])

const WORKLOAD = joinpath(@__DIR__, "workload.jl")
julia_cmd(script, args...) = `$(Base.julia_cmd()) --startup-file=no --project=$(joinpath(REPO, "scripts")) $script $args`

"What the machine says about its profiler: ncu's version and path, the driver, the legacy profiling flag."
function tool_info()
    out = Dict{String,Any}()
    exe = hw_counter_status(NsightCompute()).executable
    out["ncu_executable"] = ismissing(exe) ? "not found" : exe
    out["ncu_version"] = ismissing(exe) ? "not found" : strip(read(ignorestatus(`$exe --version`), String))
    out["nvidia_smi"] = strip(read(ignorestatus(`nvidia-smi --query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem --format=csv,noheader`), String))
    params = "/proc/driver/nvidia/params"
    flag = isfile(params) ? filter(l -> startswith(l, "RmProfilingAdminOnly"), readlines(params)) : String[]
    out["RmProfilingAdminOnly"] = isempty(flag) ? "not readable" : only(flag)
    out["nvidia_caps"] = isdir("/dev/nvidia-caps") ? readdir("/dev/nvidia-caps") : String[]
    return out
end

"Run the two collections of one cell into `<dir>/<name>/`; their logs go to `<name>/<kind>.log`."
function collect_cell(dir, (name, (case, eltype, keys)); kinds = ("time", "detail"), timeout_s = case == "3d" ? 10_800 : 3_600)
    cdir = mkpath(joinpath(dir, name))
    for kind in kinds
        info = joinpath(cdir, "workload-$kind.toml")
        work = julia_cmd(WORKLOAD, case, eltype, info, keys...)
        opts = kind == "time" ? (; metrics = ["gpu__time_duration.sum"], cache_control = :none) :
               (; section_set = :detailed, sections = ["WarpStateStats", "SchedulerStats"])
        cmd = hw_counter_command(NsightCompute(), work; dir = cdir, name = kind, profile_from_start = false, clock_control = :none, timeout_s, opts...)
        println("study.jl: $name $kind: ", cmd)
        t = @elapsed ok = success(pipeline(cmd; stdout = joinpath(cdir, "$kind.log"), stderr = joinpath(cdir, "$kind.log"), append = true))
        println("study.jl: $name $kind: ", ok ? "done" : "FAILED", " in ", round(t; digits = 1), " s")
        ok || return false
    end
    return true
end

# The raw counters of a group: the duration-weighted mean of every numeric counter over the group's dispatches.
function raw_means(hcs)
    sums, weights = Dict{String,Float64}(), Dict{String,Float64}()
    for h in hcs, (j, c) in enumerate(h.counters), (i, d) in enumerate(h.dispatches)
        v = h.values[i, j]
        (ismissing(v) || !isfinite(v)) && continue
        w = ismissing(d.duration_s) ? 1.0 : d.duration_s
        sums[c] = get(sums, c, 0.0) + w * v
        weights[c] = get(weights, c, 0.0) + w
    end
    return Dict(c => sums[c] / weights[c] for c in keys(sums) if weights[c] > 0)
end

plain(x) = x isa Symbol ? String(x) : x isa Pair ? Dict("reason" => first(x), "cycles_per_issue" => last(x)) :
           x isa AbstractVector ? map(plain, x) : ismissing(x) ? nothing : x

"One cell's `kernels.json`: the rows of the detail collection by kernel group, with the time collection's shares."
function reduce_cell(cdir)
    tcsv, dcsv = joinpath(cdir, "time_ncu.csv"), joinpath(cdir, "detail_ncu.csv")
    trows = hw_counter_by_kernel(tcsv)
    tshare = Dict(r.group => (share = r.share, time_s = r.time_s, dispatches = r.dispatches) for r in trows)
    hcs = hw_counter_kernels(dcsv)
    byg = Dict{String,Vector{HWCounters}}()
    for h in hcs
        push!(get!(() -> HWCounters[], byg, kernel_name_stem(h.kernel)), h)
    end
    rows = map(hw_counter_by_kernel(hcs)) do r
        d = Dict{String,Any}(String(k) => plain(v) for (k, v) in pairs(r))
        t = get(tshare, r.group, nothing)
        d["time_share"] = t === nothing ? nothing : plain(t.share)
        d["time_s_unflushed"] = t === nothing ? nothing : plain(t.time_s)
        d["dispatches_time"] = t === nothing ? nothing : t.dispatches
        d["raw"] = raw_means(byg[r.group])
        d["registers"] = maximum(get(dd.resources, "registers", missing) for h in byg[r.group] for dd in h.dispatches)
        d["grid_size"] = plain(maximum(get(dd.resources, "grid_size", missing) for h in byg[r.group] for dd in h.dispatches))
        d["workgroup_size"] = plain(maximum(get(dd.resources, "workgroup_size", missing) for h in byg[r.group] for dd in h.dispatches))
        d
    end
    sort!(rows; by = r -> something(r["time_share"], -1.0), rev = true)
    missing_in_detail = [g for g in keys(tshare) if !haskey(byg, g)]
    info = Dict(k => TOML.parsefile(joinpath(cdir, "workload-$k.toml")) for k in ("time", "detail") if isfile(joinpath(cdir, "workload-$k.toml")))
    out = Dict{String,Any}("cell" => basename(cdir), "rows" => rows, "workload" => info, "groups_only_in_time" => missing_in_detail,
                           "total_time_s_unflushed" => sum(r -> something(r.time_s, 0.0), trows))
    open(io -> JSON.print(io, out, 1), joinpath(cdir, "kernels.json"), "w")
    return out
end

"A short table of a reduced cell on stdout: share, throughputs, occupancy, top stall, limiter."
function print_cell(out; n = 15)
    println("\n", out["cell"], ": ", length(out["rows"]), " kernel groups")
    f(x) = x === nothing ? "-" : string(round(100 * x; digits = 1))
    for r in first(out["rows"], n)
        st = isempty(r["top_stalls"]) ? "-" : r["top_stalls"][1]["reason"]
        println(rpad(r["group"], 42), " share ", lpad(f(r["time_share"]), 5), " %  sm ", lpad(f(get(r, "nvidia_sm_throughput", nothing)), 5),
                "  mem ", lpad(f(get(r, "nvidia_memory_throughput", nothing)), 5), "  occ ", lpad(f(get(r, "nvidia_active_occupancy", nothing)), 5),
                "  regs ", lpad(string(r["registers"]), 3), "  stall ", rpad(st, 16), " ", something(r["limiter"], "-"))
    end
end

function write_run_manifest(dir, cells, what)
    gpu = strip(read(ignorestatus(`nvidia-smi --query-gpu=name --format=csv,noheader`), String))
    run_dir("counters", @__FILE__; dir, prefix = "counters", work_package = "WP-091", backend = "KernelAbstractions CUDA",
            params = Dict{String,Any}("command" => what, "cells" => [Dict("name" => n, "case" => c, "eltype" => e, "keys" => k) for (n, (c, e, k)) in cells]),
            extra_inputs = [WORKLOAD, joinpath(@__DIR__, "probe_kernel.jl"), joinpath(REPO, "scripts", "optim", "trapped3d_case.jl"),
                            joinpath(REPO, "scripts", "optim", "trapped2d_case.jl"), joinpath(REPO, "scripts", "optim", "born2d_case.jl")],
            extra = merge(Dict{String,Any}("gpu" => gpu, "gpudiagnostics" => string(pkgversion(GPUDiagnostics))), tool_info()))
end

function probe(dir)
    mkpath(dir)
    info = tool_info()
    foreach(((k, v),) -> println("study.jl probe: ", k, " = ", v), info)
    # the known count: launches 2 and 3 of counter_probe, 131072 DFMA thread instructions each
    kdir = mkpath(joinpath(dir, "known"))
    cmd = hw_counter_command(NsightCompute(), julia_cmd(joinpath(@__DIR__, "probe_kernel.jl")); set = :fp64, dir = kdir, name = "probe",
                             kernel = "counter_probe", launch_skip = 1, launch_count = 2, timeout_s = 600)
    ok = success(pipeline(cmd; stdout = joinpath(kdir, "probe.log"), stderr = joinpath(kdir, "probe.log")))
    println("study.jl probe: known-count run ", ok ? "ran" : "FAILED (see $(joinpath(kdir, "probe.log")) and probe_ncu.csv)")
    if ok
        hc = hw_counters(kdir; name = "probe", kernel = "counter_probe", slots = 4096 * 32,
                         required_metrics = COUNTER_SETS[:nvidia][:fp64].metrics)
        dfma = hc["smsp__sass_thread_inst_executed_op_dfma_pred_on.sum"]
        println("study.jl probe: DFMA per launch ", dfma, " (expected 131072 each): ", all(==(131072), dfma) ? "ok" : "WRONG")
    end
    write_run_manifest(dir, [PROBE_CELL], "probe")
    collect_cell(dir, PROBE_CELL; timeout_s = 900) || return println("study.jl probe: the small case FAILED")
    print_cell(reduce_cell(joinpath(dir, first(PROBE_CELL))); n = 40)
end

function main(args)
    isempty(args) && error("usage: study.jl probe|collect|reduce <out dir> [cell ...]")
    cmd, dir = args[1], abspath(args[2])
    if cmd == "probe"
        probe(dir)
    elseif cmd == "collect"
        names = length(args) > 2 ? args[3:end] : first.(CELLS)
        cells = [c for c in CELLS if first(c) in names]
        length(cells) == length(names) || error("study.jl: unknown cell in $names (cells: $(first.(CELLS)))")
        mkpath(dir)
        write_run_manifest(dir, cells, "collect")
        for c in cells
            collect_cell(dir, c) && print_cell(reduce_cell(joinpath(dir, first(c))))
        end
    elseif cmd == "reduce"
        for c in sort(readdir(dir)), f in (joinpath(dir, c, "detail_ncu.csv"),)
            isfile(f) && print_cell(reduce_cell(joinpath(dir, c)))
        end
    else
        error("study.jl: the command is probe, collect or reduce, got $cmd")
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
