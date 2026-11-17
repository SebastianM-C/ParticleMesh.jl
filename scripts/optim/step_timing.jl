# Where a run of the WP-045 case (wake2d_case.jl) spends its time: seconds per phase of the
# step, per element type, and the 2D Reference flux deposition's two kernels on the run's last
# state (WP-061, #442).
#
#   julia +1.13 -t 4 --project=scripts scripts/optim/step_timing.jl [cpu|cuda] [fast] [resort]
#
# The phases are those of the module's `step!`, which `timed_step!` repeats with a timer around
# each (on a device the timer waits for the phase, `KernelAbstractions.synchronize`); a test of the
# module holds `step!`'s bits, this script only its order, and it prints the plain `run!` time of
# the same run next to the sum of its own phases. Every case runs twice in one process and the
# second run is timed, so that no compilation is in the numbers (the moving window's kernels first
# run at the first shift). `fast` adds the Fast mode for the plain element types; `resort` runs
# the step with the sort from the last order (`EMNumerics(; resort = true)`) instead of the full
# sort of every step.
#
# To compare trees, run the script in each of them (copy this file into the other tree if it is
# not there: each run's manifest records its tree's commit and the hashes of its inputs), then
#
#   julia +1.13 --project=scripts scripts/optim/step_timing.jl compare <label>=<run directory> ...
#
# which reads the runs' sidecars and writes one table and one figure of them, in the order given.
include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
include(joinpath(@__DIR__, "wake2d_case.jl"))
using ParticleMeshElectromagnetic
using ForwardDiff: Dual

const PME = ParticleMeshElectromagnetic
const PMC = ParticleMesh.ParticleMeshCore
const PHASES = (:push, :sort, :deposit, :window, :fields, :other)

const BACKEND_NAME = isempty(ARGS) ? "cpu" : ARGS[1]
if BACKEND_NAME == "cuda"
    @eval using CUDACore: CUDACore, CUDABackend
    @eval const BE = CUDABackend()
elseif BACKEND_NAME in ("cpu", "compare")
    const BE = CPU()
else
    error("step_timing.jl: the first argument is cpu, cuda or compare, got $BACKEND_NAME")
end

"The element types of the run: the label and the case's two free parameters (duration, focus) in that type."
function element_cases()
    d2(T) = (Dual{Nothing}(T(7), one(T), zero(T)), Dual{Nothing}(T(100), zero(T), one(T)))
    return (("Float32", (7.0f0, 100.0f0)), ("Float64", (7.0, 100.0)), ("Dual{Float32,2}", d2(Float32)), ("Dual{Float64,2}", d2(Float64)))
end

"Run `f()`, wait for the device, and add the seconds to `times[phase]`."
function timed!(f, times, phase)
    t0 = time_ns()
    f()
    KernelAbstractions.synchronize(BE)
    times[phase] += (time_ns() - t0) / 1e9
    return nothing
end

"The module's `step!` (lib/ParticleMeshElectromagnetic/src/step.jl) with a timer around each phase."
function timed_step!(st, p, k, times)
    PME.check_params(st, p)
    (; numerics, grid, fields, dt) = st
    wgs = numerics.workgroupsize
    n = Int32.(grid.n)
    g = Int32(guards(grid))
    dt_dx = map(h -> dt / h, grid.dx)
    window = PME.window_frame(st)
    T = eltype(st.ρ)
    timed!(times, :other) do
        PME.observe!(st.recorder, st, k)
        PME.inject_witness!(PME.witness_frame(st), p.witness, grid, k, wgs)
    end
    timed!(times, :push) do
        for s in (st.species..., PME.witness_species(st)...)
            PME.launcher(PME.push_kernel!, get_backend(s), wgs)(s, fields, numerics.pusher, numerics.shape, g, s.q / s.m * dt / 2,
                                                                dt_dx, n; ndrange = capacity(s))
        end
    end
    timed!(() -> PME.retire_x!(window, st), times, :other)
    timed!(times, :sort) do
        for (s, ws) in zip(st.species, st.ws)
            PME.sorts(st, s) && PMC.resort_by_cell!(s, grid, ws; workgroupsize = wgs)   # as current! does
        end
    end
    timed!(times, :other) do
        fill!(fields.Jx, zero(T)); fill!(fields.Jy, zero(T)); fill!(fields.Jz, zero(T))
    end
    timed!(times, :deposit) do
        for (s, ws) in zip(st.species, st.ws)
            deposit_flux!(PME.deposit_mode(st, s), fields.Jx, fields.Jy, fields.Jz, s, numerics.shape, PME.rho_stagger(Val(2)), grid, ws;
                          dt = st.dt, PME.transverse_currents(Val(2))..., workgroupsize = wgs)
        end
    end
    timed!(times, :other) do
        for J in (fields.Jx, fields.Jy, fields.Jz)
            PME.zero_x_edges!(window, J; workgroupsize = wgs)
            fold_guards!(J, grid; workgroupsize = wgs)
        end
        PME.filter_current!(numerics.filter, fields, grid, st.ρ; workgroupsize = wgs, edges = window)
        PME.retire!(PME.open_absorber(st), st)
    end
    timed!(() -> PME.solve_fields!(numerics.solver, fields, grid, dt, st.modifiers, p, k; workgroupsize = wgs), times, :fields)
    timed!(() -> PME.advance_window!(window, st, p, k), times, :window)
    timed!(() -> PME.reads_kinetic(st.recorder, k + 1) && PME.kinetic!(st), times, :other)
    return st
end

"Seconds per call of `f()` on the device: the least of `reps` calls after one warm-up call."
function least(f; reps = 15)
    f(); KernelAbstractions.synchronize(BE)
    return minimum(1:reps) do _
        t0 = time_ns()
        f(); KernelAbstractions.synchronize(BE)
        (time_ns() - t0) / 1e9
    end
end

"""
The 2D Reference flux deposition on the state `st` (sorted, after its run): seconds of
`deposit_flux!`, of its patch kernel and of its reduce kernel as `_deposit_flux_2d!` launches
them (lib/ParticleMeshCore/src/flux2d.jl), and of `sort_by_cell!` on the sorted species; the
patch kernel with fixed work-groups next to the backend's default (`by_workgroup`); and, in a
tree whose workspace takes a patch layout, the three deposition times per layout (`by_layout`).
"""
function deposition_kernels(st)
    (; numerics, grid, fields) = st
    s = st.species[1]
    shape, stg = numerics.shape, PME.rho_stagger(Val(2))
    qz = PME.transverse_currents(Val(2)).quantity_z
    wgs = numerics.workgroupsize
    sx = map(h -> h / st.dt, cellsize(grid))
    n = Int32.(ncells(grid))
    M = PMC.flux_widths(shape, stg)
    r = map(x -> PMC.flux_reach(shape, x), stg)
    lo, hi = (r[1][1], r[2][1]), (r[1][2], r[2][2])
    layouts = isdefined(PMC, :flux_patch_layout)      # the kernels' arguments differ between the trees this script compares
    patch_args(ws) = layouts ? (ws.flux_patch, s, ws.seg_start, ws.seg_stop, shape, stg, n, sx, qz, ws.flux_layout, Val(prod(M))) :
                     (ws.flux_patch, s, ws.seg_start, ws.seg_stop, shape, stg, n, sx, qz, Val(prod(M)))
    reduce_args(ws) = layouts ? (fields.Jx, fields.Jy, fields.Jz, ws.flux_patch, ws.flux_layout, n, Int32(guards(grid)), lo, hi, Int32.(M)) :
                      (fields.Jx, fields.Jy, fields.Jz, ws.flux_patch, n, Int32(guards(grid)), lo, hi, Int32.(M))
    patch_launch = isdefined(PMC, :item_launcher) ? PMC.item_launcher : PMC.launcher
    function three(ws)
        whole = least() do
            deposit_flux!(Reference(), fields.Jx, fields.Jy, fields.Jz, s, shape, stg, grid, ws; dt = st.dt, quantity_z = qz, workgroupsize = wgs)
        end
        patch = least(() -> patch_launch(PMC.flux_patch_2d_kernel!, BE, wgs)(patch_args(ws)...; ndrange = prod(n)))
        reduce = least(() -> PMC.launcher(PMC.flux_reduce_2d_kernel!, BE, wgs)(reduce_args(ws)...; ndrange = size(fields.Jx)))
        return Dict{String,Any}("deposit_flux" => whole, "patch_kernel" => patch, "reduce_kernel" => reduce)
    end
    ws = st.ws[1]
    out = three(ws)
    out["sort_sorted"] = least(() -> sort_by_cell!(s, grid, ws; workgroupsize = wgs))
    out["patch_layout"] = collect(size(ws.flux_patch))
    out["by_workgroup"] = Dict{String,Any}(string(g) => least(() -> PMC.launcher(PMC.flux_patch_2d_kernel!, BE, g)(patch_args(ws)...; ndrange = prod(n)))
                                           for g in WORKGROUPS)
    if layouts
        out["by_layout"] = Dict{String,Any}()
        for layout in (PMC.StencilMajor(), PMC.CellMajor())
            ws2 = Workspace(s, grid; flux = (shape, stg), flux_layout = layout)
            sort_by_cell!(s, grid, ws2; workgroupsize = wgs)
            out["by_layout"][string(nameof(typeof(layout)))] = three(ws2)
        end
    end
    return out
end

"The work-groups of the patch kernel's scan: the backend's default (`nothing`) and fixed sizes."
const WORKGROUPS = (nothing, 1, 8, 64, 512)

"One element type and mode: the phases of the second of two runs, the plain `run!` time, the deposition's kernels."
function measure(label, (duration, focus), mode; resort = false)
    prob = wake_problem(duration, focus)
    numerics = EMNumerics(; mode, resort)
    p = params(prob)
    times = Dict{Symbol,Float64}()
    local st, total
    for _ in 1:2
        st = EMState(prob, numerics; backend = BE)
        initialize!(st, prob)
        foreach(ph -> times[ph] = 0.0, PHASES)
        KernelAbstractions.synchronize(BE)
        t0 = time_ns()
        for k in 1:prob.nsteps
            timed_step!(st, p, k, times)
        end
        total = (time_ns() - t0) / 1e9
    end
    st2 = EMState(prob, numerics; backend = BE)
    initialize!(st2, prob)
    KernelAbstractions.synchronize(BE)
    t0 = time_ns()
    run!(st2, p, prob.nsteps)
    KernelAbstractions.synchronize(BE)
    plain = (time_ns() - t0) / 1e9
    row = Dict{String,Any}("T" => label, "mode" => string(nameof(typeof(mode))), "steps" => prob.nsteps, "total" => total, "run" => plain,
                           "alive" => alive_count(st.species[1]), "capacity" => capacity(st.species[1]),
                           "shifts" => PME.window_shifts(st, prob.nsteps),
                           "phases" => Dict{String,Any}(string(ph) => times[ph] for ph in PHASES))
    PME.deposit_mode(st, st.species[1]) isa Reference && (row["kernels"] = deposition_kernels(st))
    return row
end

fmt(x) = x >= 10 ? string(round(x; digits = 1)) : string(round(x; digits = 2))

"The Markdown table of the rows: seconds per run and per phase."
function phase_table(rows)
    io = IOBuffer()
    println(io, "| element type | mode | `run!` | ", join(PHASES, " | "), " |")
    println(io, "|---|---|---|", join(("---" for _ in PHASES), "|"), "|")
    for r in rows
        println(io, "| `", r["T"], "` | ", r["mode"], " | ", fmt(r["run"]), " | ", join((fmt(r["phases"][string(ph)]) for ph in PHASES), " | "), " |")
    end
    return String(take!(io))
end

"The Markdown table of the deposition's kernels: milliseconds per call."
function kernel_table(rows)
    io = IOBuffer()
    println(io, "| element type | `deposit_flux!` | patch kernel | reduce kernel | `sort_by_cell!` (sorted input) |")
    println(io, "|---|---|---|---|---|")
    for r in rows
        haskey(r, "kernels") && r["mode"] == "Reference" || continue
        k = r["kernels"]
        println(io, "| `", r["T"], "` | ", join((fmt(1e3 * k[x]) for x in ("deposit_flux", "patch_kernel", "reduce_kernel", "sort_sorted")), " | "), " |")
    end
    return String(take!(io))
end

"The Markdown table of the patch kernel per work-group: milliseconds per call (`nothing`: the backend's default)."
function workgroup_table(rows)
    io = IOBuffer()
    println(io, "| element type | ", join(("`$g`" for g in WORKGROUPS), " | "), " |")
    println(io, "|---|", join(("---" for _ in WORKGROUPS), "|"), "|")
    for r in rows
        haskey(r, "kernels") && r["mode"] == "Reference" || continue
        println(io, "| `", r["T"], "` | ", join((fmt(1e3 * r["kernels"]["by_workgroup"][string(g)]) for g in WORKGROUPS), " | "), " |")
    end
    return String(take!(io))
end

"The Markdown table of the deposition per patch layout: milliseconds per call."
function layout_table(rows)
    io = IOBuffer()
    names = ("StencilMajor", "CellMajor")
    println(io, "| element type | ", join(("`deposit_flux!` $l | patch $l | reduce $l" for l in names), " | "), " |")
    println(io, "|---|", join(("---|---|---" for _ in names), "|"), "|")
    for r in rows
        haskey(r, "kernels") && r["mode"] == "Reference" && haskey(r["kernels"], "by_layout") || continue
        k = r["kernels"]["by_layout"]
        println(io, "| `", r["T"], "` | ", join((fmt(1e3 * k[l][x]) for l in names for x in ("deposit_flux", "patch_kernel", "reduce_kernel")), " | "), " |")
    end
    return String(take!(io))
end

function phase_figure(rows, title)
    fig = Figure(; size = (900, 420))
    ax = Axis(fig[1, 1]; title, ylabel = "seconds per run", xticks = (1:length(rows), ["$(r["T"])\n$(r["mode"])" for r in rows]))
    colors = Makie.wong_colors()
    for (i, r) in enumerate(rows)
        base = 0.0
        for (j, ph) in enumerate(PHASES)
            v = r["phases"][string(ph)]
            barplot!(ax, [i], [v]; offset = [base], color = colors[j], label = string(ph))
            base += v
        end
    end
    axislegend(ax; unique = true, position = :lt)
    return fig
end

function main()
    fast = "fast" in ARGS
    resort = "resort" in ARGS
    gpu = BACKEND_NAME == "cuda" ? CUDACore.name(CUDACore.device()) : nothing
    extra = gpu === nothing ? Dict{String,Any}() : Dict{String,Any}("gpu" => gpu, "cudacore" => string(pkgversion(CUDACore)))
    dir = run_dir("step-timing-$BACKEND_NAME", @__FILE__; prefix = "optim", work_package = "WP-061",
                  backend = BACKEND_NAME == "cuda" ? "KernelAbstractions CUDA (CUDACore)" : "KernelAbstractions CPU",
                  params = Dict{String,Any}("case" => "wake_problem(7, 100)", "cells" => collect(CASE.cells), "ppc" => collect(CASE.ppc),
                                            "nsteps" => CASE.nsteps, "fast" => fast, "resort" => resort),
                  extra_inputs = [joinpath(@__DIR__, "wake2d_case.jl")], extra)
    rows = Dict{String,Any}[]
    for (label, x) in element_cases()
        push!(rows, measure(label, x, Reference(); resort))
        fast && !startswith(label, "Dual") && push!(rows, measure(label, x, Fast(); resort))
        println(phase_table(rows[end:end]))
    end
    where_ = gpu === nothing ? "$(Threads.nthreads()) threads, $(cpu_model())" : gpu
    fig = phase_figure(rows, "WP-045 case, $(CASE.nsteps) steps, $where_")
    save_plot(dir, "step_timing", fig, Dict("rows" => rows, "threads" => Threads.nthreads(), "cpu" => cpu_model(), "gpu" => gpu,
                                            "backend" => BACKEND_NAME, "phases" => collect(string.(PHASES))))
    tables = "Seconds per run of $(CASE.nsteps) steps ($where_):\n\n" * phase_table(rows) *
             "\nMilliseconds per call on the last state:\n\n" * kernel_table(rows) *
             "\nMilliseconds per call of the patch kernel per work-group:\n\n" * workgroup_table(rows)
    any(r -> haskey(r, "kernels") && haskey(r["kernels"], "by_layout"), rows) &&
        (tables *= "\nMilliseconds per call of the deposition per patch layout:\n\n" * layout_table(rows))
    write(joinpath(dir, "tables.md"), tables)
    print(tables)
    println("step_timing: outputs in ", relpath(dir, REPO))
    return dir
end

"The label and the sidecar of each `<label>=<run directory>` argument, the directory relative to the working directory."
function compared_runs(args)
    return map(args) do a
        label, dir = split(a, '='; limit = 2)
        dir = abspath(dir)
        (; label = String(label), dir, data = JSON.parsefile(joinpath(dir, "step_timing.json")),
         manifest = TOML.parsefile(joinpath(dir, "manifest.toml")))
    end
end

"The Markdown table of several runs: seconds per run and in the deposit phase per element type and mode, one column pair per run."
function compare_table(runs)
    io = IOBuffer()
    println(io, "| element type | mode | ", join(("`run!` $(r.label)" for r in runs), " | "), " | ", join(("deposit $(r.label)" for r in runs), " | "), " |")
    println(io, "|---|---|", join(("---" for _ in 1:2length(runs)), "|"), "|")
    for row in runs[1].data["rows"]
        same = [only(filter(x -> x["T"] == row["T"] && x["mode"] == row["mode"], r.data["rows"])) for r in runs]
        println(io, "| `", row["T"], "` | ", row["mode"], " | ", join((fmt(x["run"]) for x in same), " | "), " | ",
                join((fmt(x["phases"]["deposit"]) for x in same), " | "), " |")
    end
    return String(take!(io))
end

"The Markdown table of the deposition's kernels in several runs: milliseconds per call of the patch and the reduce kernel."
function compare_kernel_table(runs)
    io = IOBuffer()
    println(io, "| element type | ", join(("patch $(r.label)" for r in runs), " | "), " | ", join(("reduce $(r.label)" for r in runs), " | "), " |")
    println(io, "|---|", join(("---" for _ in 1:2length(runs)), "|"), "|")
    for row in runs[1].data["rows"]
        row["mode"] == "Reference" || continue
        same = [only(filter(x -> x["T"] == row["T"] && x["mode"] == "Reference", r.data["rows"]))["kernels"] for r in runs]
        println(io, "| `", row["T"], "` | ", join((fmt(1e3 * k["patch_kernel"]) for k in same), " | "), " | ",
                join((fmt(1e3 * k["reduce_kernel"]) for k in same), " | "), " |")
    end
    return String(take!(io))
end

"Per element type and mode, one stacked bar of the phases per run."
function compare_figure(runs, title)
    rows = runs[1].data["rows"]
    fig = Figure(; size = (1100, 460))
    ax = Axis(fig[1, 1]; title, ylabel = "seconds per run", xticks = (1:length(rows), ["$(r["T"])\n$(r["mode"])" for r in rows]))
    colors = Makie.wong_colors()
    w = 0.8 / length(runs)
    top = 0.0
    for (i, row) in enumerate(rows), (j, r) in enumerate(runs)
        x = only(filter(y -> y["T"] == row["T"] && y["mode"] == row["mode"], r.data["rows"]))
        base = 0.0
        pos = i - 0.4 + (j - 0.5) * w
        for (c, ph) in enumerate(PHASES)
            v = x["phases"][string(ph)]
            barplot!(ax, [pos], [v]; offset = [base], width = 0.92w, color = colors[c], label = string(ph))
            base += v
        end
        text!(ax, pos, base; text = r.label, rotation = pi / 2, align = (:left, :center), fontsize = 9, offset = (0, 3))
        top = max(top, base)
    end
    ylims!(ax, 0, 1.3 * top)   # room for the labels above the tallest bar
    axislegend(ax; unique = true, position = :lt)
    return fig
end

"`compare <label>=<run directory> ...`: one table and one figure of several runs of this script."
function compare(args)
    runs = compared_runs(args)
    first_ = runs[1].data
    all(r -> r.data["backend"] == first_["backend"] && r.data["threads"] == first_["threads"] && r.data["cpu"] == first_["cpu"] &&
             r.data["gpu"] == first_["gpu"], runs) || error("step_timing.jl compare: the runs are of different machines, backends or thread counts")
    sidecars = [joinpath(r.dir, f) for r in runs for f in ("step_timing.json", "manifest.toml")]
    dir = run_dir("step-timing-compare", @__FILE__; prefix = "optim", work_package = "WP-061",
                  backend = first_["backend"] == "cuda" ? "KernelAbstractions CUDA (CUDACore)" : "KernelAbstractions CPU",
                  params = Dict{String,Any}("runs" => [Dict("label" => r.label, "run_id" => r.manifest["run_id"], "commit" => r.manifest["commit"],
                                                            "dirty" => r.manifest["dirty"]) for r in runs]),
                  extra_inputs = filter(f -> startswith(f, REPO), sidecars))
    where_ = first_["gpu"] === nothing ? "$(first_["threads"]) threads, $(first_["cpu"])" : first_["gpu"]
    nsteps = first_["rows"][1]["steps"]
    fig = compare_figure(runs, "WP-045 case, $nsteps steps, $where_")
    save_plot(dir, "step_timing_compare", fig,
              Dict("where" => where_, "runs" => [Dict("label" => r.label, "run_id" => r.manifest["run_id"], "commit" => r.manifest["commit"],
                                                      "dirty" => r.manifest["dirty"], "rows" => r.data["rows"]) for r in runs]))
    tables = "Seconds per run of $nsteps steps and in its deposit phase ($where_); the runs: " *
             join(("$(r.label) = `$(r.manifest["run_id"])` at `$(r.manifest["commit"][1:7])`" for r in runs), ", ") * ".\n\n" * compare_table(runs) *
             "\nMilliseconds per call of the deposition's kernels on the last state:\n\n" * compare_kernel_table(runs)
    write(joinpath(dir, "tables.md"), tables)
    print(tables)
    println("step_timing compare: outputs in ", relpath(dir, REPO))
    return dir
end

BACKEND_NAME == "compare" ? compare(ARGS[2:end]) : main()
