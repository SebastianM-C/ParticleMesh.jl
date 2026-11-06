# An animation of ParticleMesh's 2D LWFA case: the electron density with the laser field on top,
# one frame every few steps of the moving window, for the README and for talks (F3).
#
#   julia +1.13 --project=scripts scripts/lwfa/animate.jl run <case.toml> <out dir> [cuda|cpu] [frames] [fast|reference]
#   julia +1.13 --project=scripts scripts/lwfa/animate.jl render <out dir>
#
# `run` steps the case's plasma run (the ParticleMesh column of scripts/lwfa/lwfa.jl, Float64) and
# writes `frames.h5`: n_e/n₀ and E_y as Float32 on every 3rd column and every row of the window, at
# `frames` evenly spaced steps (default 200), and the run's manifest. On a GPU, run it through scripts/gpu-lock.sh.
# The mode defaults to Fast: the animation is a picture, not a reference result (A4).
# `render` turns `frames.h5` into `lwfa.mp4` (every frame, full size) and `lwfa.gif` (every 2nd
# frame, smaller, for the README) with CairoMakie, and the last frame into `lwfa_last.png` with
# its JSON sidecar (the axes and the time; the maps themselves are in frames.h5).
#
# The frames are not a validation: the run's agreement with Smilei is in reports/lwfa-2d-a2.md.

include(joinpath(@__DIR__, "lwfa.jl"))

using CUDACore: CUDACore, CUDABackend

backend_named(name) = name == "cpu" ? CPU() : name == "cuda" ? CUDABackend() : error("backend $name: cpu or cuda")

"Step the case's plasma run on `backend` and write `frames` maps of n_e/n₀ and E_y into `out/frames.h5`."
function run_frames(case_path, out; backend = "cuda", frames = 200, mode = "fast", stride = (3, 1))
    mkpath(out)
    cp(case_path, joinpath(out, "case.toml"); force = true)
    case = TOML.parsefile(case_path)
    d = derived(case)
    be = backend_named(backend)
    prob = pm_problem(case; plasma = true)
    numerics = EMNumerics(; mode = mode == "reference" ? Reference() : mode == "fast" ? Fast() : error("mode $mode"))
    st = EMState(prob, numerics; backend = be)
    initialize!(st, prob)
    p = params(prob)
    ρbuf = field(be, Float64, st.grid)
    g, n = guards(st.grid), ncells(st.grid)
    I, J = (g + 1):stride[1]:(g + n[1]), (g + 1):stride[2]:(g + n[2])
    n0 = case["plasma"]["density"]
    at = unique(round.(Int, range(0, prob.nsteps; length = frames)))
    t0 = time()
    h5open(joinpath(out, "frames.h5"), "w") do f
        HDF5.attributes(f)["dx"], HDF5.attributes(f)["dy"] = d.dx * stride[1], d.dy * stride[2]
        HDF5.attributes(f)["a0"], HDF5.attributes(f)["n0"] = case["laser"]["a0"], n0
        function frame!(i, k)
            window_charge!(ρbuf, st)
            fr = create_group(f, "frame$(lpad(i, 4, '0'))")
            fr["ne"] = Float32.(-Array(ρbuf)[I, J] ./ n0)
            fr["Ey"] = Float32.(Array(st.fields.Ey)[I, J])
            HDF5.attributes(fr)["t"] = k * d.dt
            HDF5.attributes(fr)["step"] = k
            HDF5.attributes(fr)["x0"] = (window_shifts(st, k) - d.na) * d.dx    # lab x of the window's first node
        end
        frame!(1, 0)
        i = 1
        for k in 1:prob.nsteps
            step!(st, p, k)
            if k in at
                i += 1
                frame!(i, k)
                println("frame $i of $(length(at)): step $k of $(prob.nsteps), $(round(time() - t0; digits = 1)) s")
                flush(stdout)
            end
        end
        HDF5.attributes(f)["frames"] = i
    end
    elapsed = time() - t0
    write_manifest(out; root = REPO, inputs = ["case.toml"],
                   extra = Dict{String,Any}("run_seconds" => elapsed, "backend" => string(typeof(be)), "mode" => mode,
                                            "frames" => length(at), "nsteps" => prob.nsteps, "T" => "Float64",
                                            "julia" => string(VERSION), "threads" => Threads.nthreads(),
                                            "cpu" => Sys.cpu_info()[1].model, "gpu" => gpu_name(be)))
    return out
end

gpu_name(::CPU) = "none"
gpu_name(::CUDABackend) = CUDACore.name(CUDACore.device())

# Transparent at zero, blue and red for the two signs of E_y: drawn over the density.
const LASER_COLORS = cgrad([RGBAf(0.15, 0.45, 1.0, 0.95), RGBAf(0.15, 0.45, 1.0, 0.0), RGBAf(1.0, 0.25, 0.15, 0.0),
                            RGBAf(1.0, 0.25, 0.15, 0.95)], [0.0, 0.5 - 1e-3, 0.5 + 1e-3, 1.0])

"Draw frame `i` of `f` (an open frames.h5) on a figure of `size`; returns the figure and `show!(i)`."
function frame_figure(f, size)
    dx, dy, a0 = read_attribute(f, "dx"), read_attribute(f, "dy"), read_attribute(f, "a0")
    fr(i) = f["frame$(lpad(i, 4, '0'))"]
    ne0 = read(fr(1)["ne"])
    y = (0:Base.size(ne0, 2) - 1) .* dy
    xi = (0:Base.size(ne0, 1) - 1) .* dx
    ne, ey, xs, title = Observable(ne0), Observable(read(fr(1)["Ey"])), Observable(collect(xi)), Observable("")
    fig = Figure(; size, backgroundcolor = :black)
    ax = Axis(fig[1, 1]; xlabel = "x (c/ω₀), lab frame", ylabel = "y (c/ω₀)", title,
              backgroundcolor = :black, titlecolor = :white, xlabelcolor = :white, ylabelcolor = :white,
              xticklabelcolor = :white, yticklabelcolor = :white, xtickcolor = :white, ytickcolor = :white,
              bottomspinecolor = :white, leftspinecolor = :white, topspinevisible = false, rightspinevisible = false)
    hm = heatmap!(ax, xs, y, ne; colorrange = (0, 3), colormap = :grays)
    heatmap!(ax, xs, y, ey; colorrange = (-a0, a0), colormap = LASER_COLORS)
    Colorbar(fig[1, 2], hm; label = "n_e/n₀", labelcolor = :white, ticklabelcolor = :white, tickcolor = :white)
    function show!(i)
        g = fr(i)
        t, x0 = read_attribute(g, "t"), read_attribute(g, "x0")
        ne.val, ey.val = read(g["ne"]), read(g["Ey"])
        xs[] = x0 .+ xi
        title[] = "ParticleMesh: n_e/n₀ (grey) and E_y (red, blue), t = $(round(Int, t))/ω₀"
        limits!(ax, first(xs[]), last(xs[]), first(y), last(y))
        return (; t, x = xs[], y = collect(y))
    end
    return fig, show!
end

"Render `out/frames.h5` as `lwfa.mp4`, `lwfa.gif` and the last frame as `lwfa_last.png` with its sidecar."
function render(out; fps = 20)
    h5open(joinpath(out, "frames.h5"), "r") do f
        nf = read_attribute(f, "frames")
        fig, show! = frame_figure(f, (1100, 420))
        record(show!, fig, joinpath(out, "lwfa.mp4"), 1:nf; framerate = fps)
        last_frame = show!(nf)
        save_plot(out, "lwfa_last", fig, Dict{String,Any}("t" => last_frame.t, "x" => last_frame.x, "y" => last_frame.y,
                  "source" => "frames.h5, frame $nf", "colorrange_ne" => [0, 3], "colorrange_Ey" => "±a0"))
        small, show_small! = frame_figure(f, (800, 300))
        record(show_small!, small, joinpath(out, "lwfa.gif"), 1:2:nf; framerate = fps ÷ 2)
    end
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    cmd = get(ARGS, 1, "")
    if cmd == "run"
        run_frames(ARGS[2], ARGS[3]; backend = get(ARGS, 4, "cuda"), frames = parse(Int, get(ARGS, 5, "200")),
                   mode = get(ARGS, 6, "fast"))
    elseif cmd == "render"
        render(ARGS[2])
    else
        error("usage: animate.jl run <case.toml> <out> [cuda|cpu] [frames] [fast|reference] | render <out>")
    end
end
