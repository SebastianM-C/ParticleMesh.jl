# The frames of a 3D laser-wakefield animation: ParticleMesh's run of the ionization-injection
# case in 3D (WP-082, `scripts/lwfa/cases/ionization-3d.toml`), for the README (F3).
#
#   julia +1.13 --project=scripts scripts/lwfa/animate3d.jl run <case.toml> <out dir> [cpu|cuda] [every] [fast|reference] [t_end]
#
# `run` steps the case's plasma run (`pm_ion_problem` of scripts/lwfa/ionization.jl, Float32) and
# writes one HDF5 file per frame, every `every` steps (default 12) from step 0, into `out/frames/`,
# so that every file stays far below the 20 MB a run branch takes. A frame holds, on every 2nd
# column of the window and every row within `half_width` (25 c/ω₀) of the axis:
#
# - `ne`: the electrons' density n_e/n₀ on the primal nodes (`window_charge!`), quantized to
#   UInt8 over [0, NE_MAX] (attribute `ne_max`);
# - `Ey`: the laser's E_y (dual along y, primal along x and z: half a cell above node row j),
#   quantized to UInt8 over [−EY_MAX, EY_MAX] (`ey_max`, in units of a₀);
# - the born electrons with u_x > UX_MIN (the ones the wake holds; at most MAX_BORN, every k-th
#   if there are more): their window x, their y and z from the axis, and γ, quantized to UInt16
#   over each array's attribute `range`.
#
# The attributes give the nodes' window coordinates (`x0`, `y0`, `z0`, the spacings `dx`, `dy`,
# `dz`), the time, the step and the window's shifts. `run` also writes the run's manifest. On a
# GPU, run it through scripts/gpu-lock.sh; on the CPU box (scripts/cpu-box.sh) it takes about the
# 3D case's CPU run. The mode defaults to Fast: the animation is a picture, not a reference result
# (A4). The frames are not a validation: the case's comparison with Smilei and PIConGPU is in
# reports/wp-082-ionization-three-codes-3d.md. `scripts/lwfa/render3d/render3d.jl` renders them.

include(joinpath(@__DIR__, "ionization.jl"))

const NE_MAX = 4.0f0       # n_e/n₀ above this is clipped: the bubble's sheath and the bunch
const EY_MAX = 1.5f0       # |E_y|/a₀ above this is clipped
const MAX_BORN = 15_000
const UX_MIN = 3.0f0       # the born electrons kept: u_x above this, where WP-065's trapping weight starts
const GAMMA_MAX = 200.0f0  # γ above this is clipped

quantize(a, lo, hi) = round.(UInt8, clamp.((a .- lo) ./ (hi - lo), 0.0f0, 1.0f0) .* 255)
quantize16(a, lo, hi) = round.(UInt16, clamp.((a .- lo) ./ (hi - lo), 0.0f0, 1.0f0) .* 65535)

"The born electrons with u_x > ux_min as window positions and γ, every k-th of them past `MAX_BORN`."
function born_cloud(st, d; ux_min = UX_MIN)
    e = ionization_electrons(st)
    h = d.h
    f(a) = Float32.(Array(a))
    ux = f(e.u[1])
    keep = (Array(e.status) .== ParticleMeshCore.ALIVE) .& (f(e.w) .> 0) .& (ux .> ux_min)
    idx = findall(keep)
    length(idx) > MAX_BORN && (idx = idx[round.(Int, range(1, length(idx); length = MAX_BORN))])
    pos = ntuple(i -> (Float32.(Array(e.cell[i])[idx]) .- 1 .+ f(e.offset[i])[idx]) .* Float32(h[i]), 3)
    γ = sqrt.(1 .+ ux[idx] .^ 2 .+ f(e.u[2])[idx] .^ 2 .+ f(e.u[3])[idx] .^ 2)
    return (; x = pos[1], y = pos[2] .- Float32(d.yc), z = pos[3] .- Float32(d.zc), gamma = γ, count = count(keep))
end

"""
Step the case's plasma run on `backend` and write a frame every `every` steps into `out/frames/`
(header); `t_end` (c/ω₀) overrides the case's end, rounded to a whole number of frames.
"""
function run_frames3d(case_path, out; backend = "cpu", every = 12, mode = "fast", t_end = nothing,
                      half_width = 25.0, xstride = 2)
    mkpath(joinpath(out, "frames"))
    cp(case_path, joinpath(out, "case.toml"); force = true)
    case = TOML.parsefile(case_path)
    t_end === nothing || (case["time"]["t_end"] = Float64(t_end))
    d = ion_derived(case)
    d.N == 3 || error("animate3d.jl takes a 3D case, got $(d.N)D")
    be = backend_named(backend)
    T = Float32
    prob = pm_ion_problem(case; T)
    numerics = pm_ion_numerics(case; mode = mode == "reference" ? Reference() : mode == "fast" ? Fast() : error("mode $mode"))
    st = EMState(prob, numerics; record = (), backend = be)
    p = params(prob)
    initialize!(st, p)
    ρbuf = field(be, T, st.grid)
    g, n = guards(st.grid), ncells(st.grid)
    jw = min(round(Int, half_width / d.dy), d.jaxis)
    kw = min(round(Int, half_width / d.dz), d.kaxis)
    I = (g + 1):xstride:(g + n[1])
    J = (g + 1 + d.jaxis - jw):(g + 1 + d.jaxis + jw)          # node row j (1-based interior) at y = (j − 1)Δy; the axis is row jaxis + 1
    K = (g + 1 + d.kaxis - kw):(g + 1 + d.kaxis + kw)
    n0 = Float32(case["plasma"]["density"])
    a0 = Float32(case["laser"]["a0"])
    nsteps = prob.nsteps
    at = collect(0:every:nsteps)
    t0 = time()
    function frame!(i, k)
        window_charge!(ρbuf, st)
        ne = -Array(ρbuf)[I, J, K] ./ n0
        ey = Array(st.fields.Ey)[I, J, K] ./ a0
        b = born_cloud(st, d)
        h5open(joinpath(out, "frames", "frame$(lpad(i, 4, '0')).h5"), "w") do f
            f["ne", chunk = size(ne), deflate = 6] = quantize(ne, 0.0f0, NE_MAX)
            f["Ey", chunk = size(ey), deflate = 6] = quantize(ey, -EY_MAX, EY_MAX)
            xmax, w = Float32(n[1] * d.dx), Float32(max(jw * d.dy, kw * d.dz))
            for (name, v, lo, hi) in (("born_x", b.x, 0.0f0, xmax), ("born_y", b.y, -w, w), ("born_z", b.z, -w, w),
                                      ("born_gamma", b.gamma, 1.0f0, GAMMA_MAX))
                isempty(v) ? (f[name] = UInt16[]) : (f[name, chunk = (length(v),), deflate = 6] = quantize16(v, lo, hi))
                HDF5.attributes(f[name])["range"] = [lo, hi]
            end
            a = HDF5.attributes(f)
            a["t"], a["step"], a["shifts"] = k * d.dt, k, window_shifts(st, k)
            a["dx"], a["dy"], a["dz"] = d.dx * xstride, d.dy, d.dz
            a["x0"], a["y0"], a["z0"] = 0.0, -jw * d.dy, -kw * d.dz                # window x of the first column; y, z from the axis
            a["ne_max"], a["ey_max"], a["a0"], a["n0"] = NE_MAX, EY_MAX, a0, n0
            a["born_count"] = b.count
        end
    end
    frame!(1, 0)
    i = 1
    for k in 1:nsteps
        step!(st, p, k)
        if k in at
            i += 1
            frame!(i, k)
            println("frame $i of $(length(at)): step $k of $nsteps, $(round(time() - t0; digits = 1)) s")
            flush(stdout)
        end
    end
    elapsed = time() - t0
    write_manifest(out; root = REPO, inputs = ["case.toml"],
                   extra = Dict{String,Any}("run_seconds" => elapsed, "backend" => backend_label(be), "mode" => mode,
                                            "frames" => i, "every" => every, "nsteps" => nsteps, "T" => string(T),
                                            "julia" => string(VERSION), "threads" => Threads.nthreads(),
                                            "cpu" => Sys.cpu_info()[1].model,
                                            "gpu" => be isa CPU ? "none" : @eval(CUDACore.name(CUDACore.device())),
                                            "lost_births" => ionization_lost(st)))
    return out
end

if abspath(PROGRAM_FILE) == @__FILE__
    cmd = get(ARGS, 1, "")
    if cmd == "run"
        t_end = length(ARGS) >= 7 ? parse(Float64, ARGS[7]) : nothing
        Base.invokelatest(run_frames3d, ARGS[2], ARGS[3]; backend = get(ARGS, 4, "cpu"), every = parse(Int, get(ARGS, 5, "12")),
                          mode = get(ARGS, 6, "fast"), t_end)
    else
        error("usage: animate3d.jl run <case.toml> <out dir> [cpu|cuda] [every] [fast|reference] [t_end]")
    end
end
