# A 2D LWFA case in ParticleMesh and Smilei, matched through one case file (A5), run on the CPU box.
#
#   julia +1.13 --project=scripts scripts/lwfa/lwfa.jl prepare <case.toml> <tag>     # scripts/runs/lwfa-<tag>/ and box-request.toml
#   julia +1.13 --project=scripts scripts/lwfa/lwfa.jl run-pm <dir>                  # ParticleMesh into <dir>/summary.json, maps.h5
#   julia +1.13 --project=scripts scripts/lwfa/lwfa.jl summarize-smilei <dir>        # Smilei's dumps into the same format
#   julia +1.13 --project=scripts scripts/lwfa/lwfa.jl summarize-picongpu <dir>      # PIConGPU's dumps into the same format
#   julia +1.13 --project=scripts scripts/lwfa/lwfa.jl compare <runs dir> <out dir>  # the comparison, figures and sidecars
#
# A prepared run has six directories: `smilei-vacuum`, `smilei-plasma`, `pm-vacuum`, `pm-plasma`,
# `picongpu-vacuum`, `picongpu-plasma`. The vacuum pairs check that the codes launch the same pulse
# (A5's "the injected field should be compared in vacuum before any plasma run",
# `questions/wp-017-lwfa-case.md` §3); the plasma runs are the comparison. The CPU box request
# covers ParticleMesh and Smilei; the PIConGPU directories go to the GPU runner (`command: picongpu
# scripts/runs/lwfa-<tag>/picongpu-<kind>`), and `compare` takes PIConGPU in when its summaries
# exist. How PIConGPU is matched, its axes swapped (its window moves along y), is in
# scripts/lwfa/picongpu.jl.
#
# How the two codes are matched (each line a choice, the case file holds the numbers):
#
# - Coordinates. The case's x is the lab coordinate with x = 0 on the laser's injection plane:
#   Smilei's xmin at t = 0, and ParticleMesh's antenna plane, `antenna_cells` columns into its box.
#   Both use the same Δx, Δy and Δt, so both grids are one lattice in the lab frame: a primal node
#   sits at x = m·Δx, a dual one at (m − ½)·Δx, for an integer m that the summaries record. The
#   comparison matches nodes by m; nothing is interpolated.
#   ParticleMesh: interior node i of the window after s shifts is at m = i − 1 + s − antenna_cells.
#   Smilei: primal index i (from 0) at m = i + x_moved/Δx, its dual index i at the same m (the
#   Fields dumps' `x_moved` attribute; `gridGlobalOffset` is 0, `Params/OpenPMDparams.cpp`).
# - The window. ParticleMesh moves by one cell once c·(t − start) has passed it (it lags the
#   continuous front by less than a cell); Smilei v5.1 moves by a whole patch as soon as
#   c·(t − start) exceeds what it has moved (`SimWindow::isMoving`), so its front leads by less than
#   a patch (24 cells here). Both fronts stay far ahead of the pulse; the comparison uses the m
#   both windows hold.
# - The laser. ParticleMesh's `GaussianLaser` at the antenna plane: E_y = a0·|u|·exp(−(t − delay)²/τ²)·
#   cos(…), the 2D paraxial beam focused at `focus`. Smilei's `LaserGaussian2D` on xmin with
#   `focus = [focus, y_c]` and `tgaussian(start = 0, duration = 2·delay, fwhm = 2τ√ln2, center =
#   delay)`: v5.1's tgaussian is exp(−(t − center)²/σ) with σ = (fwhm/2)²/ln 2 (`pyprofiles.py`), so
#   the same envelope, cut to zero after 2·delay (ParticleMesh's continues at e^−16 of the peak).
#   v5.1's normal-incidence LaserGaussian2D has no Gouy phase at the boundary, so the carriers can
#   differ by a constant phase; the comparison uses the envelope (a running max of |E_y| over one
#   wavelength) and the box maximum of |E_y|, not the carrier.
# - The plasma. Electrons of n0·sin²((x − vacuum)/ramp·π/2) (then n0), regular, 2 × 2 per cell,
#   cold; ions immobile (ParticleMesh's neutralizing background, Smilei's frozen ions on the same
#   positions). Smilei's regular load sits at 0.975·(k + ½)/n of a cell
#   (`ParticleCreator::createPosition`), ParticleMesh's at (k + ½)/n; Smilei evaluates the density
#   where its creator says, ParticleMesh at each particle (its quiet start). Both are small, known
#   differences of the loads.
# - Numerics. Yee, Esirkepov, quadratic shapes, Boris, no current filter, Float64, Reference mode
#   in ParticleMesh. Transverse boundaries periodic in both codes (the box is 7.2 waists wide);
#   along x Smilei has Silver–Müller boundaries, ParticleMesh its field-free outside, which
#   reflects what reaches the back edge (`questions/wp-028-moving-window.md` §3); the comparison
#   stays away from the back.

include(joinpath(@__DIR__, "..", "engine2d", "common.jl"))
const ParticleMeshCore = ParticleMesh.ParticleMeshCore
using ParticleMeshElectromagnetic
using ParticleMeshElectromagnetic: window_shifts, window_charge!
using .ParticleMeshCore: guards, ncells, field
using HDF5

const KINDS = ("smilei-vacuum", "smilei-plasma", "pm-vacuum", "pm-plasma")

# --- the case -----------------------------------------------------------------------------------

"The numbers every part derives from the case: cells, steps, Δt, the antenna offset, the axis."
function derived(case)
    g, t = case["grid"], case["time"]
    dx, dy = Float64(g["dx"]), Float64(g["dy"])
    nx, ny = Int.(g["cells"])
    na = Int(g["antenna_cells"])
    dt = Float64(t["courant"]) / sqrt(1 / dx^2 + 1 / dy^2)
    steps, every = Int(t["steps"]), Int(t["fields_every"])
    iseven(ny) || error("the axis must be a node row: an even number of transverse cells, got $ny")
    return (; dx, dy, nx, ny, na, xa = na * dx, dt, steps, every, yc = ny * dy / 2, jaxis = ny ÷ 2)
end

"The output steps: every `fields_every` steps from 0."
output_steps(case) = (d = derived(case); collect(0:d.every:d.steps))

polarization_angle(case) = case["laser"]["polarization"] == "y" ? 0.0 :
    error("polarization $(case["laser"]["polarization"]): only \"y\" is matched")

# --- ParticleMesh -------------------------------------------------------------------------------

"The case as a ParticleMesh `EMProblem{2}` (`plasma = false`: the laser in vacuum)."
function pm_problem(case; plasma = true, T = Float64)
    d = derived(case)
    L, P = case["laser"], case["plasma"]
    laser = GaussianLaser(; a0 = T(L["a0"]), waist = T(L["waist"]), duration = T(L["duration"]), delay = T(L["delay"]),
                          focus = T(L["focus"] + d.xa), position = T(d.xa), center = T(d.yc),
                          polarization = T(polarization_angle(case)))
    z = (zero(T), zero(T), zero(T))
    e = SpeciesDescription{T}(:electron, -one(T), one(T), T(P["density"]), zero(T), (1, 0),
                              Tuple(Int.(P["particles_per_cell"])), z, z, 1)
    n = (d.nx + d.na, d.ny)
    return EMProblem(String(case["name"]), (T(n[1] * d.dx), T(n[2] * d.dy)), n, T(d.dt), d.steps, plasma ? (e,) : (),
                     Int(P["seed"]), (1, 0); lasers = (laser,), boundaries = (:window, :periodic),
                     window = MovingWindow(; start = T(case["window"]["start"])),
                     profile = DensityRamp(; vacuum = T(P["vacuum"] + d.xa), ramp = T(P["ramp"])))
end

# A running max of |v| over ±h points: the envelope of a carrier with h about half a wavelength.
running_envelope(v, h) = [maximum(abs, @view v[max(1, i - h):min(end, i + h)]) for i in eachindex(v)]

# The kinetic-energy histogram of the alive electrons, weighted by the particle weights.
function pm_spectrum(s, edges)
    alive = Array(s.status) .== ParticleMeshCore.ALIVE
    u1, u2, u3, w = Array(s.u[1])[alive], Array(s.u[2])[alive], Array(s.u[3])[alive], Array(s.w)[alive]
    ek = @. sqrt(1 + u1^2 + u2^2 + u3^2) - 1
    h = zeros(length(edges) - 1)
    for (e, wi) in zip(ek, w)
        b = searchsortedlast(edges, e)
        1 <= b < length(edges) && (h[b] += wi)
    end
    return (; hist = h, ekin_max = isempty(ek) ? 0.0 : maximum(ek), alive = count(alive))
end

spectrum_edges(case) = (s = case["diagnostics"]["spectrum_ekin"]; collect(range(Float64(s[1]), Float64(s[2]); length = Int(s[3]) + 1)))

"One output of a ParticleMesh state at step k: the axis lineouts with their lattice offsets, the box's max |E_y|, the spectrum, the map."
function pm_output(st, case, k, ρbuf, keep_map)
    d = derived(case)
    s = window_shifts(st, k)
    g, n = guards(st.grid), ncells(st.grid)
    I, J = (g + 1):(g + n[1]), (g + 1):(g + n[2])
    Ex, Ey = Array(st.fields.Ex)[I, J], Array(st.fields.Ey)[I, J]
    m0 = s - d.na                         # node i ↔ m = i − 1 + m0; Ex index i ↔ m = i + m0 (x = (m − ½)Δx)
    ja = d.jaxis + 1                      # node row of the axis, y = (ja − 1)·Δy = y_c
    out = Dict{String,Any}("step" => k, "t" => k * d.dt, "shifts" => s,
                           "Ex_axis" => Ex[:, ja], "Ex_m0" => 1 + m0,
                           "Ey_axis" => (Ey[:, ja - 1] .+ Ey[:, ja]) ./ 2, "Ey_m0" => m0,      # Ey at y = (j − ½)Δy: rows around the axis
                           "Ey_absmax" => maximum(abs, Ey))
    out["Ey_envelope"] = running_envelope(out["Ey_axis"], 16)
    if !isempty(st.species)
        window_charge!(ρbuf, st)
        ρ = Array(ρbuf)[I, J]
        out["rho_axis"], out["rho_m0"] = ρ[:, ja], m0
        sp = pm_spectrum(st.species[1], spectrum_edges(case))
        out["spectrum"], out["ekin_max"], out["alive"] = sp.hist, sp.ekin_max, sp.alive
        keep_map && (out["map"] = (rho = ρ, Ey = Ey, m0 = m0))
    end
    return out
end

"Run the ParticleMesh column of the case in `dir` (its `case.toml` and `kind.txt`) into `summary.json` and `maps.h5`."
function run_pm(dir)
    case = TOML.parsefile(joinpath(dir, "case.toml"))
    plasma = strip(read(joinpath(dir, "kind.txt"), String)) == "plasma"
    prob = pm_problem(case; plasma)
    st = EMState(prob, EMNumerics(; mode = Reference()))
    initialize!(st, prob)
    p = params(prob)
    ρbuf = field(CPU(), Float64, st.grid)
    steps = output_steps(case)
    maps_at = Set(steps[Int.(case["diagnostics"]["map_outputs"])])
    outs = Any[pm_output(st, case, 0, ρbuf, 0 in maps_at)]
    t0 = time()
    for k in 1:prob.nsteps
        step!(st, p, k)
        if k in steps
            push!(outs, pm_output(st, case, k, ρbuf, k in maps_at))
            println("pm $(basename(dir)): step $k of $(prob.nsteps), $(round(time() - t0; digits = 1)) s")
            flush(stdout)
        end
    end
    elapsed = time() - t0
    lost = isempty(st.species) ? Int[] : collect(ParticleMeshElectromagnetic.window_lost(st))
    write_summary(dir, outs; code = "ParticleMesh", extra = Dict{String,Any}("seconds" => elapsed, "lost" => lost,
                  "threads" => Threads.nthreads(), "cpu" => Sys.cpu_info()[1].model, "nsteps" => prob.nsteps,
                  "spectrum_edges" => spectrum_edges(case)))
    write_manifest(dir; root = REPO, extra = Dict{String,Any}("run_seconds" => elapsed, "threads" => Threads.nthreads(),
                   "cpu" => Sys.cpu_info()[1].model, "julia" => string(VERSION)))
    return dir
end

# --- Smilei ---------------------------------------------------------------------------------------

pyf(x) = repr(Float64(x))

"The Smilei v5.1 namelist of the case (`plasma = false`: the laser in vacuum)."
function smilei_namelist(case; plasma = true)
    d = derived(case)
    L, P, D = case["laser"], case["plasma"], case["diagnostics"]
    ppc = Int.(P["particles_per_cell"])
    fwhm = 2 * L["duration"] * sqrt(log(2))
    patches = Int.(case["grid"]["smilei_patches"])
    io = IOBuffer()
    println(io, """
    # Smilei namelist generated by scripts/lwfa/lwfa.jl from case "$(case["name"])" ($(plasma ? "plasma" : "vacuum")).
    # Do not edit; regenerate from the case file. Units: Smilei normalized, reference frequency = ω₀ (A1).
    import math
    import numpy as np

    Main(
        geometry = "2Dcartesian",
        interpolation_order = 2,
        cell_length = [$(pyf(d.dx)), $(pyf(d.dy))],
        number_of_cells = [$(d.nx), $(d.ny)],
        timestep = $(pyf(d.dt)),
        simulation_time = $(pyf((d.steps + 0.5) * d.dt)),
        number_of_patches = [$(patches[1]), $(patches[2])],
        EM_boundary_conditions = [["silver-muller"], ["periodic"]],
        solve_poisson = False,
        random_seed = $(Int(P["seed"])),
        print_every = 500,
    )

    MovingWindow(
        time_start = $(pyf(case["window"]["start"])),
        velocity_x = 1.0,
    )

    LaserGaussian2D(
        box_side = "xmin",
        a0 = $(pyf(L["a0"])),
        omega = 1.0,
        focus = [$(pyf(L["focus"])), $(pyf(d.yc))],
        waist = $(pyf(L["waist"])),
        polarization_phi = $(pyf(polarization_angle(case))),
        time_envelope = tgaussian(start = 0.0, duration = $(pyf(2 * L["delay"])), fwhm = $(pyf(fwhm)), center = $(pyf(L["delay"]))),
    )

    DiagScalar(every = 50, precision = 10)
    """)
    if plasma
        println(io, """
        def plasma_density(x, y):
            s = np.clip((x - $(pyf(P["vacuum"]))) / $(pyf(P["ramp"])), 0.0, 1.0)
            return $(pyf(P["density"])) * np.sin(0.5 * math.pi * s)**2

        for name, charge, mass, frozen in (("electron", -1.0, 1.0, 0.0), ("ion", 1.0, 1836.15267343, $(pyf(4 * d.steps * d.dt)))):
            Species(
                name = name,
                position_initialization = "regular",
                regular_number = [$(ppc[1]), $(ppc[2])],
                momentum_initialization = "cold",
                particles_per_cell = $(prod(ppc)),
                mass = mass,
                charge = charge,
                number_density = plasma_density,
                time_frozen = frozen,
                boundary_conditions = [["remove", "remove"], ["periodic", "periodic"]],
            )

        DiagParticleBinning(
            deposited_quantity = "weight",
            every = $(d.every),
            species = ["electron"],
            axes = [["ekin", $(pyf(D["spectrum_ekin"][1])), $(pyf(D["spectrum_ekin"][2])), $(Int(D["spectrum_ekin"][3]))]],
        )
        """)
    end
    println(io, """
    DiagFields(
        every = $(d.every),
        fields = [$(plasma ? "\"Ex\", \"Ey\", \"Rho_electron\"" : "\"Ex\", \"Ey\"")],
        datatype = "double",
    )""")
    return String(take!(io))
end

const SMILEI_RUN_SH = """
#!/usr/bin/env bash
# Generated by scripts/lwfa/lwfa.jl. Runs inside the Smilei container with the run directory at
# /run-dir (PM_RUN_DIR) and the worktree read-only at /worktree.
set -euo pipefail
cd "\${PM_RUN_DIR:-/run-dir}"
export OMP_NUM_THREADS=16
export OMP_SCHEDULE=dynamic
{
  echo "# smilei binary: \$(command -v smilei)"
  echo "# smilei version: \$(cat /opt/smilei/VERSION 2>/dev/null || echo unknown)"
  echo "# threads: \$OMP_NUM_THREADS, cpus: \$(nproc)"
  echo "# cpu model: \$(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')"
} > container.txt
echo "# start: \$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" >> container.txt
smilei namelist.py > smilei.log 2>&1
echo "# end: \$(date -u +%Y-%m-%dT%H:%M:%S.%NZ)" >> container.txt
"""

_xfirst(a::AbstractMatrix) = permutedims(a)

"Read Smilei's dumps in `dir` into the summary format of `pm_output`."
function summarize_smilei(dir)
    case = TOML.parsefile(joinpath(dir, "case.toml"))
    d = derived(case)
    plasma = strip(read(joinpath(dir, "kind.txt"), String)) == "plasma"
    steps = output_steps(case)
    maps_at = Set(steps[Int.(case["diagnostics"]["map_outputs"])])
    outs = Any[]
    spectra = Dict{Int,Vector{Float64}}()
    if plasma
        h5open(joinpath(dir, "ParticleBinning0.h5"), "r") do f
            for k in keys(f)
                startswith(k, "timestep") || continue
                spectra[parse(Int, k[9:end])] = vec(read(f[k]))
            end
        end
    end
    h5open(joinpath(dir, "Fields0.h5"), "r") do f
        data = f["data"]
        for key in sort(collect(keys(data)); by = k -> parse(Int, k))
            it = parse(Int, key)
            grp = data[key]
            xm = Float64(read_attribute(grp, "x_moved"))
            mx = round(Int, xm / d.dx)
            abs(mx * d.dx - xm) <= 1e-6 * d.dx || error("x_moved = $xm is not a whole number of cells")
            Ex, Ey = _xfirst(read(grp["Ex"])), _xfirst(read(grp["Ey"]))   # Ex (nx + 2, ny + 1) dual in x; Ey (nx + 1, ny + 2) dual in y
            ja = d.jaxis + 1                                                # primal row of the axis (index jaxis from 0)
            out = Dict{String,Any}("step" => it, "t" => it * d.dt, "x_moved" => xm,
                                   "Ex_axis" => Ex[:, ja], "Ex_m0" => mx,  # dual index i (from 0) ↔ m = i + mx, at x = (m − ½)Δx
                                   "Ey_axis" => (Ey[:, ja] .+ Ey[:, ja + 1]) ./ 2, "Ey_m0" => mx,   # dual rows jaxis, jaxis + 1 at y_c ∓ Δy/2
                                   "Ey_absmax" => maximum(abs, Ey))
            out["Ey_envelope"] = running_envelope(out["Ey_axis"], 16)
            if plasma
                ρ = _xfirst(read(grp["Rho_electron"]))                      # (nx + 1, ny + 1), primal
                out["rho_axis"], out["rho_m0"] = ρ[:, ja], mx
                out["spectrum"] = get(spectra, it, Float64[])
                it in maps_at && (out["map"] = (rho = ρ[1:(end - 1), 1:(end - 1)], Ey = Ey[1:(end - 1), 2:(end - 1)], m0 = mx))
            end
            push!(outs, out)
        end
    end
    secs = try
        lines = readlines(joinpath(dir, "container.txt"))
        stamp(p) = DateTime(first(filter(l -> startswith(l, p), lines))[length(p) + 1:end][1:23], "yyyy-mm-ddTHH:MM:SS.sss")
        Dates.value(stamp("# end: ") - stamp("# start: ")) / 1000
    catch
        nothing
    end
    write_summary(dir, outs; code = "Smilei", extra = Dict{String,Any}("seconds" => secs, "spectrum_edges" => spectrum_edges(case)))
    return dir
end

# --- summaries ------------------------------------------------------------------------------------

"Write the outputs as `summary.json` (lineouts, scalars, spectra) and the 2D maps as `maps.h5`."
function write_summary(dir, outs; code, extra = Dict{String,Any}())
    h5open(joinpath(dir, "maps.h5"), "w") do f
        for o in outs
            haskey(o, "map") || continue
            g = create_group(f, "step$(o["step"])")
            mp = o["map"]
            g["rho"], g["Ey"] = mp.rho, mp.Ey
            HDF5.attributes(g)["m0"] = mp.m0
            HDF5.attributes(g)["t"] = o["t"]
        end
    end
    rows = [Dict{String,Any}(k => v for (k, v) in o if k != "map") for o in outs]
    open(io -> JSON.print(io, merge(Dict{String,Any}("code" => code, "outputs" => rows), extra)), joinpath(dir, "summary.json"), "w")
    return dir
end

# --- prepare --------------------------------------------------------------------------------------

"Write the six run directories of `case_path` under scripts/runs/lwfa-<tag>/ and the CPU box request of ParticleMesh and Smilei."
function prepare(case_path, tag)
    case = TOML.parsefile(case_path)
    root = joinpath(REPO, "scripts", "runs", "lwfa-$tag")
    ispath(root) && error("$root exists")
    for kind in KINDS
        dir = joinpath(root, kind)
        mkpath(dir)
        cp(case_path, joinpath(dir, "case.toml"))
        plasma = endswith(kind, "plasma")
        write(joinpath(dir, "kind.txt"), plasma ? "plasma\n" : "vacuum\n")
        inputs = ["case.toml", "kind.txt"]
        if startswith(kind, "smilei")
            write(joinpath(dir, "namelist.py"), smilei_namelist(case; plasma))
            write(joinpath(dir, "run.sh"), SMILEI_RUN_SH)
            chmod(joinpath(dir, "run.sh"), 0o755)
            append!(inputs, ["namelist.py", "run.sh"])
        end
        write_manifest(dir; root = REPO, inputs,
                       extra = Dict{String,Any}("run_id" => "lwfa-$tag/$kind", "code" => startswith(kind, "smilei") ? "Smilei v5.1" : "ParticleMesh",
                                                "case" => case["name"], "generator" => "scripts/lwfa/lwfa.jl"))
    end
    for kind in PIC_KINDS
        dir = joinpath(root, kind)
        mkpath(dir)
        cp(case_path, joinpath(dir, "case.toml"))
        plasma = endswith(kind, "plasma")
        write(joinpath(dir, "kind.txt"), plasma ? "plasma\n" : "vacuum\n")
        prepare_picongpu(dir, case_path, case; plasma, tag, kind)
    end
    rel(k) = "scripts/runs/lwfa-$tag/$k"
    req = IOBuffer()
    println(req, "# CPU box request of the LWFA case $(case["name"]) (scripts/lwfa/lwfa.jl prepare)")
    println(req, "timeout = 7200\ncategory = \"crosscode\"")
    println(req, "outputs = [", join(repr.(rel.(KINDS)), ", "), "]")
    # ParticleMesh first: the box stops at the first failing step, and a Smilei namelist that
    # Smilei refuses should not cost the ParticleMesh column
    julia_step(args...) = println(req, "\n[[steps]]\nkind = \"julia\"\nargs = [", join(repr.(["--project=scripts", "scripts/lwfa/lwfa.jl", args...]), ", "), "]")
    for k in ("pm-vacuum", "pm-plasma")
        julia_step("run-pm", rel(k))
    end
    for k in ("smilei-vacuum", "smilei-plasma")
        println(req, "\n[[steps]]\nkind = \"smilei\"\ndir = \"$(rel(k))\"")
        julia_step("summarize-smilei", rel(k))
    end
    write(joinpath(root, "box-request.toml"), String(take!(req)))
    return root
end

# --- compare --------------------------------------------------------------------------------------

load_summary(dir) = JSON.parsefile(joinpath(dir, "summary.json"))

# The values of two lineouts on the lattice points m both hold, in the range [mlo, mhi].
function common_points(a, a0, b, b0; mlo = typemin(Int), mhi = typemax(Int))
    ma, mb = a0 .+ (0:(length(a) - 1)), b0 .+ (0:(length(b) - 1))
    lo, hi = max(first(ma), first(mb), mlo), min(last(ma), last(mb), mhi)
    lo <= hi || return (m = Int[], a = Float64[], b = Float64[])
    m = collect(lo:hi)
    return (; m, a = Float64.(a[m .- a0 .+ 1]), b = Float64.(b[m .- b0 .+ 1]))
end

relrms(a, b) = sqrt(sum(abs2, a .- b) / max(sum(abs2, b), eps()))

# The m of the envelope peak (the laser's centre on the axis, to within the carrier: the running
# max snaps to a crest); used to place the wake window.
peak_m(o) = o["Ey_m0"] + argmax(Float64.(o["Ey_envelope"])) - 1

# The energy centroid of E_y² on the axis, in c/ω₀: the laser's position, independent of the
# carrier phase (which differs between the codes by Smilei's missing Gouy phase).
function centroid_x(o, dx)
    e = Float64.(o["Ey_axis"]) .^ 2
    m = o["Ey_m0"] .+ (0:(length(e) - 1))
    return sum(e) > 0 ? sum(m .* e) / sum(e) * dx : nothing
end

# The rows of one medium's comparison of ParticleMesh (`pm`) with another code (`other`), per output both hold.
function compare_rows(pm, other, medium, d, λp)
    bystep = Dict(o["step"] => o for o in other)
    rows = Dict{String,Any}[]
    for o in pm
        s = get(bystep, o["step"], nothing)
        s === nothing && continue
        env = common_points(o["Ey_envelope"], o["Ey_m0"], s["Ey_envelope"], s["Ey_m0"])
        r = Dict{String,Any}("step" => o["step"], "t" => o["t"],
                             "Ey_absmax" => [o["Ey_absmax"], s["Ey_absmax"]],
                             "peak_x" => [peak_m(o) * d.dx, peak_m(s) * d.dx],
                             "centroid_x" => Any[centroid_x(o, d.dx), centroid_x(s, d.dx)],
                             "envelope_relrms" => isempty(env.m) ? nothing : relrms(env.a, env.b))
        if medium == "plasma"
            # the wake: from the laser's peak back two plasma wavelengths, on the axis
            mp = peak_m(o)
            wake = common_points(o["Ex_axis"], o["Ex_m0"], s["Ex_axis"], s["Ex_m0"];
                                 mlo = mp - round(Int, 2λp / d.dx), mhi = mp)
            rho = common_points(o["rho_axis"], o["rho_m0"], s["rho_axis"], s["rho_m0"];
                                mlo = mp - round(Int, 2λp / d.dx), mhi = mp)
            if !isempty(wake.m)
                r["wake_Ex_relrms"] = relrms(wake.a, wake.b)
                r["wake_Ex_min"] = [minimum(wake.a), minimum(wake.b)]
                r["wake_Ex_max"] = [maximum(wake.a), maximum(wake.b)]
                r["wake_rho_relrms"] = relrms(rho.a, rho.b)
            end
            r["ekin_max_pm"] = get(o, "ekin_max", nothing)
        end
        push!(rows, r)
    end
    return rows
end

"The codes whose summaries are under `runs`: ParticleMesh and Smilei always, PIConGPU when both its summaries exist."
function present_codes(runs)
    codes = ["pm", "smilei"]
    all(k -> isfile(joinpath(runs, k, "summary.json")), PIC_KINDS) && push!(codes, "picongpu")
    return codes
end

"""
Compare the summaries under `runs` (pm-*, smilei-*, and picongpu-* when present) and write the
figures, sidecars and `comparison.json` into `out`. `comparison.json` keeps ParticleMesh against
Smilei under `vacuum` and `plasma`, and ParticleMesh against PIConGPU under `picongpu_vacuum` and
`picongpu_plasma`; each pair is [ParticleMesh, other].
"""
function compare(runs, out)
    mkpath(out)
    case = TOML.parsefile(joinpath(runs, "pm-plasma", "case.toml"))
    d = derived(case)
    λp = 2π / sqrt(case["plasma"]["density"])
    codes = present_codes(runs)
    kinds = ["$c-$m" for c in codes for m in ("vacuum", "plasma")]
    S = Dict(k => load_summary(joinpath(runs, k)) for k in kinds)
    result = Dict{String,Any}("case" => case["name"], "dx" => d.dx, "dt" => d.dt, "lambda_p" => λp, "codes" => codes)
    for medium in ("vacuum", "plasma")
        result[medium] = compare_rows(S["pm-$medium"]["outputs"], S["smilei-$medium"]["outputs"], medium, d, λp)
        "picongpu" in codes &&
            (result["picongpu_$medium"] = compare_rows(S["pm-$medium"]["outputs"], S["picongpu-$medium"]["outputs"], medium, d, λp))
    end
    result["seconds"] = Dict(k => get(S[k], "seconds", nothing) for k in kinds)
    open(io -> JSON.print(io, result, 2), joinpath(out, "comparison.json"), "w")
    plot_laser(out, S, d, codes)
    plot_wake(out, S, case, d, λp, codes)
    plot_maps(out, runs, case, d, codes)
    plot_spectra(out, S, codes)
    write_manifest(out; root = REPO, extra = Dict{String,Any}("runs" => relpath(abspath(runs), REPO), "case" => case["name"]))
    return result
end

const COLORS = (pm = "#1f77b4", smilei = "#d62728", picongpu = "#2ca02c")
const NAMES = Dict("pm" => "ParticleMesh", "smilei" => "Smilei", "picongpu" => "PIConGPU")
const STYLES = Dict("pm" => :solid, "smilei" => :dash, "picongpu" => :dot)

function plot_laser(out, S, d, codes)
    fig = Figure(size = (1100, 420))
    side = Dict{String,Any}()
    for (col, medium) in enumerate(("vacuum", "plasma"))
        ax = Axis(fig[1, col]; xlabel = "t (1/ω₀)", ylabel = "max |E_y| in the box (mₑcω₀/e)", title = "laser amplitude, $medium")
        for key in codes
            o = S["$key-$medium"]["outputs"]
            t, a = Float64[x["t"] for x in o], Float64[x["Ey_absmax"] for x in o]
            scatterlines!(ax, t, a; label = NAMES[key], color = getfield(COLORS, Symbol(key)))
            side["$key-$medium"] = Dict("t" => t, "Ey_absmax" => a)
        end
        axislegend(ax; position = :lt)
    end
    save_plot(out, "laser_amplitude", fig, side)
end

function plot_wake(out, S, case, d, λp, codes)
    pm = S["pm-plasma"]["outputs"]
    others = [c for c in codes if c != "pm"]
    bystep = Dict(c => Dict(o["step"] => o for o in S["$c-plasma"]["outputs"]) for c in others)
    picks = pm[Int.(get(case["diagnostics"], "wake_outputs", [6, 9, 13]))]   # in the plasma: t ≈ 500, 800, 1300 for the proposed case
    fig = Figure(size = (1100, 300 * length(picks)))
    side = Dict{String,Any}()
    for (row, o) in enumerate(picks)
        mp = peak_m(o)
        lo, hi = mp - round(Int, 2.5λp / d.dx), mp + round(Int, 0.5λp / d.dx)
        for (col, (key, m0key, lab)) in enumerate((("Ex_axis", "Ex_m0", "E_x on axis"), ("rho_axis", "rho_m0", "ρ_e on axis (n_c e)")))
            ax = Axis(fig[row, col]; xlabel = "x (c/ω₀), lab frame", ylabel = lab, title = "t = $(round(o["t"]; digits = 1))")
            # the points every code holds, so that the curves share one x range
            l, h = lo, hi
            for c in others
                s = bystep[c][o["step"]]
                m = common_points(o[key], o[m0key], s[key], s[m0key]; mlo = l, mhi = h).m
                isempty(m) || ((l, h) = (first(m), last(m)))
            end
            entry = Dict{String,Any}()
            for c in codes
                s = c == "pm" ? o : bystep[c][o["step"]]
                v = common_points(s[key], s[m0key], o[key], o[m0key]; mlo = l, mhi = h)
                ξ = key == "Ex_axis" ? (v.m .- 0.5) .* d.dx : v.m .* d.dx
                lines!(ax, ξ, v.a; label = NAMES[c], color = getfield(COLORS, Symbol(c)), linestyle = STYLES[c])
                entry["x"] = ξ
                entry[c] = v.a
            end
            row == 1 && axislegend(ax; position = :lb)
            side["$(key)_step$(o["step"])"] = entry
        end
    end
    save_plot(out, "wake_axis", fig, side)
end

function plot_maps(out, runs, case, d, codes)
    stride = Int.(case["diagnostics"]["map_stride"])
    steps = output_steps(case)[Int.(case["diagnostics"]["map_outputs"])]
    k = last(steps)
    read_map(code) = h5open(joinpath(runs, "$code-plasma", "maps.h5"), "r") do f
        g = f["step$k"]
        (rho = read(g["rho"]), m0 = read_attribute(g, "m0"))
    end
    maps = [read_map(c) for c in codes]
    lo, hi = maximum(a.m0 for a in maps), minimum(a.m0 + size(a.rho, 1) for a in maps) - 1
    n = length(lo:hi)
    sel = 1:stride[1]:n
    ys = 1:stride[2]:d.ny
    x, y = collect(lo:hi)[sel] .* d.dx, (collect(ys) .- 1) .* d.dy
    n0 = case["plasma"]["density"]
    fig = Figure(size = (1100, 350 * length(codes)))
    side = Dict{String,Any}("x" => x, "y" => y, "t" => k * d.dt)
    for (row, (a, c)) in enumerate(zip(maps, codes))
        M = a.rho[((lo:hi) .- a.m0 .+ 1)[sel], ys]
        ax = Axis(fig[row, 1]; xlabel = "x (c/ω₀), lab frame", ylabel = "y (c/ω₀)", title = "$(NAMES[c]): n_e/n₀ at t = $(round(k * d.dt; digits = 1))")
        hm = heatmap!(ax, x, y, -M ./ n0; colorrange = (0, 3), colormap = :viridis)
        Colorbar(fig[row, 2], hm)
        side[NAMES[c]] = -M ./ n0
    end
    save_plot(out, "density_maps", fig, side)
end

# PIConGPU's histogram covers its whole simulated domain, the hidden row of cold plasma
# included, so its fractions in the hot bins are smaller than the other codes' by the ratio of
# the electron weights the domains hold (scripts/lwfa/picongpu.jl).
function plot_spectra(out, S, codes)
    edges = Float64.(S["pm-plasma"]["spectrum_edges"])
    mids = (edges[1:(end - 1)] .+ edges[2:end]) ./ 2
    fig = Figure(size = (700, 420))
    ax = Axis(fig[1, 1]; xlabel = "E_kin (mₑc²)", ylabel = "fraction of the electron weight per bin", yscale = log10,
              title = "electron spectrum at the end (bins of $(round(edges[2] - edges[1]; digits = 3)) mₑc²)")
    side = Dict{String,Any}("ekin" => mids)
    for key in codes
        h = Float64.(last(S["$key-plasma"]["outputs"])["spectrum"])
        isempty(h) && continue
        norm = sum(h)
        hn = norm > 0 ? h ./ norm : h
        ok = hn .> 0
        any(ok) && scatterlines!(ax, mids[ok], hn[ok]; label = NAMES[key], color = getfield(COLORS, Symbol(key)))
        side[key] = hn
    end
    axislegend(ax)
    save_plot(out, "spectrum", fig, side)
end

# --- PIConGPU -------------------------------------------------------------------------------------

include(joinpath(@__DIR__, "picongpu.jl"))

# --- command line ---------------------------------------------------------------------------------

if abspath(PROGRAM_FILE) == @__FILE__
    cmd = get(ARGS, 1, "")
    if cmd == "prepare"
        println(prepare(ARGS[2], ARGS[3]))
    elseif cmd == "run-pm"
        run_pm(ARGS[2])
    elseif cmd == "summarize-smilei"
        summarize_smilei(ARGS[2])
    elseif cmd == "summarize-picongpu"
        summarize_picongpu(ARGS[2])
    elseif cmd == "compare"
        compare(ARGS[2], ARGS[3])
    else
        error("usage: lwfa.jl prepare <case> <tag> | run-pm <dir> | summarize-smilei <dir> | summarize-picongpu <dir> | compare <runs> <out>")
    end
end
