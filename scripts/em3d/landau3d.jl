# The full-size 3D Landau case of WP-042: a Langmuir wave of mode (1, 1, 1), oblique to all three
# axes, in a periodic cube, against the relativistic linear theory of its Maxwell–Jüttner load.
# The case is T-954's (lib/ParticleMeshElectromagnetic/test/em3d_step.jl, `landau_3d`) at a size
# chosen on the command line: a cube of side 2π√3/|k| with |k| = 10 ω₀/c, v_th = 0.05 c, a 10 %
# density perturbation, the quadratic shape, Reference mode, Δt = 0.95 of the 3D Courant limit.
# The observable is WP-023's `damping_rate_mode_ensemble`: the fit of |⟨k̂·Ê⟩_seeds|² at its
# maxima after t = 3 above ten times the noise floor (scripts/crosscode/ensemble.jl), with the
# jackknife over seeds as its error; per seed the same rule on one seed's |k̂·Ê|².
#
#   julia +1.13 --project=scripts scripts/em3d/landau3d.jl run cells=64 ppc=6 seeds=20261031,20262031 float=32 backend=cuda t_end=24
#   julia +1.13 --project=scripts scripts/em3d/landau3d.jl table <run dir>...
#
# `run` writes scripts/runs/em3d-landau3d-<cells>c<ppc>p-f<float>-<timestamp>/: manifest.toml (F3),
# seed-<seed>/series.json (the mode coefficients and energies of every step) per seed,
# observables.json (theory, the ensemble and per-seed rates, the checks, the cost) and
# landau3d.png with its sidecar landau3d.json. `table` prints the Markdown table of the run
# directories' observables, for the report.
include(joinpath(@__DIR__, "..", "em1d", "common.jl"))
include(joinpath(REPO, "scripts", "crosscode", "ensemble.jl"))
using .Ensemble: ensemble_damping
using Statistics: mean, std

const FIT_START, FLOOR_FACTOR = 3.0, 10.0

"The KernelAbstractions backend of a name: `cpu`, or `cuda` (CUDACore is loaded then); call at top level (world age)."
function backend_named(name)
    name == "cpu" && return CPU()
    name == "cuda" || error("backend $name: cpu or cuda")
    @eval using CUDACore: CUDACore
    return @eval CUDACore.CUDABackend()
end

backend_label(be) = be isa CPU ? "KernelAbstractions CPU" : string(nameof(typeof(be)))
gpu_name(be) = be isa CPU ? "none" : string(CUDACore.name(CUDACore.device()))
const KMAG, VTH, AMPLITUDE = 10, 1 / 20, 1 / 10

"T-954's problem at `cells`³ cells, `ppc`³ electrons per cell and t_end, element type T."
function landau3d_problem(::Type{T}; cells, ppc, seed, t_end) where {T}
    L = T(2π * sqrt(3) / KMAG)
    dt = T(95 // 100) * inv(sqrt(3 * (cells / L)^2))
    nsteps = round(Int, t_end / dt)
    e = SpeciesDescription{T}(:electron, -one(T), one(T), one(T), T(AMPLITUDE), (1, 1, 1), (ppc, ppc, ppc),
                              (zero(T), zero(T), zero(T)), (T(VTH), T(VTH), T(VTH)), 1)
    return EMProblem("landau-oblique-3d", (L, L, L), (cells, cells, cells), dt, nsteps, (e,), seed, (1, 1, 1))
end

"""
    gauss_3d(state) -> (; gauss, divB)

T-950's Gauss check at the end of a run: max|Σ_d D⁻_d E_d − (ρ − mean ρ)| over the interior nodes
divided by max|ρ| (ρ deposited now at the primal nodes in the state's mode), and
max|Σ_d D⁺_d B_d| at the cell centres divided by (Σ_d 2/Δ_d)·max|B|. Host copies.
"""
function gauss_3d(st)
    (; grid, numerics) = st
    T = eltype(st.ρ)
    g, n, h = guards(grid), ncells(grid), cellsize(grid)
    I = CartesianIndices(ntuple(d -> (g + 1):(g + n[d]), 3))
    e = ntuple(d -> CartesianIndex(ntuple(a -> Int(a == d), 3)), 3)
    ρs = field(get_backend(st.ρ), T, grid)
    deposit!(numerics.mode, ρs, only(st.species), numerics.shape, (Node(), Node(), Node()), grid, only(st.ws))
    fold_guards!(ρs, grid)
    ρ = Float64.(Array(ρs))
    E = map(c -> Float64.(Array(getfield(st.fields, c))), (:Ex, :Ey, :Ez))
    B = map(c -> Float64.(Array(getfield(st.fields, c))), (:Bx, :By, :Bz))
    hd = Float64.(h)
    ρ̄ = sum(ρ[I]) / prod(n)
    res = maximum(abs(sum(d -> (E[d][j] - E[d][j - e[d]]) / hd[d], 1:3) - (ρ[j] - ρ̄)) for j in I)
    divB = maximum(abs(sum(d -> (B[d][j + e[d]] - B[d][j]) / hd[d], 1:3)) for j in I)
    bmax = maximum(b -> maximum(abs, b[I]), B)
    return (; gauss = res / maximum(abs, ρ[I]), divB = divB / (sum(d -> 2 / hd[d], 1:3) * bmax))
end

"One seed: run it, write seed-<seed>/series.json, return its mode coefficient and record."
function run_seed(dir, ::Type{T}, be; cells, ppc, seed, t_end) where {T}
    prob = landau3d_problem(T; cells, ppc, seed, t_end)
    st = EMState(prob, EMNumerics(; shape = Quadratic, mode = Reference()); backend = be)
    initialize!(st, prob)
    p = params(prob)
    run!(st, p, 1)                      # compiles the step; its entry is part of the run
    wall = @elapsed begin
        run!(st, p, prob.nsteps - 1)
        KernelAbstractions.synchronize(be)
    end
    h = histories(st)
    ts = collect((0:(prob.nsteps - 1)) .* Float64(prob.dt))
    k̂ = (1, 1, 1) ./ sqrt(3)
    El = complex.(Float64.(h.El_re), Float64.(h.El_im))
    Ê = map(c -> complex.(Float64.(getfield(h, Symbol(c, :_re))), Float64.(getfield(h, Symbol(c, :_im)))), (:Ex, :Ey, :Ez))
    Et = sqrt.(sum(d -> abs2.(Ê[d] .- k̂[d] .* El), 1:3))
    Wtot = Float64.(total_energy_history(h))
    chk = gauss_3d(st)
    np = alive_count(only(st.species))
    sdir = joinpath(dir, "seed-$seed")
    mkpath(sdir)
    open(joinpath(sdir, "series.json"), "w") do io
        JSON.print(io, Dict("t" => ts, "El_re" => real.(El), "El_im" => imag.(El),
                            "WE" => Float64.(h.WEx .+ h.WEy .+ h.WEz), "WB" => Float64.(h.WBx .+ h.WBy .+ h.WBz),
                            "K" => Float64.(h.K)))
    end
    rec = Dict{String,Any}("seed" => seed, "steps" => prob.nsteps, "particles" => np, "wall_seconds" => wall,
        "ns_per_particle_step" => 1e9 * wall / (np * (prob.nsteps - 1)),
        "transverse" => maximum(Et) / maximum(abs, El),
        "energy_drift" => (Wtot[end] - Wtot[1]) / Wtot[1], "gauss_end" => chk.gauss, "divB_end" => chk.divB)
    println("seed $seed: $(round(wall; digits = 1)) s, $np particles, $(prob.nsteps) steps, Gauss $(chk.gauss), ",
            "energy drift $(rec["energy_drift"])")
    flush(stdout)
    return ts, El, rec, prob
end

function landau3d_run(be; cells, ppc, seeds, float, t_end)
    T = float == 32 ? Float32 : float == 64 ? Float64 : error("float = 32 or 64")
    gpu = gpu_name(be)
    params_ = Dict{String,Any}("cells" => cells, "ppc" => ppc, "seeds" => seeds, "element_type" => string(T), "t_end" => t_end,
        "k" => KMAG, "vth" => VTH, "amplitude" => AMPLITUDE, "mode" => "Reference", "shape" => "Quadratic", "kmode" => [1, 1, 1],
        "fit_start" => FIT_START, "floor_factor" => FLOOR_FACTOR)
    dir = run_dir("landau3d-$(cells)c$(ppc)p-f$float", @__FILE__, nothing; params = params_, work_package = "WP-042",
                  backend = backend_label(be), extra = Dict{String,Any}("gpu" => gpu), prefix = "em3d",
                  extra_inputs = [joinpath(REPO, "scripts", "crosscode", "ensemble.jl")])
    println("# run directory: $(relpath(dir, REPO))")
    coeffs, recs, ts, prob = Vector{ComplexF64}[], Dict{String,Any}[], nothing, nothing
    for s in seeds
        ts, El, rec, prob = run_seed(dir, T, be; cells, ppc, seed = s, t_end)
        push!(coeffs, El)
        push!(recs, rec)
        GC.gc()
    end
    th = landau_rate_relativistic(Float64(KMAG), VTH^2; load = :maxwell_juttner)
    ens = ensemble_damping(ts, coeffs; fit_start = FIT_START, floor_factor = FLOOR_FACTOR)
    single = [r.gamma for r in ens.single_seed]
    L = 2π * sqrt(3) / KMAG
    out = Dict{String,Any}(
        "theory" => Dict("gamma" => th.gamma, "omega" => th.omega, "source" => "landau_rate_relativistic, Maxwell–Jüttner, |k| = $KMAG, v_th = $VTH"),
        "size" => Dict("cells" => cells, "ppc" => ppc^3, "particles" => recs[1]["particles"], "steps" => prob.nsteps,
                       "dt" => Float64(prob.dt), "dx_over_debye" => L / cells / VTH, "element_type" => string(T)),
        "ensemble" => Dict("gamma" => ens.gamma, "gamma_rel" => (ens.gamma - th.gamma) / th.gamma, "jackknife" => ens.jackknife.stderr,
                           "jackknife_rel" => ens.jackknife.stderr / th.gamma, "omega" => ens.omega,
                           "omega_rel" => (ens.omega - th.omega) / th.omega, "maxima" => length(ens.t_fit),
                           "t_fit" => ens.t_fit, "seeds" => length(seeds)),
        "single_seed" => Dict("gamma" => single, "mean" => mean(single), "std" => length(single) > 1 ? std(single) : NaN,
                              "mean_rel" => (mean(single) - th.gamma) / th.gamma,
                              "omega" => [r.omega for r in ens.single_seed], "maxima" => [length(r.t_fit) for r in ens.single_seed]),
        "per_seed" => recs)
    open(io -> JSON.print(io, out, 2), joinpath(dir, "observables.json"), "w")
    landau3d_figure(dir, ts, coeffs, ens, th, cells, ppc, T)
    write_manifest(dir; root = REPO, extra = Dict{String,Any}("finished" => string(now(UTC)) * "Z",
        "outputs" => Dict{String,Any}(f => "sha256:" * RunManifest.file_hash(joinpath(dir, f))
                                      for f in ("observables.json", "landau3d.png", "landau3d.json"))))
    e = out["ensemble"]
    println("ensemble γ = $(e["gamma"]) ± $(e["jackknife"]) (theory $(th.gamma), rel $(e["gamma_rel"])), ω = $(e["omega"]) ",
            "(theory $(th.omega)), $(e["maxima"]) maxima; per-seed γ $(out["single_seed"]["mean"]) ± $(out["single_seed"]["std"])")
    return dir
end

function landau3d_figure(dir, ts, coeffs, ens, th, cells, ppc, T)
    fig = Figure(size = (900, 520))
    ax = Axis(fig[1, 1]; yscale = log10, xlabel = "t ω₀", ylabel = "|k̂·Ê|² of mode (1, 1, 1)",
              title = "3D Landau damping, $(cells)³ cells, $(ppc)³ per cell, $(length(coeffs)) seeds, $T")
    for c in coeffs
        lines!(ax, ts, abs2.(c); color = (:gray, 0.4), linewidth = 0.7)
    end
    lines!(ax, ts, ens.U; color = :black, linewidth = 1.2, label = "seed mean")
    scatter!(ax, ens.t_fit, ens.U_fit; color = :red, markersize = 8, label = "fitted maxima, γ = $(round(ens.gamma; sigdigits = 4))")
    t0 = first(ens.t_fit)
    tl = range(t0, last(ts); length = 50)
    lines!(ax, tl, first(ens.U_fit) .* exp.(-2th.gamma .* (tl .- t0)); color = :blue, linestyle = :dash,
           label = "theory, γ = $(round(th.gamma; sigdigits = 5))")
    hlines!(ax, [ens.floor]; color = :orange, linestyle = :dot, label = "noise floor")
    axislegend(ax; position = :rt)
    save(joinpath(dir, "landau3d.png"), fig)
    open(joinpath(dir, "landau3d.json"), "w") do io
        JSON.print(io, Dict("t" => ts, "ensemble_energy" => ens.U, "t_fit" => ens.t_fit, "U_fit" => ens.U_fit,
                            "gamma" => ens.gamma, "gamma_theory" => th.gamma, "floor" => ens.floor), 2)
    end
end

"The Markdown table of the run directories' observables."
function landau3d_table(dirs)
    println("| size | particles | Δx/λ_D | steps | ensemble γ ± jackknife | rel. to theory | ω rel. | maxima | per-seed γ, mean ± std | Gauss, ∇·B at end (max) | energy drift (max) | s per seed |")
    println("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for d in dirs
        o = JSON.parsefile(joinpath(d, "observables.json"))
        s, e, ss, ps = o["size"], o["ensemble"], o["single_seed"], o["per_seed"]
        f(x; d = 4) = string(round(x; sigdigits = d))
        println("| $(s["cells"])³, $(s["ppc"]) per cell, $(length(ps)) seeds, $(s["element_type"]) | $(s["particles"]) | $(f(s["dx_over_debye"]; d = 3)) | $(s["steps"]) | ",
                "$(f(e["gamma"])) ± $(f(e["jackknife"]; d = 2)) | $(f(100 * e["gamma_rel"]; d = 3)) % ± $(f(100 * e["jackknife_rel"]; d = 2)) % | ",
                "$(f(100 * e["omega_rel"]; d = 2)) % | $(e["maxima"]) | $(f(ss["mean"])) ± $(f(ss["std"]; d = 2)) | ",
                "$(f(maximum(p -> p["gauss_end"], ps); d = 2)), $(f(maximum(p -> p["divB_end"], ps); d = 2)) | ",
                "$(f(maximum(p -> abs(p["energy_drift"]), ps); d = 2)) | $(round(Int, mean(p -> p["wall_seconds"], ps))) |")
    end
    th = JSON.parsefile(joinpath(first(dirs), "observables.json"))["theory"]
    println("\nTheory: γ = $(th["gamma"]), ω = $(th["omega"]) ($(th["source"])).")
end

function main(args)
    isempty(args) && error("usage: run key=value... | table <dir>...")
    cmd = args[1]
    if cmd == "run"
        kv = Dict(split(a, "="; limit = 2) for a in args[2:end])
        be = backend_named(get(kv, "backend", "cuda"))
        Base.invokelatest(landau3d_run, be; cells = parse(Int, kv["cells"]), ppc = parse(Int, kv["ppc"]),
                          seeds = parse.(Int, split(kv["seeds"], ",")), float = parse(Int, get(kv, "float", "32")),
                          t_end = parse(Float64, get(kv, "t_end", "24")))
    elseif cmd == "table"
        landau3d_table(args[2:end])
    else
        error("unknown command $cmd")
    end
end

abspath(PROGRAM_FILE) == (@__FILE__) && main(ARGS)
