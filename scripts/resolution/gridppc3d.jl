# The grid and particle budget of the trapped-beam case in 3D: the case of WP-065 (WP-064's
# ionization-injection case at a fixed laser energy, the first bucket's trapped electrons as the
# objective J) on a 3D box, a round beam polarized along y, the dopant on a square tube about the
# axis, run over the cell size along x, the transverse cell (Δy = Δz), the transverse box, the
# particles per cell, the field solver and the gather, with ForwardDiff duals in the four
# parameters of the optimization (duration, waist, focus, density) where a job asks for the
# gradient. Each run gives J, the bunch, the time and the memory it took.
#
#   julia +1.13 --project=scripts -t 16 scripts/resolution/gridppc3d.jl campaign <name>[,<name>...] <parent dir> [cpu|cuda]
#   julia +1.13 --project=scripts -t 16 scripts/resolution/gridppc3d.jl job <dir> <job JSON> [cpu|cuda]
#   julia +1.13 --project=scripts scripts/resolution/gridppc3d.jl list <name>
#
# The 3D case keeps the 2D case's longitudinal geometry (the window of 14 λ₀, the vacuum, the
# ramp, the dopant's section from 108 to 132, the pulse's place, focus and timing, t = 300) and
# its transverse box (128 cells of π/4 = 100.5 c/ω₀ with 12 PML cells at each edge) along y and z.
# The energy budget is the 3D pulse's (`laser_energy` of a round beam, ∝ a₀²·w²·τ) at a₀ = 1.9,
# τ = 6, w = 10, so the base point has the 2D case's a₀ and peak intensity. J is
#
#     J = Δx·Δy·Δz · Σ_i w_i·(γ_i − 1)·S(x_i)·R(u_x,i)
#
# in n_c·mₑc²·(c/ω₀)³, the 2D case's sum with the cell volume for its area. The campaigns, the job
# files, PAUSE and the read-outs work as in gridppc.jl, whose helpers this file includes. The
# numbers are measurements of single runs, not independently reviewed.

include(joinpath(@__DIR__, "gridppc.jl"))

"The defaults of a 3D job; a job's own keys replace them. `nt` refines the transverse cell (Δy = Δz = π/(4·nt)), `rows` the transverse cells at nt = 1."
const JOB3_DEFAULTS = Dict{String,Any}("ppl" => 16, "nt" => 1, "rows" => 128, "ppc" => [2, 2, 2], "ion_ppc" => [1, 1, 1], "T" => "f64",
                                       "mode" => "reference", "solver" => "yee", "gather" => "compensated", "cfl" => 95, "law" => "adk",
                                       "every" => 1, "a0E" => 1.9, "x" => collect(TCASE.x0), "grad" => false, "series" => true,
                                       "frames" => 0, "t_end" => BCASE.t_end, "capacity_factor" => 1.0)

"The energy of the 3D case's round pulse (`laser_energy` of a beam with a centre of two coordinates)."
pulse_energy3(a0, duration, waist) = π * sqrt(π / 2) * a0^2 * waist^2 * duration / 4

"The 3D energy budget: the pulse's energy at a₀ = `a0E` and the case's base duration and waist."
budget3(a0E = TCASE.a0_energy) = pulse_energy3(a0E, TCASE.x0[1], TCASE.x0[2])

"The a₀ of a round pulse of `duration` and `waist` that carries `energy`."
a0_for_energy3(duration, waist; energy) = sqrt(energy / pulse_energy3(one(duration), duration, waist))

"""
    study3d_problem(x, a0E; T, ppl, nt, rows, ppc, ion_ppc, cfl, law, every, capacity, t_end)

The 3D case at x = (duration, waist, focus, delay, density) and the 3D budget of `a0E`: Δx = λ₀/`ppl`,
Δy = Δz = π/(4·`nt`) on `rows`·`nt` cells along each transverse axis, the plasma's and the dopant's
particles per cell, the time step as `cfl` hundredths of the Courant limit, the rate law and the
element type `T` (that of the parameters, which are converted to `T` or are duals of it).
"""
function study3d_problem(x, a0E; T = Float64, ppl::Integer, nt::Integer = 1, rows::Integer = 128, ppc = (2, 2, 2), ion_ppc = (1, 1, 1),
                         cfl::Integer = 95, law = "adk", every::Integer = 1, capacity::Integer, t_end = BCASE.t_end)
    duration, waist, focus, delay, density = x
    a0 = a0_for_energy3(duration, waist; energy = T(budget3(a0E)))
    dopant = T(TCASE.dopant_fraction) * density
    R = promote_type(typeof(a0), typeof(dopant), typeof(duration), typeof(density), typeof(waist), typeof(delay), typeof(focus), T)
    ForwardDiff.valtype(R) === T || error("study3d_problem: the run's element type is $R, asked for $T")
    z3 = (zero(R), zero(R), zero(R))
    h = 2 * R(π) / ppl
    ht = R(π) / (4 * nt)
    cells = (BCASE.length_wavelengths * ppl, rows * nt, rows * nt)
    dt = R(cfl) / 100 / sqrt(1 / h^2 + 2 / ht^2)
    nsteps = ceil(Int, t_end / Float64(value_of(dt)))
    yc = cells[2] * Float64(value_of(ht)) / 2
    hw = BCASE.half_rows * π / 4
    pulse = GaussianLaser(; a0 = R(a0), waist = R(waist), duration = R(duration), delay = R(delay), focus = R(focus),
                          position = max(12, 192 ÷ ppl) * (2 * R(π) / 32), center = (cells[2] * ht / 2, cells[3] * ht / 2))
    e = SpeciesDescription{R}(:electron, -one(R), one(R), R(density), zero(R), (1, 0, 0), Tuple(ppc), z3, z3, 1)
    ion = FieldIonization(; density = R(dopant), levels = study_levels(R, law), capacity, every, ppc = Tuple(ion_ppc),
                          rows = ((yc - hw, yc + hw), (yc - hw, yc + hw)),
                          section = DopantSection(; start = R(BCASE.section.start), stop = R(BCASE.section.stop), ramp = R(BCASE.section.ramp)))
    return EMProblem("gridppc3d", (cells[1] * h, cells[2] * ht, cells[3] * ht), cells, dt, nsteps, (e,), 1, (1, 0, 0);
                     lasers = (pulse,), boundaries = (:window, :open, :open), window = MovingWindow(; start = R(BCASE.start)),
                     profile = DensityRamp(; vacuum = R(BCASE.vacuum), ramp = R(BCASE.ramp)), ionization = ion)
end

"The 3D case's numerics: the 2D case's with PMLs of 12·`nt` cells on the four open faces."
study3d_numerics(ppl, nt; shape = 2, mode = "reference", gather = "compensated", solver = "yee") =
    EMNumerics(; shape = shape == 2 ? Quadratic : shape == 1 ? Linear : error("shape order $shape: 1 or 2"),
               mode = mode == "reference" ? Reference() : mode == "fast" ? Fast() : error("mode $mode: reference or fast"),
               gather = pick(GATHERS, gather, "gather"), solver = pick(SOLVERS, solver, "solver"),
               absorber = PML(; cells = BCASE.pml_y * nt), back_absorber = PML(; cells = max(4, round(Int, BCASE.pml_x_wavelengths * ppl))))

"""
The electron slots of a 3D job: the dopant lattice of the window (a birth step offers one slot per atom slot) and room for the
electrons alive, as the 2D case's rule (six times the atoms at λ₀/16, growing with the cells along x) times `capacity_factor`.
"""
function study3d_capacity(job)
    ppl, nt, ip = Int(job["ppl"]), Int(job["nt"]), Int.(job["ion_ppc"])
    atoms = BCASE.length_wavelengths * ppl * (2 * BCASE.half_rows * nt)^2 * prod(ip)
    haskey(job, "capacity") && return Int(job["capacity"])
    return atoms + round(Int, Float64(job["capacity_factor"]) * max(400_000, 6 * atoms * (ppl / 16) / Int(job["every"])))
end

# --- read-outs -----------------------------------------------------------------------------------

"The on-axis line of a 3D field over the window's columns, on the host as Float64."
function axis_line3(st, a)
    g, n = guards(st.grid), ncells(st.grid)
    return f64(view(a, (g + 1):(g + n[1]), g + 1 + n[2] ÷ 2, g + 1 + n[3] ÷ 2))
end

"The parts of J (the born electrons', the plasma's) on a 3D grid: the 2D sum of `trapped2d_case.jl` with the cell's volume."
function trapped_parts3(st; fade = TCASE.fade, rise = TCASE.rise)
    check_fade(st, fade)
    check_rise(rise)
    h = cellsize(st.grid)
    v = h[1] * h[2] * h[3]
    born = v * species_trapped(ionization_electrons(st), h[1], fade[1], fade[2], rise[1], rise[2])
    plasma = v * foldl(+, map(s -> species_trapped(s, h[1], fade[1], fade[2], rise[1], rise[2]), st.species))
    return (; born, plasma)
end

"The alive electrons of a species as host arrays of plain numbers: window x, y, z, the momenta and the weight."
function particle_arrays3(e, h)
    alive = Array(e.status) .== ALIVE
    pos(d) = (Array(e.cell[d])[alive] .- 1 .+ f64(e.offset[d])[alive]) .* h[d]
    return (; x = pos(1), y = pos(2), z = pos(3), ux = f64(e.u[1])[alive], uy = f64(e.u[2])[alive], uz = f64(e.u[3])[alive], w = f64(e.w)[alive])
end

"""
The first bucket's trapped electrons (born and the plasma's) as a bunch, with the objective's weights w·S·R: their charge in
n_c·(c/ω₀)³, the mean and rms of γ − 1, the rms of y and z and of u_y and u_z about their means and the normalized rms emittances,
the effective number of J's summands, and the spectrum over γ − 1 in bins of 1 from 0 to 60; the born electrons' share of J.
"""
function bunch3(st; fade = TCASE.fade)
    h = Float64.(value_of.(cellsize(st.grid)))
    vol = prod(h)
    sets = [particle_arrays3(ionization_electrons(st), h); [particle_arrays3(s, h) for s in st.species]]
    cat(f) = reduce(vcat, map(f, sets))
    x, y, z, ux, uy, uz, w = cat(b -> b.x), cat(b -> b.y), cat(b -> b.z), cat(b -> b.ux), cat(b -> b.uy), cat(b -> b.uz), cat(b -> b.w)
    nborn = length(sets[1].w)
    g = sqrt.(1 .+ ux .^ 2 .+ uy .^ 2 .+ uz .^ 2) .- 1
    s = w .* smooth_rise.(x, fade[1], fade[2]) .* smooth_rise.(ux, TCASE.rise[1], TCASE.rise[2])
    q = s .* g
    W, Q = sum(s; init = 0.0), sum(q; init = 0.0)
    out = Dict{String,Any}("born_alive" => nborn, "weight" => vol * W, "born_share_J" => Q > 0 ? sum(@view q[1:nborn]) / Q : nothing)
    spec = zeros(60)
    for i in eachindex(g)
        j = floor(Int, g[i]) + 1
        1 <= j <= 60 && (spec[j] += vol * s[i])
    end
    out["spectrum"] = spec
    W > 0 || return out
    m(v) = sum(s .* v) / W
    eg = m(g)
    emit(a, u) = (ea = m(a); eu = m(u); va = m((a .- ea) .^ 2); vu = m((u .- eu) .^ 2); c = m((a .- ea) .* (u .- eu)); (sqrt(va), sqrt(vu), sqrt(max(0.0, va * vu - c^2))))
    ry, ruy, ey = emit(y, uy)
    rz, ruz, ez = emit(z, uz)
    merge!(out, Dict{String,Any}("mean_energy" => eg, "rms_energy" => sqrt(m((g .- eg) .^ 2)), "rms_y" => ry, "rms_z" => rz, "rms_uy" => ruy,
                                 "rms_uz" => ruz, "emittance_y" => ey, "emittance_z" => ez, "neff_J" => Q^2 / sum(abs2, q), "mean_x" => m(x),
                                 "ux_max" => maximum(ux; init = 0.0)))
    return out
end

"The laser's on-axis envelope peak and place, the energy centroid, the wake's first zero crossings behind the pulse and the window's lab x after step `k` (as `axis_readout` in 2D)."
function axis_readout3(st, k, ppl)
    h = Float64(value_of(cellsize(st.grid)[1]))
    n = ncells(st.grid)[1]
    Ey, Ex = axis_line3(st, st.fields.Ey), axis_line3(st, st.fields.Ex)
    L = n * h
    env = envelope(Ey, max(1, round(Int, 0.3 * L / 2π)), round(Int, 3 * L / 2π))
    ip, a_peak = parabolic_peak(env)
    e2 = env .^ 2
    xs = (1:n) .* h
    xc = sum(xs .* e2) / sum(e2)
    half = ppl ÷ 2
    sm = [mean(@view Ex[max(1, i - half):min(n, i + half)]) for i in 1:n]
    ic = clamp(round(Int, xc / h), 2, n)
    zeros_x = Float64[]
    for i in ic:-1:2
        if sm[i - 1] * sm[i] < 0
            push!(zeros_x, (i - 1 + sm[i - 1] / (sm[i - 1] - sm[i])) * h)
            length(zeros_x) == 3 && break
        end
    end
    lo = max(1, ic - round(Int, 40 / h))
    return Dict{String,Any}("a_peak" => a_peak, "x_peak" => ip * h, "x_centroid" => xc, "Ex_zeros" => zeros_x, "Ex_min" => minimum(@view sm[lo:ic]),
                            "x0" => window_shifts(st, k) * h)
end

"One sample of a 3D run after step `k`: the axis read-out, J and its parts, the born electrons alive and their weight."
function sample3(st, k, dt, ppl)
    out = axis_readout3(st, k, ppl)
    parts = trapped_parts3(st)
    h = Float64.(value_of.(cellsize(st.grid)))
    b = particle_arrays3(ionization_electrons(st), h)
    merge!(out, Dict{String,Any}("t" => k * dt, "step" => k, "J" => Float64(value_of(parts.born + parts.plasma)), "J_plasma" => Float64(value_of(parts.plasma)),
                                 "born_alive" => length(b.w), "born_weight" => prod(h) * sum(b.w; init = 0.0), "ux_max" => maximum(b.ux; init = 0.0)))
    return out
end

"""
Maps of a 3D frame on rasters of `nx × ny` points (nearest cell): E_y and the electron density over the plasma's in the plane
z = z_c (polarization plane) and in the plane y = y_c, and E_y on the transverse plane through the first bucket (window x = 36).
"""
function frame_maps3(st, ρbuf, density, nx, ny)
    g, n = guards(st.grid), ncells(st.grid)
    I = [g + clamp(round(Int, (i - 0.5) * n[1] / nx + 0.5), 1, n[1]) for i in 1:nx]
    Jy = [g + clamp(round(Int, (j - 0.5) * n[2] / ny + 0.5), 1, n[2]) for j in 1:ny]
    Jz = [g + clamp(round(Int, (j - 0.5) * n[3] / ny + 0.5), 1, n[3]) for j in 1:ny]
    jc, kc = g + 1 + n[2] ÷ 2, g + 1 + n[3] ÷ 2
    window_charge!(ρbuf, st)
    Ey = Array(st.fields.Ey)
    ne = -Array(ρbuf) ./ density
    ib = g + clamp(round(Int, 36 / Float64(value_of(cellsize(st.grid)[1]))), 1, n[1])
    Ex = Array(st.fields.Ex)
    return (Float32.(Ey[I, Jy, kc]), Float32.(ne[I, Jy, kc]), Float32.(Ey[I, jc, Jz]), Float32.(ne[I, jc, Jz]), Float32.(Ex[ib, Jy, Jz]),
            Float32.(ne[ib, Jy, Jz]))
end

# --- a job -----------------------------------------------------------------------------------------

"""
    run_job3(job, dir) -> Dict

One run of the 3D case as the job describes it (`JOB3_DEFAULTS`): J and its parts, the bunch, the time series every 5 c/ω₀ and,
with `frames`, that many frames (`<id>.frames.bin`, six Float16 maps each, `frame_maps3`); a `grad` job runs with duals in duration,
waist, focus and density and gives J with its gradient. `seconds` is the wall time of the state's construction, the load and the
steps, `state_mib` the host state's size (CPU), `gpu_peak_used_mib` the device pool's high-water mark (CUDA).
"""
function run_job3(job0, dir)
    job = merge(JOB3_DEFAULTS, job0)
    T = eltype_named(job["T"])
    ppl, nt = Int(job["ppl"]), Int(job["nt"])
    x64, a0E = Float64.(job["x"]), Float64(job["a0E"])
    grad = job["grad"] == true
    x = if grad
        ntuple(i -> (q = findfirst(==(i), GRAD_DIRS); q === nothing ? T(x64[i]) : ForwardDiff.Dual{Nothing}(T(x64[i]), ntuple(j -> T(j == q), length(GRAD_DIRS)))), 5)
    else
        Tuple(T.(x64))
    end
    capacity = study3d_capacity(job)
    prob = study3d_problem(x, T(a0E); T, ppl, nt, rows = Int(job["rows"]), ppc = Int.(job["ppc"]), ion_ppc = Int.(job["ion_ppc"]),
                           cfl = Int(job["cfl"]), law = job["law"], every = Int(job["every"]), capacity, t_end = Float64(job["t_end"]))
    numerics = study3d_numerics(ppl, nt; mode = job["mode"], gather = job["gather"], solver = job["solver"])
    reset_pool_peaks!()
    t0 = time()
    st = EMState(prob, numerics; backend = BE, record = ())
    p = params(prob)
    initialize!(st, p)
    KernelAbstractions.synchronize(BE)
    seconds_init = time() - t0
    state_mib = BACKEND_NAME == "cpu" ? mib(Base.summarysize(st)) : nothing
    dt = Float64(value_of(prob.dt))
    nframes = Int(job["frames"])
    frame_steps = nframes > 0 ? unique(round.(Int, range(prob.nsteps / nframes, prob.nsteps; length = nframes))) : Int[]
    sample_steps = job["series"] == true && !grad ? unique(round.(Int, (5.0:5.0:Float64(job["t_end"])) ./ dt)) : Int[]
    stops = sort(unique(vcat(frame_steps, sample_steps, prob.nsteps)))
    frame_io = nframes > 0 ? open(joinpath(dir, job["id"] * ".frames.bin"), "w") : nothing
    ρbuf = nframes > 0 ? field(BE, eltype(st.fields.Ex), st.grid) : nothing
    series, frames_meta = Dict{String,Any}[], Dict{String,Any}[]
    FX, FY = 448, 128
    t_steps = 0.0
    k = 0
    for stop in stops
        ts = time()
        stop > k && run!(st, p, stop - k; first = k + 1)
        KernelAbstractions.synchronize(BE)
        t_steps += time() - ts
        k = stop
        stop in sample_steps && push!(series, sample3(st, k, dt, ppl))
        if stop in frame_steps
            foreach(m -> write(frame_io, Float16.(clamp.(m, -6.0f4, 6.0f4))), frame_maps3(st, ρbuf, x64[5], FX, FY))
            h = Float64.(value_of.(cellsize(st.grid)))
            b = particle_arrays3(ionization_electrons(st), h)
            keep = length(b.w) <= 6000 ? collect(eachindex(b.w)) : collect(1:(length(b.w) ÷ 6000):length(b.w))
            push!(frames_meta, Dict{String,Any}("t" => k * dt, "x0" => window_shifts(st, k) * h[1], "born_alive" => length(b.w),
                                                "born_x" => round.(b.x[keep]; digits = 2), "born_y" => round.(b.y[keep]; digits = 2),
                                                "born_z" => round.(b.z[keep]; digits = 2), "born_ux" => round.(b.ux[keep]; digits = 2),
                                                "born_w" => Float32.(b.w[keep])))
        end
    end
    frame_io === nothing || close(frame_io)
    seconds = time() - t0
    peak = (pool_attribute(:MEMPOOL_ATTR_USED_MEM_HIGH), pool_attribute(:MEMPOOL_ATTR_RESERVED_MEM_HIGH))
    parts = trapped_parts3(st)
    J = parts.born + parts.plasma
    h = Float64.(value_of.(cellsize(st.grid)))
    out = Dict{String,Any}(job)
    merge!(out, Dict{String,Any}("J" => Float64(value_of(J)), "J_born" => Float64(value_of(parts.born)), "J_plasma" => Float64(value_of(parts.plasma)),
                                 "seconds" => seconds, "seconds_init" => seconds_init, "seconds_steps" => t_steps, "state_mib" => state_mib,
                                 "gpu_peak_used_mib" => mib(peak[1]), "gpu_peak_reserved_mib" => mib(peak[2]), "gpu_pool_mib" => gpu_pool_mib(),
                                 "threads" => Threads.nthreads(), "backend" => BACKEND_NAME, "eltype" => string(eltype(st.fields.Ex)),
                                 "cells" => collect(ncells(st.grid)), "dx" => h[1], "dy" => h[2], "dz" => h[3], "dt" => dt, "nsteps" => prob.nsteps,
                                 "capacity" => capacity, "ionization_lost" => ionization_lost(st), "atoms" => length(ionization_state(st).w),
                                 "a0" => Float64(value_of(prob.lasers[1].a0)), "plasma_slots" => sum(s -> length(s.w), st.species),
                                 "bunch" => bunch3(st), "axis" => axis_readout3(st, prob.nsteps, ppl)))
    if grad
        out["gradient"] = Float64.(collect(ForwardDiff.partials(J)))
        out["gradient_born"] = Float64.(collect(ForwardDiff.partials(parts.born)))
        out["gradient_names"] = collect(GRAD_NAMES)
    else
        nx = ncells(st.grid)[1]
        stride = max(1, nx ÷ 896)
        out["line_x"] = round.(collect((1:stride:nx) .* h[1]); digits = 4)
        out["line_Ey"] = Float32.(axis_line3(st, st.fields.Ey)[1:stride:nx])
        out["line_Ex"] = Float32.(axis_line3(st, st.fields.Ex)[1:stride:nx])
        isempty(series) || (out["series"] = series)
        isempty(frames_meta) || (out["frames"] = frames_meta; out["frame_raster"] = [FX, FY])
    end
    return out
end

# --- campaigns -------------------------------------------------------------------------------------

"The jobs of a 3D campaign, the cheap ones first."
function campaign3_jobs(name)
    jobs = Dict{String,Any}[]
    if name == "smoke3"
        # a short, coarse box on the cloud CPU: does the pipeline run, what one particle step costs
        plain!(jobs, "smoke3-r8"; ppl = 8, rows = 64, ppc = [1, 1, 1], t_end = 60.0)
        plain!(jobs, "smoke3-r8-m4"; ppl = 8, rows = 64, ppc = [1, 1, 1], t_end = 60.0, solver = "m4")
        push!(jobs, Dict{String,Any}("id" => "smoke3-r8-grad", "ppl" => 8, "rows" => 64, "ppc" => [1, 1, 1], "t_end" => 60.0, "grad" => true))
        plain!(jobs, "smoke3-r8-frames"; ppl = 8, rows = 64, ppc = [1, 1, 1], t_end = 60.0, frames = 3)
    elseif name == "c16"
        # λ₀/16 on the CPU box: the particles per cell, the transverse box, the solver and the gather
        for ppc in ([1, 1, 1], [2, 1, 1], [2, 2, 2])
            plain!(jobs, jid("c16", "p$(tag(ppc))"); ppl = 16, ppc)
        end
        plain!(jobs, "c16-p2x2x2-m4"; ppl = 16, solver = "m4")
        plain!(jobs, "c16-p2x2x2-plain"; ppl = 16, gather = "plain")
        plain!(jobs, "c16-p2x2x2-gc"; ppl = 16, gather = "galerkin-compensated")
        plain!(jobs, "c16-p2x2x2-rows96"; ppl = 16, rows = 96)
        plain!(jobs, "c16-p2x2x2-frames"; ppl = 16, frames = 30)
    elseif name == "c24"
        plain!(jobs, "c24-p2x2x2"; ppl = 24)
        plain!(jobs, "c24-p2x2x2-m4"; ppl = 24, solver = "m4")
        plain!(jobs, "c24-p2x1x1"; ppl = 24, ppc = [2, 1, 1])
    elseif name == "c32"
        plain!(jobs, "c32-p2x2x2"; ppl = 32)
        plain!(jobs, "c32-p2x2x2-m4"; ppl = 32, solver = "m4")
        plain!(jobs, "c32-p2x1x1"; ppl = 32, ppc = [2, 1, 1])
    elseif name == "c48"
        plain!(jobs, "c48-p2x1x1"; ppl = 48, ppc = [2, 1, 1])
        plain!(jobs, "c48-p2x1x1-m4"; ppl = 48, ppc = [2, 1, 1], solver = "m4")
    elseif name == "g3"
        # the GPU campaign, in Float32: compiling jobs first (8 cells per λ₀, a short run of each variant), then the grids along x
        # with 2 × 2 × 2 and 2 × 1 × 1 particles per cell, the two solvers, the gathers, the transverse cell and box, the frames of
        # two runs for the page, and the gradient with duals where it fits on the card
        f = (; T = "f32")
        g(ppl, ppc) = "r$(ppl)-p$(tag(ppc))"
        for (sv, ga) in (("yee", "compensated"), ("m4", "compensated"), ("yee", "plain"), ("yee", "galerkin-compensated"), ("m4", "galerkin-compensated"))
            plain!(jobs, jid("g3", "warm", sv, ga); ppl = 8, rows = 64, ppc = [1, 1, 1], t_end = 20.0, solver = sv, gather = ga, series = false, f...)
        end
        push!(jobs, Dict{String,Any}("id" => "g3-warm-grad", "ppl" => 8, "rows" => 64, "ppc" => [1, 1, 1], "t_end" => 20.0, "grad" => true, "T" => "f32"))
        for (ppl, ppcs) in ((16, ([1, 1, 1], [2, 1, 1], [2, 2, 2], [3, 3, 3])), (24, ([2, 1, 1], [2, 2, 2])), (32, ([1, 1, 1], [2, 1, 1], [2, 2, 2])),
                            (48, ([2, 1, 1], [2, 2, 2])), (64, ([2, 1, 1], [2, 2, 2])), (96, ([2, 1, 1],)))
            for ppc in ppcs, sv in ("yee", "m4")
                (ppc == [1, 1, 1] || ppc == [3, 3, 3]) && sv == "m4" && continue
                plain!(jobs, jid("g3", g(ppl, ppc), sv); ppl, ppc, solver = sv, f...)
            end
        end
        for ppl in (16, 32, 48), ga in ("plain", "galerkin-compensated")
            plain!(jobs, jid("g3", g(ppl, [2, 1, 1]), "m4", ga); ppl, ppc = [2, 1, 1], solver = "m4", gather = ga, f...)
        end
        plain!(jobs, "g3-r16-p2x2x2-yee-rows96"; ppl = 16, rows = 96, f...)
        plain!(jobs, "g3-r32-p2x1x1-m4-rows96"; ppl = 32, rows = 96, ppc = [2, 1, 1], solver = "m4", f...)
        for ppl in (16, 32)
            plain!(jobs, jid("g3", g(ppl, [2, 1, 1]), "m4", "nt2"); ppl, nt = 2, ppc = [2, 1, 1], solver = "m4", f...)
        end
        plain!(jobs, "g3-r16-p2x2x2-yee-frames"; ppl = 16, frames = 30, f...)
        plain!(jobs, "g3-r32-p2x1x1-m4-frames"; ppl = 32, ppc = [2, 1, 1], solver = "m4", frames = 30, f...)
        for (ppl, ppc, sv) in ((16, [2, 1, 1], "yee"), (16, [2, 2, 2], "yee"), (24, [2, 1, 1], "m4"), (32, [2, 1, 1], "m4"), (32, [2, 1, 1], "yee"),
                               (32, [2, 2, 2], "m4"), (48, [2, 1, 1], "m4"))
            push!(jobs, Dict{String,Any}("id" => jid("g3", g(ppl, ppc), sv, "grad"), "ppl" => ppl, "ppc" => ppc, "solver" => sv, "grad" => true, "T" => "f32"))
        end
    elseif name == "t16"
        # the transverse cell at λ₀/16: Δy = Δz = π/8
        plain!(jobs, "t16-nt2-p2x1x1"; ppl = 16, nt = 2, ppc = [2, 1, 1])
    else
        error("unknown 3D campaign $name")
    end
    cost(j) = (q = merge(JOB3_DEFAULTS, j); q["ppl"] * (q["ppl"] * q["nt"]^2 + 8) * (q["rows"] / 128)^2 * prod(q["ppc"]) * (q["grad"] == true ? 4 : 1) * q["t_end"])
    return sort(jobs; by = cost)
end

function run_campaign3(name, dir)
    jobs = campaign3_jobs(name)
    mkpath(dir)
    if !isfile(joinpath(dir, "manifest.toml"))
        gpu = BACKEND_NAME == "cuda" ? CUDACore.name(CUDACore.device()) : nothing
        extra = gpu === nothing ? Dict{String,Any}() : Dict{String,Any}("gpu" => gpu, "cudacore" => string(pkgversion(CUDACore)))
        run_dir("gridppc3d-$name", @__FILE__; prefix = "resolution", work_package = "none (study for a report)", dir,
                backend = BACKEND_NAME == "cuda" ? "KernelAbstractions CUDA (CUDACore)" : "KernelAbstractions CPU", extra,
                extra_inputs = [joinpath(@__DIR__, "gridppc.jl"), joinpath(@__DIR__, "..", "optim", "trapped2d_case.jl"),
                                joinpath(@__DIR__, "..", "optim", "born2d_case.jl"), joinpath(@__DIR__, "..", "ionization", "atomic.jl")],
                params = Dict{String,Any}("campaign" => name, "jobs" => length(jobs), "defaults" => JOB3_DEFAULTS))
        open(io -> JSON.print(io, jobs, 1), joinpath(dir, "jobs.json"), "w")
    end
    todo = filter(j -> !isfile(joinpath(dir, j["id"] * ".json")), jobs)
    println("3D campaign $name on $BACKEND_NAME with $(Threads.nthreads()) threads: $(length(todo)) of $(length(jobs)) jobs to run, into $dir"); flush(stdout)
    t0 = time()
    for (i, j) in enumerate(todo)
        while isfile(joinpath(dir, "PAUSE")) || isfile(joinpath(dirname(dir), "PAUSE"))
            sleep(10)
        end
        try
            out = run_job3(j, dir)
            open(io -> JSON.print(io, out), joinpath(dir, j["id"] * ".json"), "w")
            println("done $(j["id"]) ($i of $(length(todo))): J = $(out["J"]), lost $(out["ionization_lost"]), $(round(out["seconds"]; digits = 1)) s, ",
                    "state $(out["state_mib"]) MiB, device $(out["gpu_peak_used_mib"]) MiB, total $(round(time() - t0; digits = 0)) s")
        catch err
            open(io -> showerror(io, err, catch_backtrace()), joinpath(dir, j["id"] * ".err"), "w")
            println("FAILED $(j["id"]): $(sprint(showerror, err)[1:min(end, 300)])")
        end
        flush(stdout)
        GC.gc(true)
        BACKEND_NAME == "cuda" && CUDACore.reclaim()
    end
    println("3D campaign $name finished in $(round(time() - t0; digits = 0)) s")
    return dir
end

if abspath(PROGRAM_FILE) == @__FILE__
    if length(ARGS) >= 3 && ARGS[1] == "campaign"
        foreach(c -> run_campaign3(c, abspath(joinpath(ARGS[3], c))), split(ARGS[2], ","))
    elseif length(ARGS) >= 3 && ARGS[1] == "job"
        job = JSON.parse(ARGS[3])
        mkpath(ARGS[2])
        out = run_job3(job, abspath(ARGS[2]))
        open(io -> JSON.print(io, out), joinpath(ARGS[2], job["id"] * ".json"), "w")
        println(job["id"], ": J = ", out["J"], " in ", round(out["seconds"]; digits = 1), " s")
    elseif length(ARGS) >= 2 && ARGS[1] == "list"
        js = campaign3_jobs(ARGS[2])
        println(length(js), " jobs")
        foreach(j -> println(j["id"]), js)
    end
end
