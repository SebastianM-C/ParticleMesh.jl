# WP-042: the periodic 3D3V electromagnetic step with the charge-conserving current, T-950 to
# T-957 (T-958 is their rerun on CUDA, in gpu.jl). The plasma of T-950, T-951, T-952 and T-955
# (`plasma_step_3d`, em3d_cases.jl) has two mobile species of opposite charge whose particles
# cross cells and the seams of every axis within a few steps.

"Start em3d_worker.jl at `nt` threads, its output going to a file; once per suite."
function start_em3d_worker(nt)
    get!(EM3D_WORKERS, nt) do
        worker = joinpath(@__DIR__, "em3d_worker.jl")
        out = tempname()
        cmd = `$(Base.julia_cmd()) --startup-file=no --threads=$nt --project=$(dirname(Base.active_project())) $worker`
        (run(pipeline(cmd; stdout = out, stderr = stderr); wait = false), out)
    end
end
const EM3D_WORKERS = Dict{Int,Tuple{Base.Process,String}}()

"The `EM3DJ` lines of the worker at `nt` threads, after it finished."
function em3d_worker_lines(nt)
    proc, out = start_em3d_worker(nt)
    wait(proc)
    success(proc) || error("em3d_worker.jl with $nt threads failed: $proc")
    return filter(startswith("EM3DJ "), split(read(out, String), '\n'))
end

foreach(start_em3d_worker, (1, 4))     # the workers run beside this file's in-process runs

"""
    gauss_residual_3d_step(state) -> (; res, scale, divB, bscale)

The discrete Gauss law of the 3D layout, `res = max |Σ_d D⁻_d E_d − (ρ − mean ρ)|` over the
interior nodes, ρ the sum of each species' charge deposited now in the state's mode at the
primal nodes (written out here rather than the module's `rho_stagger`, so that a wrong stagger
there cannot cancel; the step sorts the species by their current cells where the mode needs it),
and `scale = (n₁ + n₂ + n₃)·Σ_s max|ρ_s|`, the species' charges taken apart so that the scale
does not shrink where they cancel (as `gauss_residual`'s `species_scale` at N = 1); and
`divB = max |Σ_d D⁺_d B_d|` at the cell centres with `bscale = (Σ_d 2/Δ_d)·max_d |B_d|`.
`absmax` takes the partials of a dual too.
"""
function gauss_residual_3d_step(st)
    (; grid, numerics) = st
    T = eltype(st.ρ)
    g, n, h = guards(grid), ncells(grid), cellsize(grid)
    I = CartesianIndices(ntuple(d -> (g + 1):(g + n[d]), 3))
    e = ntuple(d -> CartesianIndex(ntuple(a -> Int(a == d), 3)), 3)
    ρ = zeros(T, size(st.ρ))
    species_scale = 0.0
    for (s, ws) in zip(st.species, st.ws)
        ρs = field(get_backend(st.ρ), T, grid)
        deposit!(numerics.mode, ρs, s, numerics.shape, (Node(), Node(), Node()), grid, ws)
        fold_guards!(ρs, grid)
        a = Array(ρs)
        species_scale += Float64(maximum(absmax, a[I]))
        ρ .+= a
    end
    E = map(Array, (st.fields.Ex, st.fields.Ey, st.fields.Ez))
    B = map(Array, (st.fields.Bx, st.fields.By, st.fields.Bz))
    ρ̄ = sum(ρ[I]) / prod(n)
    res = maximum(absmax(sum(d -> (E[d][j] - E[d][j - e[d]]) / h[d], 1:3) - (ρ[j] - ρ̄)) for j in I)
    divB = maximum(absmax(sum(d -> (B[d][j + e[d]] - B[d][j]) / h[d], 1:3)) for j in I)
    bmax = maximum(b -> maximum(absmax, b[I]), B)
    hv = map(value_of, h)
    return (; res, scale = sum(n) * species_scale, divB, bscale = sum(d -> 2 / hv[d], 1:3) * bmax)
end

# --- T-950 ----------------------------------------------------------------------------------

# Gauss's law: after initialize!, D⁻·E⁰ = ρ⁰ − mean ρ to roundoff (T-881). Each step changes E by
# Δt(∇×B − J); D⁻·(∇×B) vanishes identically (the discrete curls are each other's transpose,
# T-322), and the 3D deposit_flux! makes (ρⁿ⁺¹ − ρⁿ)/Δt + D⁻·J = 0 to roundoff (T-940), so
# D⁻·Eⁿ⁺¹ = ρⁿ⁺¹ − mean ρ after every step, every component of J now being Esirkepov's. The
# residual is not re-solved: each step adds its roundoff, hence T-360's scale with the species'
# charges taken apart, (Σ n_d)·Σ_s max|ρ_s|·√(k + 1). ∇·B starts at zero and stays at roundoff of the curl updates, scaled by
# (Σ 2/Δ_d)·max|B|·√k.
function body_T950(be)
    C, CB = emC("T-950"), emC("T-950", "C_divB")
    for T in TYPES, mode in (Reference(), Fast()), shape in SHAPES
        prob = plasma_step_3d(T; amplitude = seeded(T, 1 / 10))
        st = state_3d(prob; mode, be, shape)
        p = params(prob)
        r = gauss_residual_3d_step(st)
        worst = r.res / (base_eps(T) * r.scale)
        worstB = 0.0
        for k in 1:prob.nsteps
            step!(st, p, k)
            r = gauss_residual_3d_step(st)
            worst = max(worst, r.res / (base_eps(T) * r.scale * sqrt(k + 1)))
            r.bscale > 0 && (worstB = max(worstB, r.divB / (base_eps(T) * r.bscale * sqrt(k))))
        end
        # a current flows along every axis and a magnetic field grows (over 20 steps the drifts
        # carry the particles across cells and the seams)
        @test all(c -> maximum(absmax, Array(getfield(st.fields, c))) > 0, (:Jx, :Jy, :Jz, :Bx, :By, :Bz))
        label = "$(nameof(typeof(mode))) order $(order(shape))"
        record!("T-950 $label", T, worst, be)
        record!("T-950 divB $label", T, worstB, be)
        @test worst <= C
        @test worstB <= CB
    end
    # over many steps: 400 at order 2 (t = 145/ω₀, about 24 periods of the two species' plasma
    # frequency √(1 + 1/8)), on the same scale, checked every tenth step
    for T in (Float32, Float64), mode in (Reference(), Fast())
        prob = plasma_step_3d(T; amplitude = seeded(T, 1 / 10), nsteps = 400)
        st = state_3d(prob; mode, be)
        p = params(prob)
        worst, worstB = 0.0, 0.0
        for k in 1:prob.nsteps
            step!(st, p, k)
            k % 10 == 0 || continue
            r = gauss_residual_3d_step(st)
            worst = max(worst, r.res / (base_eps(T) * r.scale * sqrt(k + 1)))
            r.bscale > 0 && (worstB = max(worstB, r.divB / (base_eps(T) * r.bscale * sqrt(k))))
        end
        record!("T-950 400 steps $(nameof(typeof(mode)))", T, worst, be)
        record!("T-950 divB 400 steps $(nameof(typeof(mode)))", T, worstB, be)
        @test worst <= C
        @test worstB <= CB
    end
    # a quasi-neutral plasma, where the net charge is a thousandth of each species' (the scale
    # takes the species apart)
    for T in (Float32, Float64), mode in (Reference(), Fast())
        prob = neutral_plasma_3d(T)
        st = state_3d(prob; mode, be)
        p = params(prob)
        worst, worstB = 0.0, 0.0
        for k in 1:prob.nsteps
            step!(st, p, k)
            r = gauss_residual_3d_step(st)
            worst = max(worst, r.res / (base_eps(T) * r.scale * sqrt(k + 1)))
            r.bscale > 0 && (worstB = max(worstB, r.divB / (base_eps(T) * r.bscale * sqrt(k))))
        end
        record!("T-950 neutral $(nameof(typeof(mode)))", T, worst, be)
        record!("T-950 divB neutral $(nameof(typeof(mode)))", T, worstB, be)
        @test worst <= C
        @test worstB <= CB
    end
end

@testset "T-950 Gauss's law and ∇·B to roundoff after every 3D step with moving particles, both modes, orders 1 and 2, over 400 steps and in a quasi-neutral plasma [A2, A7, A8, WP-042]" begin
    body_T950(CPU())
end

# --- T-951 ----------------------------------------------------------------------------------

# A4 rule 3: the two deposition modes agree. Their currents differ by the association order of
# the atomic sums only, so after k steps the fields differ by roundoff that the dynamics carry
# along, measured relative to k·√(particles per cell per species)·(the largest Reference value
# of the kind: E and B together, J together, as T-953), over 20 steps; and the kinetic energy
# likewise. Per component, as T-362 measures it, a component that is still small (Bx after one
# step, of order Δt² times the current's curl) would divide the current's roundoff by itself.
function body_T951(be)
    C = emC("T-951")
    for T in (Float32, Float64)
        prob = plasma_step_3d(T)
        sr = state_3d(prob; mode = Reference(), be)
        sf = state_3d(prob; mode = Fast(), be)
        worst = 0.0
        bitwise = true
        scale = sqrt(prod(prob.species[1].ppc))
        for k in 1:prob.nsteps
            step!(sr, params(prob), k)
            step!(sf, params(prob), k)
            for kind in ((:Ex, :Ey, :Ez, :Bx, :By, :Bz), (:Jx, :Jy, :Jz))
                A = map(c -> Array(getfield(sr.fields, c)), kind)
                F = map(c -> Array(getfield(sf.fields, c)), kind)
                m = maximum(a -> maximum(abs, a), A)
                for (a, b) in zip(A, F)
                    worst = max(worst, maximum(abs, a .- b) / (eps(T) * m * k * scale))
                    bitwise &= a == b
                end
            end
        end
        hr, hf = histories(sr), histories(sf)
        worst = max(worst, maximum(abs, hr.K .- hf.K) / (eps(T) * maximum(abs, hr.K) * prob.nsteps * scale))
        @info "T-951 Reference against Fast" backend = backend_label(be) T worst bitwise
        record!("T-951", T, worst, be)
        @test worst <= C
    end
end

@testset "T-951 the Reference and Fast 3D steps agree to roundoff [A4, A7, WP-042]" begin
    body_T951(CPU())
end

# --- T-952 ----------------------------------------------------------------------------------

@testset "T-952 the 3D step with current is bitwise across reruns, launch shapes and 1 and 4 threads in Reference mode [A4, WP-042]" begin
    ref = em3d_step_digests()
    @test length(ref) == 3 * length(TYPES)
    @test allunique(last.(ref))
    @test em3d_step_digests() == ref
    for wgs in (32, 256)
        @test em3d_step_digests(; workgroupsize = wgs) == ref
    end
    for nt in (1, 4)
        fields = split.(em3d_worker_lines(nt))
        got = Dict((String(f[3]), String(f[4])) => parse(UInt64, f[5]) for f in fields)
        @test all(f -> parse(Int, f[2]) == nt, fields)
        @test length(got) == length(ref)
        for (label, T, d) in ref
            @test get(got, (label, type_label(T)), nothing) == d
        end
    end
end

# --- T-953 ----------------------------------------------------------------------------------

# The 3D step on a problem that does not vary along one axis is the 2D step: two cold beams with
# opposite drifts in all three components, one of them with a density perturbation of mode (1, 1)
# in the plane, so the net charge is perturbed and the initial field and the half kick back of
# initialize! are not zero (`beams_plane_2d`, `beams_3d`). The 2D run's current along its
# collapsed axis is the quantity q·w·u_z/γ; the 3D run's along the invariant axis is Esirkepov's,
# whose sum over a uniform line of particles that move alike is that quantity, so the two agree
# to roundoff. The plane maps onto the 3D axes by a cyclic permutation (`inv3_axes`), a rotation,
# so the components map with no sign. The quiet start fills each invariant line with a uniform
# lattice, so every deposit is invariant by the partition of unity of the shape; the 3D and 2D
# initial fields from the spectral solves agree to roundoff. Compared after initialize!, then
# after every step, as T-361: relative to the largest 2D value of the kind (fields E, B; currents
# J), scaled by k·(n₁ + n₂)·√ppc, k ≥ 1.

"The 3D component of each 2D one, by the cyclic map of [`inv3_axes`](@ref)."
function invariant_map_3d(axis)
    σ = inv3_axes(axis)
    names(F) = map(c -> Symbol(F, (:x, :y, :z)[c]), σ)
    E, B, J = names(:E), names(:B), names(:J)
    return (Ex = E[1], Ey = E[2], Ez = E[3], Bx = B[1], By = B[2], Bz = B[3], Jx = J[1], Jy = J[2], Jz = J[3])
end

"""
Largest |3D − 2D| over every component and every line along the invariant axis, each relative to
the largest 2D value of its kind (the fields E, B together; the currents J together).
"""
function invariant_error_3d(s2, s3, axis)
    σ = inv3_axes(axis)
    g2, g3, n = guards(s2.grid), guards(s3.grid), ncells(s2.grid)
    m = invariant_map_3d(axis)
    plane(c) = Array(getfield(s2.fields, c))[(g2 + 1):(g2 + n[1]), (g2 + 1):(g2 + n[2])]
    kind(c) = c in (:Jx, :Jy, :Jz) ? :J : :F
    scale = Dict(k => maximum(c -> Float64(maximum(absmax, plane(c))), filter(c -> kind(c) == k, collect(keys(m)))) for k in (:F, :J))
    err = 0.0
    for (c, c3) in pairs(m)
        a2 = plane(c)
        A3 = Array(getfield(s3.fields, c3))
        for r in 1:ncells(s3.grid)[σ[3]]
            # the 3D values on the line r of the invariant axis, as a plane of the 2D axes
            idx = Vector{Any}(undef, 3)
            idx[σ[1]] = (g3 + 1):(g3 + n[1])
            idx[σ[2]] = (g3 + 1):(g3 + n[2])
            idx[σ[3]] = g3 + r
            sl = A3[idx...]
            a3 = σ[1] < σ[2] ? sl : permutedims(sl)
            d = Float64(maximum(absmax, a3 .- a2))
            err = max(err, iszero(d) ? 0.0 : d / scale[kind(c)])     # J is zero after initialize!
        end
    end
    return err
end

function body_T953(be)
    C = emC("T-953")
    for T in TYPES, mode in (Reference(), Fast()), axis in (3, 1, 2)
        amplitude = seeded(T, 1 / 20)
        p2, p3 = beams_plane_2d(T; amplitude), beams_3d(T, axis; amplitude)
        s2 = EMState(p2, EMNumerics(; mode); backend = be)
        s3 = EMState(p3, EMNumerics(; mode); backend = be)
        initialize!(s2, p2)
        initialize!(s3, p3)
        @test maximum(absmax, Array(s2.fields.Ex)) > 0 && maximum(absmax, Array(s2.fields.Ey)) > 0   # the initial field is not trivial
        scale = sum(ncells(s2.grid)) * sqrt(prod(p2.species[1].ppc))
        worst = invariant_error_3d(s2, s3, axis) / (base_eps(T) * scale)
        for k in 1:p2.nsteps
            step!(s2, params(p2), k)
            step!(s3, params(p3), k)
            worst = max(worst, invariant_error_3d(s2, s3, axis) / (base_eps(T) * k * scale))
        end
        # the run is not trivial: every component it compares carries a field
        @test all(c -> maximum(absmax, Array(getfield(s2.fields, c))) > 0, keys(invariant_map_3d(axis)))
        record!("T-953 axis $axis $(nameof(typeof(mode)))", T, worst, be)
        @test worst <= C
    end
end

@testset "T-953 a 3D run invariant along one axis is the 2D run, along each of the three axes [A2, A7, A8, WP-042]" begin
    body_T953(CPU())
end

# --- T-954 ----------------------------------------------------------------------------------

# Landau damping of a Langmuir wave whose wave vector has a component along every axis: mode
# (1, 1, 1) of a cube of side 2π√3/|k|, |k| = 10 ω₀/c and v_th = 0.05, so |k|λ_D = 0.5 as in the
# 2D case (`landau-oblique-2d`) and T-364. For an isotropic load the rate depends on |k| alone, so
# the reference is the relativistic linear theory of the loaded distribution at |k|
# (`landau_rate_relativistic`, Maxwell–Jüttner), as T-364's. The run is CI-sized: LANDAU3_CELLS
# cells per axis and LANDAU3_PPC electrons per cell, a 10 % perturbation, t = 18; the fit is the
# case's rule (maxima of |k̂·Ê|² after t = 3 above ten times the noise floor). As T-364, one seed
# is an order check of the rate (one seed's std is 0.10 of theory at this size) and a closer check
# of the frequency and of the longitudinal polarization, and the ensemble of the four seeds of
# LANDAU3_SEEDS checks the rate within about 10 to 18 % around the test size's offset from theory
# (+0.18 over sixteen seeds; it falls with resolution, tolerances.toml). At 16³ cells and 4³
# electrons per cell (as many electrons, Δx/λ_D = 1.36) one seed in four left only two maxima
# above the floor; 24³ and 3³ (Δx/λ_D = 0.91, as T-364's 0.88) give three to five for each of
# sixteen seeds.
const LANDAU3_CELLS = 24
const LANDAU3_PPC = (3, 3, 3)
const LANDAU3_SEEDS = (20261031, 20262031, 20263031, 20264031)
const LANDAU3 = Dict{String,Any}()

function landau_3d(::Type{T}; cells = LANDAU3_CELLS, ppc = LANDAU3_PPC, amplitude = 1 / 10, t_end = 18, seed = first(LANDAU3_SEEDS),
                   vth = 1 / 20, k = 10) where {T}
    L = T(2π * sqrt(3) / k)
    box = (L, L, L)
    n = (cells, cells, cells)
    dt = T(95 // 100) * inv(sqrt(3 * (cells / L)^2))
    nsteps = round(Int, t_end / dt)
    e = SpeciesDescription{T}(:electron, -one(T), one(T), one(T), T(amplitude), (1, 1, 1), ppc, zero3(T), (T(vth), T(vth), T(vth)), 1)
    return EMProblem("landau-oblique-3d", box, n, dt, nsteps, (e,), seed, (1, 1, 1))
end

function body_T954(be)
    prob = landau_3d(Float64)
    st = EMState(prob, EMNumerics(; shape = Quadratic); backend = be)
    initialize!(st, prob)
    seconds = @elapsed run!(st, params(prob), prob.nsteps)
    h = histories(st)
    ts = history_times(prob)
    sp = only(prob.species)
    @test sp.momentum === :maxwell_juttner
    kmag = 10.0
    th = landau_rate_relativistic(kmag / sqrt(1.0), (1 / 20)^2; load = :maxwell_juttner)
    U = abs2.(complex.(h.El_re, h.El_im))
    f = damping_from_peaks(ts, U; fit_start = 3.0, floor_factor = 10.0)
    rel, relω = (f.gamma - th.gamma) / th.gamma, (f.omega - th.omega) / th.omega
    # the wave is longitudinal: the part of the mode's E perpendicular to k̂ stays small
    k̂ = (1, 1, 1) ./ sqrt(3)
    Ê = map(c -> complex.(getfield(h, Symbol(c, :_re)), getfield(h, Symbol(c, :_im))), (:Ex, :Ey, :Ez))
    El = sum(d -> k̂[d] .* Ê[d], 1:3)
    Et = sqrt.(sum(d -> abs2.(Ê[d] .- k̂[d] .* El), 1:3))
    transverse = maximum(Et) / maximum(sqrt.(U))
    @info "T-954 oblique Landau in 3D" backend = backend_label(be) γ_theory = th.gamma γ = f.gamma rel ω_theory = th.omega ω = f.omega relω peaks = length(f.t_fit) transverse seconds steps = prob.nsteps particles = alive_count(only(st.species))
    LANDAU3[physics_key("landau-oblique-3d", be)] = (; st, prob, h, ts, f)
    d = get!(MEASURED, "T-954 relative to theory", Dict{String,Float64}())
    d["γ $(backend_label(be))"], d["ω $(backend_label(be))"], d["transverse $(backend_label(be))"] = rel, relω, transverse
    @test length(f.t_fit) >= 3
    @test f.gamma > 0
    @test abs(rel) <= emC("T-954", "rate")
    @test abs(relω) <= emC("T-954", "frequency")
    @test transverse <= emC("T-954", "transverse")
    # As T-364's: the rate of the ensemble over four seeds (the run above and three more), fitted
    # on |⟨k̂·Ê⟩_seeds|², the seed mean of the complex coefficient, whose seed noise is a few times
    # smaller than one seed's, bounded around the test size's offset from theory
    # (tolerances.toml: |rel − rate_ensemble_offset| ≤ rate_ensemble).
    @test prob.seed == first(LANDAU3_SEEDS)
    coeffs = [complex.(h.El_re, h.El_im)]
    for seed in LANDAU3_SEEDS[2:end]
        ps = landau_3d(Float64; seed)
        ss = EMState(ps, EMNumerics(; shape = Quadratic); backend = be)
        initialize!(ss, ps)
        run!(ss, params(ps), ps.nsteps)
        hs = histories(ss)
        push!(coeffs, complex.(hs.El_re, hs.El_im))
    end
    fe = damping_from_peaks(ts, abs2.(sum(coeffs) ./ length(coeffs)); fit_start = 3.0, floor_factor = 10.0)
    rele = (fe.gamma - th.gamma) / th.gamma
    @info "T-954 four-seed ensemble" backend = backend_label(be) seeds = LANDAU3_SEEDS γ = fe.gamma rel = rele peaks = length(fe.t_fit)
    d["ensemble γ $(backend_label(be))"] = rele
    @test length(fe.t_fit) >= 3
    @test abs(rele - emC("T-954", "rate_ensemble_offset")) <= emC("T-954", "rate_ensemble")
end

@testset "T-954 Landau damping of a Langmuir wave oblique to all three axes against the relativistic linear theory of its load, one seed and a four-seed ensemble [A5, A7, WP-042]" begin
    body_T954(CPU())
end

# --- T-955 ----------------------------------------------------------------------------------

# Dual{Float32,3} through the full 3D step (T-950 checks Gauss's law on its partials): the primal
# against the Float32 run, and the partials ∂/∂(amplitude) of the fields finite and nonzero, the
# unseeded directions zero.
@testset "T-955 Dual through the 3D step with current [A3, A8, WP-042]" begin
    C = emC("T-955")
    a = mkdual(1 / 10, 1, 0, 0)
    pd, pf = plasma_step_3d(D3; amplitude = a, nsteps = 10), plasma_step_3d(Float32; nsteps = 10)
    sd, sf = state_3d(pd), state_3d(pf)
    run!(sd, params(pd), pd.nsteps)
    run!(sf, params(pf), pf.nsteps)
    worst = 0.0
    for c in (:Ex, :Ey, :Ez, :Bx, :By, :Bz)
        A, B = Array(getfield(sd.fields, c)), Array(getfield(sf.fields, c))
        worst = max(worst, maximum(abs, value_of.(A) .- B) / (eps(Float32) * maximum(abs, B) * pd.nsteps))
        @test all(x -> all(isfinite, partials(x)), A)
        @test any(x -> !iszero(partials(x)[1]), A)
        @test all(x -> iszero(partials(x)[2]) && iszero(partials(x)[3]), A)     # unseeded directions stay zero
    end
    record!("T-955", D3, worst)
    @test worst <= C
end

# --- T-956 ----------------------------------------------------------------------------------

function step_bytes_3d(mode; ppc, cells)
    prob = plasma_step_3d(Float64; ppc, cells, nsteps = 4)
    st = state_3d(prob; mode)
    p = params(prob)
    step!(st, p, 1)
    step!(st, p, 2)
    bytes = @allocated step!(st, p, 3)
    return bytes, count(s -> ParticleMeshElectromagnetic.sorts(st, s), st.species)   # and the step's sorts
end

@testset "T-956 allocation per 3D step with current does not grow with particle count or grid size [A7, WP-042]" begin
    for mode in (Reference(), Fast())
        small, sorts = step_bytes_3d(mode; ppc = (1, 1, 2), cells = (8, 7, 6))
        more_particles, _ = step_bytes_3d(mode; ppc = (3, 3, 2), cells = (8, 7, 6))
        larger_grid, _ = step_bytes_3d(mode; ppc = (1, 1, 2), cells = (16, 14, 12))
        @info "T-956 bytes per 3D step with current" mode small more_particles larger_grid sorts threads = Threads.nthreads()
        # at more than one thread the Reference sort allocates per particle (#82, as T-220 and T-367); CI runs one thread
        # each sort may add Base's radix count buffer, bounded whatever the size (`SORT_COUNTS_BYTES`, as T-367)
        bound = small + sorts * SORT_COUNTS_BYTES
        if Threads.nthreads() == 1 || mode isa Fast
            @test_alloc more_particles <= bound
            @test_alloc larger_grid <= bound
        end
    end
end

# --- T-957 ----------------------------------------------------------------------------------

@testset "T-957 input checks of the 3D problem and state [A7, A10, WP-042]" begin
    T = Float64
    prob = plasma_step_3d(T)
    d = prob.species[1]
    sp(mode, ppc) = (SpeciesDescription{T}(:e, -one(T), one(T), one(T), zero(T), mode, ppc, zero3(T), zero3(T), 1),)
    L, n = prob.length, prob.cells
    @test EMProblem("ok", L, n, prob.dt, 2, sp((1, 0, 0), (1, 1, 1)), 1, (1, 0, 0)) isa EMProblem{3,T}
    @test ndims(prob) == 3 && eltype(prob) === T
    @test params(prob).species[1] isa SpeciesParams{T,3}
    @test params(prob).species[1].wavevector == ntuple(a -> 2 * T(π) * d.mode[a] / L[a], 3)
    @test_throws ArgumentError EMProblem("box", L[1:2], n, prob.dt, 2, sp((1, 0, 0), (1, 1, 1)), 1, (1, 0, 0))            # two box lengths
    @test_throws ArgumentError EMProblem("kmode", L, n, prob.dt, 2, sp((1, 0, 0), (1, 1, 1)), 1, (1, 0))                  # two mode integers
    @test_throws ArgumentError EMProblem("zero", L, n, prob.dt, 2, sp((1, 0, 0), (1, 1, 1)), 1, (0, 0, 0))                # the zero mode
    @test_throws ArgumentError EMProblem("2d", L, n, prob.dt, 2, sp((1, 0), (1, 1)), 1, (1, 0, 0))                        # a 2D species
    @test_throws ArgumentError EMProblem("ppc0", L, n, prob.dt, 2, sp((1, 0, 0), (1, 0, 1)), 1, (1, 0, 0))                # no particle on an axis
    @test_throws ArgumentError EMProblem("four", (L..., L[1]), (n..., 6), prob.dt, 2, sp((1, 0, 0, 0), (1, 1, 1, 1)), 1, (1, 0, 0, 0))
    for bd in ((:periodic, :periodic, :open), (:periodic, :open, :periodic), (:window, :periodic, :periodic))
        @test_throws ArgumentError EMProblem("bd", L, n, prob.dt, 2, sp((1, 0, 0), (1, 1, 1)), 1, (1, 0, 0); boundaries = bd)
    end
    @test EMProblem("bd", L, n, prob.dt, 2, sp((1, 0, 0), (1, 1, 1)), 1, (1, 0, 0); boundaries = (:periodic, :periodic, :periodic)) isa EMProblem{3}
    # the state: the numerics a 3D state does not take, and a step above the Courant limit
    @test_throws ArgumentError EMState(prob, EMNumerics(; filter = BinomialFilter(1)))
    @test_throws ArgumentError EMState(prob, EMNumerics(; resort = true))
    @test_throws ArgumentError EMState(prob, EMNumerics(; gather = CompensatedGather()))
    @test_throws ArgumentError EMState(prob, EMNumerics(; absorber = PML(; cells = 4)))
    @test_throws ArgumentError EMState(prob, EMNumerics(; back_absorber = PML(; cells = 4)))
    @test_throws ArgumentError EMState(prob, EMNumerics(; current = nothing))
    fast = EMProblem("fast", L, n, 2 * prob.dt, 2, prob.species, 1, prob.kmode)
    @test_throws ArgumentError EMState(fast, EMNumerics())
    # a state of NoCurrent runs (prescribed fields): its J stays zero
    st = EMState(prob, EMNumerics(; current = NoCurrent()))
    @test st isa EMState{3,T} && st.ws[1].flux === nothing
    initialize!(st, prob)
    step!(st, params(prob), 1)
    @test all(c -> all(iszero, getfield(st.fields, c)), (:Jx, :Jy, :Jz))
    # the state's recorder is that of the problem's box (not a cube, so a permuted box shows in
    # k̂), mode and step count, with the cadence and the groups it is given; its kinetic-energy
    # chunks cover every slot of every species
    st = state_3d(prob)
    ref = HistoryRecorder(CPU(), T, st.grid, L, prob.kmode, prob.nsteps)
    @test st.recorder.aux.khat === ref.aux.khat
    @test st.recorder.aux.khat != HistoryRecorder(CPU(), T, st.grid, reverse(L), prob.kmode, prob.nsteps).aux.khat
    @test all(d -> Array(st.recorder.aux.tables[d]) == Array(ref.aux.tables[d]), 1:3)
    @test st.recorder.kmode == prob.kmode && st.recorder.nsteps == prob.nsteps && st.recorder.every == 1
    @test keys(st.recorder.hist) == keys(ref.hist)
    every2 = EMState(prob, EMNumerics(); every = 2).recorder
    @test every2.every == 2 && all(h -> length(h) == cld(prob.nsteps, 2), values(every2.hist))
    @test isempty(EMState(prob, EMNumerics(); record = ()).recorder.hist)
    @test length(st.K) == sum(s -> cld(ParticleMeshCore.capacity(s), ParticleMeshElectromagnetic.KINETIC_CHUNK), st.species)
    # a state is built for its problem's configuration: another cell count, or another N, is refused
    other = plasma_step_3d(T; cells = (8, 7, 7))
    @test_throws ArgumentError initialize!(state_3d(prob), other)
    @test_throws ArgumentError initialize!(state_3d(prob), plasma_step_2d(T))
end
