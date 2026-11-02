# WP-023: the periodic 2D3V electromagnetic step with the charge-conserving current, T-360 to
# T-369. The plasma of T-360, T-362, T-363 and T-366 (`plasma_step_2d`, em2d_cases.jl) has two
# mobile species of opposite charge whose particles cross cells and both periodic seams.

using ParticleMeshElectromagnetic: RHO_STAGGER_2D, case_seed
using .Theory: weibel_cold_growth_2d

"""
    gauss_residual_2d(state) -> (; res, scale, divB, bscale)

The discrete Gauss law of the 2D layout, `res = max |D⁻ₓEx + D⁻_yEy − (ρ − mean ρ)|` over the
interior nodes with ρ deposited now in the state's mode (the species are sorted by their current
cells after `initialize!` and after every step), `scale = (n₁ + n₂)·max|ρ|`; and
`divB = max |D⁺ₓBx + D⁺_yBy|` at the cell centres with `bscale = (Σ_d 2/Δ_d)·max(|Bx|, |By|)`.
`absmax` takes the partials of a dual too.
"""
function gauss_residual_2d(st)
    (; grid, numerics) = st
    T = eltype(st.ρ)
    ρd = field(get_backend(st.ρ), T, grid)
    for (s, ws) in zip(st.species, st.ws)
        deposit!(numerics.mode, ρd, s, numerics.shape, RHO_STAGGER_2D, grid, ws)
    end
    fold_guards!(ρd, grid)
    g, n, h = guards(grid), ncells(grid), cellsize(grid)
    I1, I2 = (g + 1):(g + n[1]), (g + 1):(g + n[2])
    ρ, Ex, Ey, Bx, By = Array(ρd), Array(st.fields.Ex), Array(st.fields.Ey), Array(st.fields.Bx), Array(st.fields.By)
    ρ̄ = sum(ρ[I1, I2]) / prod(n)
    res = maximum(absmax((Ex[i, j] - Ex[i - 1, j]) / h[1] + (Ey[i, j] - Ey[i, j - 1]) / h[2] - (ρ[i, j] - ρ̄)) for i in I1, j in I2)
    divB = maximum(absmax((Bx[i + 1, j] - Bx[i, j]) / h[1] + (By[i, j + 1] - By[i, j]) / h[2]) for i in I1, j in I2)
    bmax = max(maximum(absmax, Bx[I1, I2]), maximum(absmax, By[I1, I2]))
    hv = map(value_of, h)
    return (; res, scale = sum(n) * maximum(absmax, ρ[I1, I2]), divB, bscale = (2 / hv[1] + 2 / hv[2]) * bmax)
end

# --- T-360 ----------------------------------------------------------------------------------

# Gauss's law: after initialize!, D⁻·E⁰ = ρ⁰ − mean ρ to roundoff (T-353). Each step changes E by
# Δt(∇×B − J); D⁻·(∇×B) vanishes identically (the discrete curls are each other's transpose,
# T-322), and deposit_flux! makes (ρⁿ⁺¹ − ρⁿ)/Δt + D⁻·J = 0 to roundoff (T-300), so D⁻·Eⁿ⁺¹ =
# ρⁿ⁺¹ − mean ρ after every step. The residual is not re-solved: each step adds its roundoff, a
# random walk, hence the scale (n₁ + n₂)·max|ρ|·√(k + 1). ∇·B starts at zero and stays at roundoff
# of the curl updates, scaled by (Σ 2/Δ_d)·max|B|·√k. Jz has no divergence at N = 2, so Gauss's
# law cannot see it; its total is exact instead: the shape weights sum to one, so after the fold
# Σ Jz over the nodes equals Σ q·w·u_z/γ over the particles at the momenta uⁿ⁺¹ᐟ² (T-309), which
# the step's `CurrentZ` must supply (the drifts make γ up to about 1.1, so a missing 1/γ shows).

"Σ q·w·u_z/γ over the alive particles of every species, on the host, and Σ of its absolute values."
function current_z_total(st)
    tot, mag = zero(eltype(st.ρ)), zero(eltype(st.ρ))
    for s in st.species
        w, st_, u = Array(s.w), Array(s.status), map(Array, s.u)
        for i in eachindex(w)
            st_[i] == ParticleMeshCore.ALIVE || continue
            jz = s.q * w[i] * u[3][i] / sqrt(1 + u[1][i]^2 + u[2][i]^2 + u[3][i]^2)
            tot += jz
            mag += abs(jz)
        end
    end
    return tot, mag
end
function body_T360(be)
    C, CB = emC("T-360"), emC("T-360", "C_divB")
    for T in TYPES, mode in (Reference(), Fast())
        prob = plasma_step_2d(T; amplitude = seeded(T, 1 / 10))
        st = state_2d(prob; mode, be, current = Esirkepov())
        p = params(prob)
        r = gauss_residual_2d(st)
        worst = r.res / (base_eps(T) * r.scale)
        worstB = worstZ = 0.0
        g, n = guards(st.grid), ncells(st.grid)
        for k in 1:prob.nsteps
            step!(st, p, k)
            r = gauss_residual_2d(st)
            worst = max(worst, r.res / (base_eps(T) * r.scale * sqrt(k + 1)))
            r.bscale > 0 && (worstB = max(worstB, r.divB / (base_eps(T) * r.bscale * sqrt(k))))
            tot, mag = current_z_total(st)
            Jz = sum(Array(st.fields.Jz)[(g + 1):(g + n[1]), (g + 1):(g + n[2])])
            worstZ = max(worstZ, Float64(absmax(Jz - tot)) / (base_eps(T) * Float64(value_of(mag))))
        end
        # a current flows and a magnetic field grows (over 40 steps the drifts carry the particles
        # across several cells and both seams)
        @test all(c -> maximum(absmax, Array(getfield(st.fields, c))) > 0, (:Jx, :Jy, :Jz, :Bx, :By, :Bz))
        record!("T-360 $(nameof(typeof(mode)))", T, worst, be)
        record!("T-360 divB $(nameof(typeof(mode)))", T, worstB, be)
        record!("T-360 sum Jz $(nameof(typeof(mode)))", T, worstZ, be)
        @test worst <= C
        @test worstB <= CB
        @test worstZ <= emC("T-360", "C_Jz")
    end
end

@testset "T-360 Gauss's law and ∇·B to roundoff after every 2D step with moving particles, and Σ Jz exact, both modes [A2, A7, A8, WP-023]" begin
    body_T360(CPU())
end

# --- T-361 ----------------------------------------------------------------------------------

# The 2D step on a problem that does not vary along one axis is the 1D step: two cold beams with
# opposite drifts in all three components (Jx through 2D Esirkepov along the resolved axis, the
# motion along the invariant axis through the other Esirkepov component, Jz), one of them with a
# density perturbation, so the net charge is perturbed and the initial field and the half kick
# back of initialize! are not zero. y-invariant: mode (1, 0), the 1D fields on every row.
# x-invariant: mode (0, 1) and the problem rotated by +90° about z, x̂ → ŷ, ŷ → −x̂, so the 1D
# (Ex, Ey, Ez, Bx, By, Bz) are the 2D (Ey, −Ex, Ez, By, −Bx, Bz) on every column. The quiet start
# fills each invariant column with a uniform lattice, so every deposit is invariant by the
# partition of unity of the shape; the 2D initial field from the spectral solve and the 1D prefix
# sum agree to roundoff (compared after initialize!, then after every step). The 1D prefix sum's
# error grows with the cell count n and the atomic sums' with the particles per cell, hence the
# scale k·n·√ppc·(largest 1D value of the kind), k ≥ 1 (measured by the WP-023 breakers at 16 to
# 64 cells and 8 to 64 particles per cell).

const INV_DRIFT = (1 // 5, 3 // 20, -1 // 10)

function beams_1d(::Type{T}; n = 16, ppc = 8, nsteps = 30, amplitude = one(T) / 20) where {T}
    L = 2 * T(π)
    dx = L / n
    v = T.(INV_DRIFT)
    sp = (SpeciesDescription{T}(:a, -one(T), one(T), one(T) / 2, T(amplitude), 1, ppc, v, zero3(T), 1),
          SpeciesDescription{T}(:b, -one(T), one(T), one(T) / 2, zero(T), 1, ppc, .-v, zero3(T), 2))
    return EMProblem("beams-1d", L, n, 4dx / 5, nsteps, sp, 1, 2 * T(π) / L)
end

# axis = 1: varies along x (y-invariant); axis = 2: varies along y. The invariant axis has 6 cells
# (the fewest the guards allow) of twice the 1D cell size, 2 particles per cell.
function beams_2d(::Type{T}, axis; n = 16, ppc = 8, nsteps = 30, amplitude = one(T) / 20) where {T}
    L = 2 * T(π)
    dx = L / n
    v = T.(INV_DRIFT)
    v2 = axis == 1 ? v : (-v[2], v[1], v[3])
    box = axis == 1 ? (L, 12dx) : (12dx, L)
    cells = axis == 1 ? (n, 6) : (6, n)
    p2 = axis == 1 ? (ppc, 2) : (2, ppc)
    m = axis == 1 ? (1, 0) : (0, 1)
    sp = (SpeciesDescription{T}(:a, -one(T), one(T), one(T) / 2, T(amplitude), m, p2, v2, zero3(T), 1),
          SpeciesDescription{T}(:b, -one(T), one(T), one(T) / 2, zero(T), m, p2, .-v2, zero3(T), 2))
    return EMProblem("beams-2d", box, cells, 4dx / 5, nsteps, sp, 1, m)
end

"The 2D fields of an invariant run mapped to the 1D components: (2D array, sign) per 1D key."
function invariant_map(f, axis)
    axis == 1 && return (Ex = (f.Ex, 1), Ey = (f.Ey, 1), Ez = (f.Ez, 1), By = (f.By, 1), Bz = (f.Bz, 1),
                         Jx = (f.Jx, 1), Jy = (f.Jy, 1), Jz = (f.Jz, 1))
    return (Ex = (f.Ey, 1), Ey = (f.Ex, -1), Ez = (f.Ez, 1), By = (f.Bx, -1), Bz = (f.Bz, 1),
            Jx = (f.Jy, 1), Jy = (f.Jx, -1), Jz = (f.Jz, 1))
end

"""
Largest |2D − 1D| over the mapped components and every line along the invariant axis, each
relative to the largest 1D value of its kind (the fields E, B together; the currents J together),
and the 2D-only in-plane B component along the resolved axis relative to the field scale.
"""
function invariant_error(s1, s2, axis)
    g1, g2, n = guards(s1.grid), guards(s2.grid), ncells(s1.grid)
    m = invariant_map(s2.fields, axis)
    line1(c) = Array(getfield(s1.fields, c))[(g1 + 1):(g1 + n)]
    kind(c) = c in (:Jx, :Jy, :Jz) ? :J : :F
    scale = Dict(k => maximum(c -> Float64(maximum(absmax, line1(c))), filter(c -> kind(c) == k, collect(keys(m)))) for k in (:F, :J))
    err = 0.0
    for (c, (a2, sgn)) in pairs(m)
        a1 = line1(c)
        A2 = Array(a2)
        lines = axis == 1 ? [A2[(g2 + 1):(g2 + n), g2 + r] for r in 1:ncells(s2.grid)[2]] :
                            [A2[g2 + r, (g2 + 1):(g2 + n)] for r in 1:ncells(s2.grid)[1]]
        for l in lines
            d = Float64(maximum(absmax, sgn .* l .- a1))
            err = max(err, iszero(d) ? 0.0 : d / scale[kind(c)])     # J is zero after initialize!
        end
    end
    extra = axis == 1 ? s2.fields.Bx : s2.fields.By
    return max(err, Float64(maximum(absmax, Array(extra))) / scale[:F])
end

function body_T361(be)
    C = emC("T-361")
    for T in TYPES, mode in (Reference(), Fast()), axis in (1, 2)
        amplitude = seeded(T, 1 / 20)
        p1, p2 = beams_1d(T; amplitude), beams_2d(T, axis; amplitude)
        s1 = EMState(p1, EMNumerics(; mode); backend = be)
        s2 = EMState(p2, EMNumerics(; mode); backend = be)
        initialize!(s1, p1)
        initialize!(s2, p2)
        @test maximum(absmax, Array(s1.fields.Ex)) > 0          # the initial field is not trivial
        scale = ncells(s1.grid) * sqrt(p1.species[1].ppc)
        worst = invariant_error(s1, s2, axis) / (base_eps(T) * scale)
        for k in 1:p1.nsteps
            step!(s1, params(p1), k)
            step!(s2, params(p2), k)
            worst = max(worst, invariant_error(s1, s2, axis) / (base_eps(T) * k * scale))
        end
        # the run is not trivial: every 1D component it compares carries a field
        f1 = s1.fields
        @test all(c -> maximum(absmax, Array(getfield(f1, c))) > 0, (:Ex, :Ey, :Ez, :By, :Bz, :Jx, :Jy, :Jz))
        record!("T-361 axis $axis $(nameof(typeof(mode)))", T, worst, be)
        @test worst <= C
    end
end

@testset "T-361 a 2D run invariant along one axis is the 1D run, along x and along y [A2, A7, A8, WP-023]" begin
    body_T361(CPU())
end

# --- T-362 ----------------------------------------------------------------------------------

# A4 rule 3: the two deposition modes agree. Their currents differ by the association order of
# the atomic sums only, a roundoff that grows as the square root of the contributions per node,
# so after k steps the fields differ by roundoff that the dynamics carry along; measured
# relative to k·√(particles per cell per species)·(the largest Reference value), for 20 steps (the
# plasma is unstable: over a few hundred steps the instability amplifies the difference
# exponentially, as it amplifies any roundoff; measured by the WP-023 breakers).
@testset "T-362 the Reference and Fast 2D steps agree to roundoff [A4, A7, WP-023]" begin
    C = emC("T-362")
    for T in (Float32, Float64)
        prob = plasma_step_2d(T; nsteps = 20)
        sr = state_2d(prob; mode = Reference(), current = Esirkepov())
        sf = state_2d(prob; mode = Fast(), current = Esirkepov())
        worst = 0.0
        bitwise = true
        for k in 1:prob.nsteps
            step!(sr, params(prob), k)
            step!(sf, params(prob), k)
            for c in (:Ex, :Ey, :Ez, :Bx, :By, :Bz, :Jx, :Jy, :Jz)
                a, b = Array(getfield(sr.fields, c)), Array(getfield(sf.fields, c))
                iszero(maximum(abs, a)) && continue
                worst = max(worst, maximum(abs, a .- b) / (eps(T) * maximum(abs, a) * k * sqrt(prod(prob.species[1].ppc))))
                bitwise &= a == b
            end
        end
        hr, hf = histories(sr), histories(sf)
        worst = max(worst, maximum(abs, hr.K .- hf.K) / (eps(T) * maximum(abs, hr.K) * prob.nsteps * sqrt(prod(prob.species[1].ppc))))
        @info "T-362 Reference against Fast" T worst bitwise
        record!("T-362", T, worst)
        @test worst <= C
    end
end

# --- T-363 ----------------------------------------------------------------------------------

@testset "T-363 the 2D step with current is bitwise across reruns, launch shapes and 1 and 4 threads in Reference mode [A4, WP-023]" begin
    ref = em2d_step_digests()
    @test ref == em2d_step_digests()
    for wgs in (32, 256)
        @test em2d_step_digests(; workgroupsize = wgs) == ref
    end
    for nt in (1, 4)
        lines = filter(startswith("EM2DJ "), split(worker_output(nt), '\n'))
        got = Dict(String(f[3]) => parse(UInt64, f[4]) for f in split.(lines))
        @test all(f -> parse(Int, f[2]) == nt, split.(lines))
        @test length(got) == length(ref)
        for (T, d) in ref
            @test get(got, type_label(T), nothing) == d
        end
    end
end

# --- T-364, T-365 ---------------------------------------------------------------------------

# The case files of WP-019 scaled down to test size (a quarter of the cells per axis, Δt four
# times the case's, so the Courant fraction 0.95 stays), fitted with the case's own rules on the
# module's diagnostics at every step. Landau also uses 16 × 16 electrons per cell and a 5 %
# perturbation, so that enough maxima clear the noise floor. At this size the thermal noise of
# one seed spreads the fitted Landau rate by 27 % of theory (20 seeds, `scripts/em2d/landau_seeds.jl`,
# whose seed mean agrees with theory), so T-364 bounds the rate by three seed standard deviations:
# it checks that the wave damps at the right order, and the frequency (which every seed gives
# within 5 %) and the longitudinal polarization more closely. The rate itself is established by
# the full-size runs against theory, Smilei and PIConGPU in reports/wp-023-em-step-2d/. The cold
# Weibel case has no seed noise.
const PHYSICS_2D = Dict{String,Any}()

function run_case_2d(name; be = CPU(), ppc, cells, amplitude = nothing, t_end = nothing)
    case = load_case(name)
    n0 = Tuple(case["grid"]["cells"])
    dt = case["time"]["dt"] * minimum(n0 ./ cells)
    prob = problem_from_case(case, Float64; ppc, cells, dt, amplitude, t_end)
    st = EMState(prob, numerics_from_case(case); backend = be)
    initialize!(st, prob)
    t = @elapsed run!(st, params(prob), prob.nsteps)
    return (; st, prob, case, h = histories(st), ts = history_times(prob), seconds = t)
end

function body_T364(be)
    r = run_case_2d("landau-oblique-2d"; be, ppc = (16, 16), cells = (32, 32), amplitude = 1 / 20, t_end = 18)
    o = r.case["observables"]["damping_rate_mode"]
    e = r.case["species"][1]
    kλ = r.case["wavevector"]["magnitude"] * e["thermal_velocity"] / sqrt(e["density"])
    ω = landau_root(kλ)
    @test -imag(ω) ≈ o["theory"] rtol = 1e-10
    U = abs2.(complex.(r.h.El_re, r.h.El_im))
    f = damping_from_peaks(r.ts, U; fit_start = o["fit_start"], floor_factor = o["floor_factor"])
    rel, relω = (f.gamma - o["theory"]) / o["theory"], (f.omega - real(ω)) / real(ω)
    # the wave is longitudinal: the part of mode (2, 1) perpendicular to k stays small
    k̂ = (2, 1) ./ sqrt(5)
    Et = abs.(-k̂[2] .* complex.(r.h.Ex_re, r.h.Ex_im) .+ k̂[1] .* complex.(r.h.Ey_re, r.h.Ey_im))
    transverse = maximum(Et) / maximum(sqrt.(U))
    @info "T-364 oblique Landau" backend = backend_label(be) γ_theory = o["theory"] γ = f.gamma rel ω_theory = real(ω) ω = f.omega relω peaks = length(f.t_fit) transverse seconds = r.seconds
    PHYSICS_2D[physics_key("landau-oblique-2d", be)] = (; r..., f)
    d = get!(MEASURED, "T-364 relative to theory", Dict{String,Float64}())
    d["γ $(backend_label(be))"], d["ω $(backend_label(be))"], d["transverse $(backend_label(be))"] = rel, relω, transverse
    @test length(f.t_fit) >= 3
    @test f.gamma > 0
    @test abs(rel) <= emC("T-364", "rate")
    @test abs(relω) <= emC("T-364", "frequency")
    @test transverse <= emC("T-364", "transverse")
end

@testset "T-364 oblique Landau damping in 2D against linear theory [A5, A7, WP-023]" begin
    body_T364(CPU())
end

function body_T365(be)
    r = run_case_2d("weibel-2d"; be, ppc = (4, 4), cells = (64, 64))
    o = r.case["observables"]["growth_rate_mode"]
    v0 = abs(r.case["species"][1]["drift"][3])
    γth = weibel_cold_growth_2d(r.case["wavevector"]["magnitude"], v0)
    @test γth ≈ o["theory"] rtol = 1e-12
    Bt = abs2.(complex.(r.h.Bt_re, r.h.Bt_im))
    f = growth_from_window(r.ts, Bt; window = o["window"])
    rel = (f.gamma - γth) / γth
    # the in-plane B of the seeded mode lies along ê = ẑ × k̂: its part along k̂ (∇·B = 0) stays small
    k̂ = (2, 1) ./ sqrt(5)
    Bl = abs.(k̂[1] .* complex.(r.h.Bx_re, r.h.Bx_im) .+ k̂[2] .* complex.(r.h.By_re, r.h.By_im))
    longitudinal = maximum(Bl) / maximum(sqrt.(Bt))
    imax = findfirst(==(f.t_max), r.ts)
    ΔWB = (r.h.WBx[imax] + r.h.WBy[imax]) - (r.h.WBx[1] + r.h.WBy[1])
    ΔK = r.h.K[imax] - r.h.K[1]
    @info "T-365 2D Weibel" backend = backend_label(be) γ_theory = γth γ = f.gamma rel window = extrema(f.t_fit) t_saturation = f.t_max longitudinal ΔWB ΔK seconds = r.seconds
    PHYSICS_2D[physics_key("weibel-2d", be)] = (; r..., f)
    d = get!(MEASURED, "T-365 relative to theory", Dict{String,Float64}())
    d["γ $(backend_label(be))"], d["longitudinal $(backend_label(be))"] = rel, longitudinal
    @test length(f.t_fit) >= 5
    @test abs(rel) <= emC("T-365", "rate")
    @test longitudinal <= emC("T-365", "longitudinal")
    @test ΔWB > 0 && ΔK < 0
end

@testset "T-365 2D Weibel growth rate against the cold-fluid relation [A1, A5, A7, WP-023]" begin
    body_T365(CPU())
end

# --- T-366 ----------------------------------------------------------------------------------

# Dual{Float32,3} through the full 2D step (T-360 checks Gauss's law on its partials): the primal
# against the Float32 run, and the partials ∂/∂(amplitude) of the fields finite and nonzero.
@testset "T-366 Dual through the 2D step with current [A3, A8, WP-023]" begin
    C = emC("T-366")
    a = mkdual(1 / 10, 1, 0, 0)
    pd, pf = plasma_step_2d(D3; amplitude = a, nsteps = 10), plasma_step_2d(Float32; nsteps = 10)
    sd, sf = state_2d(pd; current = Esirkepov()), state_2d(pf; current = Esirkepov())
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
    record!("T-366", D3, worst)
    @test worst <= C
end

# --- T-367 ----------------------------------------------------------------------------------

function step_bytes_2d_current(mode; ppc, cells)
    prob = plasma_step_2d(Float64; ppc, cells, nsteps = 4)
    st = state_2d(prob; mode, current = Esirkepov())
    p = params(prob)
    step!(st, p, 1)
    step!(st, p, 2)
    return @allocated step!(st, p, 3)
end

@testset "T-367 allocation per 2D step with current does not grow with particle count or grid size [A7, WP-023]" begin
    for mode in (Reference(), Fast())
        small = step_bytes_2d_current(mode; ppc = (2, 2), cells = (16, 12))
        more_particles = step_bytes_2d_current(mode; ppc = (8, 8), cells = (16, 12))
        larger_grid = step_bytes_2d_current(mode; ppc = (2, 2), cells = (64, 48))
        @info "T-367 bytes per 2D step with current" mode small more_particles larger_grid threads = Threads.nthreads()
        # at more than one thread the Reference sort allocates per particle (#82, as T-220); CI runs one thread
        if Threads.nthreads() == 1 || mode isa Fast
            @test more_particles <= small
            @test larger_grid <= small
        end
    end
end

# --- T-368 ----------------------------------------------------------------------------------

@testset "T-368 the case files at N = 2 and their seeds (#183), and the current choice [A1, A4, A10, WP-023]" begin
    # the default seed is the case's: the first of `seeds`, else `random_seed`; both or neither refused
    landau, weibel = load_case("landau"), load_case("weibel-2d")
    @test case_seed(landau) == first(landau["numerics"]["seeds"]) == 20261101
    @test problem_from_case(landau, Float64).seed == 20261101
    @test problem_from_case(landau, Float64; seed = 5).seed == 5
    @test case_seed(weibel) == weibel["numerics"]["random_seed"]
    both = deepcopy(weibel)
    both["numerics"]["seeds"] = [1, 2]
    neither = deepcopy(weibel)
    delete!(neither["numerics"], "random_seed")
    @test_throws ArgumentError problem_from_case(both, Float64)
    @test_throws ArgumentError problem_from_case(neither, Float64)
    # schema 2: every per-axis entry, the drift in three components, the isotropic thermal spread
    lo = load_case("landau-oblique-2d")
    p = problem_from_case(lo, Float64)
    @test p isa EMProblem{2,Float64}
    @test p.length == Tuple(lo["grid"]["length"]) && p.cells == (128, 128) && p.kmode == (2, 1)
    @test p.nsteps == round(Int, lo["time"]["t_end"] / lo["time"]["dt"]) && p.dt == lo["time"]["dt"]
    @test p.seed == 20261031
    e = only(p.species)                                 # the immobile ions are the background
    @test e.ppc == (32, 32) && e.mode == (2, 1) && e.amplitude == 0.01 && e.thermal == (0.05, 0.05, 0.05)
    w = problem_from_case(weibel, Float32; ppc = (2, 3), cells = (16, 12), amplitude = 1 // 10)
    @test w isa EMProblem{2,Float32} && w.cells == (16, 12)
    @test [s.drift for s in w.species] == [(0f0, 0f0, 0.3f0), (0f0, 0f0, -0.3f0)]
    @test [s.amplitude for s in w.species] == [0.1f0, -0.1f0] && all(s -> s.ppc == (2, 3) && s.thermal == (0f0, 0f0, 0f0), w.species)
    # an asymmetric schema-2 case: every per-axis entry keeps its axis (both committed cases are
    # symmetric per axis, so they cannot show a swap)
    asym = deepcopy(weibel)
    asym["grid"]["length"], asym["grid"]["cells"], asym["wavevector"]["mode"] = [3.0, 2.0], [12, 10], [2, -1]
    asym["species"][1]["particles_per_cell"], asym["species"][1]["drift"] = [3, 2], [0.1, -0.2, 0.3]
    asym["species"][1]["perturbation"] = Dict("amplitude" => 0.01, "mode" => [2, -1])
    pa = problem_from_case(asym, Float64)
    @test pa.length == (3.0, 2.0) && pa.cells == (12, 10) && pa.kmode == (2, -1)
    a1 = first(pa.species)
    @test a1.ppc == (3, 2) && a1.drift == (0.1, -0.2, 0.3) && a1.mode == (2, -1) && a1.amplitude == 0.01
    @test problem_from_case(asym, Float64; ppc = (5, 4), cells = (9, 7)).species[2].ppc == (5, 4)
    @test numerics_from_case(lo).current isa Esirkepov
    # the current: Esirkepov() at N = 1 and 2 (the default), NoCurrent() at N = 2 only
    @test EMNumerics().current isa Esirkepov
    @test EMState(plasma_step_2d(Float64), EMNumerics()) isa EMState{2}
    @test_throws ArgumentError EMState(plasma_step_2d(Float64), EMNumerics(; current = nothing))
    @test_throws ArgumentError EMState(beams_1d(Float64), EMNumerics(; current = nothing))
    @test_throws ArgumentError EMState(beams_1d(Float64), EMNumerics(; current = NoCurrent()))
end

# --- T-369 ----------------------------------------------------------------------------------

@testset "T-369 the in-plane magnetic diagnostics against closed forms [A7, WP-023]" begin
    C = emC("T-354")
    for T in TYPES
        ε = base_eps(T)
        m = (2, -1)
        L = (T(3), T(2))
        prob = test_particles_2d(T; cells = (12, 10), length = L, dt = 1 // 10, nsteps = 2, kmode = m)
        st = state_2d(prob)
        (Ax, ψx), (Ay, ψy) = (T(2) / 3, -one(T)), (T(1) / 7, T(1) / 2)
        set_mode!(st.fields.Bx, st.grid, L2D.Bx, m, L, Ax, ψx)
        set_mode!(st.fields.By, st.grid, L2D.By, m, L, Ay, ψy)
        record_diagnostics!(st, 1)
        h = histories(st)
        k = (2π * m[1] / value_of(L[1]), 2π * m[2] / value_of(L[2]))
        k̂ = k ./ hypot(k...)
        ax, ay = value_of(Ax), value_of(Ay)
        exact = Dict(:Bx_re => ax * cos(value_of(ψx)), :Bx_im => ax * sin(value_of(ψx)),
                     :By_re => ay * cos(value_of(ψy)), :By_im => ay * sin(value_of(ψy)))
        exact[:Bt_re] = k̂[1] * exact[:By_re] - k̂[2] * exact[:Bx_re]
        exact[:Bt_im] = k̂[1] * exact[:By_im] - k̂[2] * exact[:Bx_im]
        worst = maximum(abs(value_of(getfield(h, key)[1]) - v) / ε for (key, v) in exact)
        # the transposed stagger (Bx on By's positions) is off by half a cell on both axes
        set_mode!(st.fields.Bx, st.grid, L2D.By, m, L, Ax, ψx)
        record_diagnostics!(st, 2)
        off = abs(value_of(histories(st).Bx_re[2]) - exact[:Bx_re]) / ε
        record!("T-369", T, worst)
        @test worst <= C
        @test off > 100C
    end
end
