# The three 1D cases through the module (A5): T-213 to T-217. Rates are measured with the
# observables of the case files (scripts/crosscode/analysis.jl, the same fits the Smilei runs
# use) and recorded, not thresholded (A7: measure, then propose); only finiteness and sign are
# asserted. The electrostatic reference (lib/ParticleMeshCore/test/es_reference.jl) runs the
# same edge cases with the same seed, grid, time step and particles per cell; the schemes differ
# by construction (Ampère with the relativistic Boris push against Gauss with the
# nonrelativistic leapfrog), so their difference is recorded, never asserted.

const PHYSICS = Dict{String,Any}()

# Test configurations: the case files with fewer particles. Landau also uses a larger
# perturbation (5 %, as T-141 does): at the case's 1 % the damped signal falls under the noise
# floor before the fit window opens. At 4096 particles per cell only two maxima after the
# case's fit_start clear its floor_factor (the fit needs three, measured); 8192 gives three.
const LANDAU_OVERRIDES = (; ppc = 8192, amplitude = 1 // 20)
const TWO_STREAM_OVERRIDES = (; ppc = 256)
const WEIBEL_OVERRIDES = (;)

"Run a case to its end; returns the histories, the times, the final state and the run time."
function run_case(name; overrides...)
    st, p, prob, case = case_state(name, Float64; overrides...)
    t = @elapsed run!(st, p, prob.nsteps)
    return (; st, prob, case, h = histories(st), ts = history_times(prob), seconds = t)
end

"The electrostatic reference on the same edge case (same grid, Δt, particles per cell, seed)."
function es_reference_run(name, prob, case)
    s = case["species"]
    tmax = prob.nsteps * prob.dt
    if name == "landau"
        e = s[1]
        kλD = 2π / case["grid"]["length"] * e["thermal_velocity"] / sqrt(e["density"])
        st, p, info = landau_case(Float64; n = prob.cells, ppc = prob.species[1].ppc, n0 = e["density"],
                                  vth = e["thermal_velocity"], kλD, α = prob.species[1].amplitude,
                                  dt_ωp = prob.dt, tmax_ωp = tmax, seed = prob.seed)
    else
        b = s[2]
        n0 = 2b["density"]
        kv0 = 2π / case["grid"]["length"] * abs(b["drift"]) / sqrt(n0)
        st, p, info = two_stream_case(Float64; n = prob.cells, ppc = prob.species[1].ppc, n0, v0 = abs(b["drift"]),
                                      kv0_ωp = kv0, α = prob.species[1].amplitude, dt_ωp = prob.dt,
                                      tmax_ωp = tmax, seed = prob.seed)
    end
    ESReference.initialize!(st, p)
    ESReference.run!(st, p, info.nsteps)
    return field_energy_history(st)
end

"T-213: the transverse fields and momenta of an edge case are +0.0 bit for bit, at every step and at the end."
function check_electrostatic(r)
    f = r.st.fields
    @test all(c -> all_plus_zero(getfield(r.h, c)), (:WEy, :WEz, :WBx, :WBy, :WBz))
    @test all(c -> all_plus_zero(getfield(f, c)), (:Ey, :Ez, :Bx, :By, :Bz, :Jy, :Jz))
    @test all(s -> all_plus_zero(s.u[2]) && all_plus_zero(s.u[3]), r.st.species)
end

"T-217: the largest relative change of the total energy (fields plus kinetic) over the run."
function energy_drift(h)
    tot = total_energy_history(h)
    return maximum(abs, (tot .- tot[1]) ./ tot[1])
end

@testset "T-214 Landau damping through the module against linear theory [A5, A7, A9, WP-014]" begin
    r = run_case("landau"; LANDAU_OVERRIDES...)
    o = r.case["observables"]["damping_rate"]
    e = r.case["species"][1]
    kλ = 2π / r.case["grid"]["length"] * e["thermal_velocity"] / sqrt(e["density"])
    ω = landau_root(kλ)
    f = damping_from_peaks(r.ts, r.h.WEx; fit_start = o["fit_start"], floor_factor = o["floor_factor"])
    W_es = es_reference_run("landau", r.prob, r.case)
    fe = damping_from_peaks(r.ts, W_es; fit_start = o["fit_start"], floor_factor = o["floor_factor"])
    rel = (f.gamma - (-imag(ω))) / (-imag(ω))
    @info "T-214 Landau" γ_theory = -imag(ω) γ = f.gamma ω_theory = real(ω) ω = f.omega rel peaks = length(f.t_fit) γ_es = fe.gamma rel_es = (f.gamma - fe.gamma) / fe.gamma seconds = r.seconds
    PHYSICS["landau"] = (; r..., f, fe, ω)
    @test isfinite(f.gamma) && isfinite(f.omega)
    @test f.gamma > 0                    # the field damps (γ is the damping rate of the amplitude)
end

@testset "T-215 two-stream through the module against linear theory [A5, A7, A9, WP-014]" begin
    r = run_case("two-stream"; TWO_STREAM_OVERRIDES...)
    o = r.case["observables"]["growth_rate"]
    b = r.case["species"][2]
    v0 = abs(b["drift"])
    kv0 = 2π / r.case["grid"]["length"] * v0 / sqrt(2b["density"])
    γth = two_stream_cold_growth(kv0; gamma0 = 1 / sqrt(1 - v0^2))
    @test γth ≈ o["theory"] rtol = 1e-12
    f = growth_from_window(r.ts, r.h.WEx; window = o["window"])
    W_es = es_reference_run("two-stream", r.prob, r.case)
    fe = growth_from_window(r.ts, W_es; window = o["window"])
    rel = (f.gamma - γth) / γth
    @info "T-215 two-stream" γ_theory = γth γ = f.gamma rel window = extrema(f.t_fit) γ_es = fe.gamma rel_es = (f.gamma - fe.gamma) / fe.gamma seconds = r.seconds
    PHYSICS["two-stream"] = (; r..., f, fe, γth)
    @test isfinite(f.gamma)
    @test f.gamma > 0
end

@testset "T-213 the edge cases stay electrostatic, bit for bit [A9, WP-014]" begin
    for name in ("landau", "two-stream")
        haskey(PHYSICS, name) && check_electrostatic(PHYSICS[name])
    end
end

@testset "T-216 Weibel growth rate against the cold-fluid relation [A1, A5, A7, WP-014]" begin
    r = run_case("weibel"; WEIBEL_OVERRIDES...)
    o = r.case["observables"]["growth_rate"]
    b = r.case["species"][1]
    v0 = abs(b["transverse_drift"][1])
    n0 = sum(s -> s["mobile"] ? s["density"] : 0.0, r.case["species"])
    k = 2π / r.case["grid"]["length"]
    γth = weibel_cold_growth(k, v0; ωp2 = n0)
    # the theory value: in the case file, a root of the dispersion relation, and its limits
    @test γth ≈ o["theory"] rtol = 1e-12
    @test abs(weibel_dispersion(im * γth, k, v0; ωp2 = n0)) < 1e-12
    @test weibel_cold_growth(1e4, v0) ≈ v0 * (1 - v0^2)^(1 / 4) rtol = 1e-6       # ω_p v₀/√γ₀
    @test weibel_cold_growth(1e4, 1e-3) ≈ 1e-3 rtol = 1e-6                         # nonrelativistic: ω_p v₀
    @test issorted([weibel_cold_growth(q, v0) for q in 0.5:0.5:10])
    f = growth_from_window(r.ts, r.h.WBz; window = o["window"])
    rel = (f.gamma - γth) / γth
    # energy transfer from the beams to the magnetic field over the linear phase to saturation
    imax = findfirst(==(f.t_max), r.ts)
    ΔWB = r.h.WBz[imax] - r.h.WBz[1]
    ΔK = r.h.K[imax] - r.h.K[1]
    @info "T-216 Weibel" γ_theory = γth γ = f.gamma rel window = extrema(f.t_fit) t_saturation = f.t_max ΔWB ΔK seconds = r.seconds
    PHYSICS["weibel"] = (; r..., f, γth)
    @test isfinite(f.gamma)
    @test f.gamma > 0
    @test ΔWB > 0 && ΔK < 0
end

@testset "T-217 total energy drift per case [A7, WP-014]" begin
    for name in ("landau", "two-stream", "weibel")
        haskey(PHYSICS, name) || continue
        d = energy_drift(PHYSICS[name].h)
        get!(MEASURED, "T-217 relative energy drift", Dict{String,Float64}())[name] = d
        @info "T-217 energy drift" name drift = d steps = PHYSICS[name].prob.nsteps
        @test d <= EMTOL["T-217"][name]
    end
end
