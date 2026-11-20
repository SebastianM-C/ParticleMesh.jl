# WP-070: the 1D cases of physics.jl with the Galerkin gather, T-994 and T-995. Reuses physics.jl's
# configurations and its runs with today's gather (PHYSICS, run_case, energy_drift).

"Σ_s m_s Σ_i w_i u_i of the alive particles, per component, on the host."
total_momentum(st) = ntuple(d -> sum(s -> s.m * sum(Array(s.w) .* Array(s.u[d]) .* (Array(s.status) .== ParticleMeshCore.ALIVE)), st.species), 3)
"Σ_s m_s Σ_i w_i |u_i| of the alive particles, the scale of `total_momentum`."
momentum_scale(st) = sum(s -> s.m * sum(Array(s.w) .* hypot.(Array(s.u[1]), Array(s.u[2]), Array(s.u[3])) .* (Array(s.status) .== ParticleMeshCore.ALIVE)), st.species)

"The largest change of the total momentum's components from the start to the end of the run `r`, over the momentum scale at the start."
function momentum_change(r, name, overrides)
    st0, = case_state(name, Float64; overrides...)                   # the same load: the start of the run
    return maximum(abs.(total_momentum(r.st) .- total_momentum(st0))) / momentum_scale(st0)
end

# --- T-994 ----------------------------------------------------------------------------------

# The three 1D cases of T-214 to T-217 in their test configurations with the Galerkin gather. The
# growth rates of two-stream and Weibel keep T-215's and T-216's bounds against theory (the
# coupling changes nothing the linear theory sees), Landau damps; the total energy (fields plus
# kinetic, T-217's measure) changes at most by T-994's bounds, each less than a fifth of today's
# measured change (T-217's `measured`). The total momentum: the cold symmetric beams keep it to
# roundoff with either gather; in Landau's thermal plasma it changes over the run by the same
# amount with both, to 1 % (measured 1.6·10⁻³ of Σ m w |u| with both).
function body_T994(be)
    tol = EMTOL["T-994"]
    for (name, ov) in (("landau", LANDAU_OVERRIDES), ("two-stream", TWO_STREAM_OVERRIDES), ("weibel", WEIBEL_OVERRIDES))
        r = run_case(name; be, gather = GalerkinGather(), ov...)
        d = energy_drift(r.h)
        k = physics_key(name, be)
        dP = momentum_change(r, name, (; be, gather = GalerkinGather(), ov...))
        get!(MEASURED, "T-994 relative energy drift", Dict{String,Float64}())[k] = d
        get!(MEASURED, "T-994 relative momentum change", Dict{String,Float64}())[k] = dP
        if name == "landau"
            o = r.case["observables"]["damping_rate"]
            f = damping_from_peaks(r.ts, r.h.WEx; fit_start = o["fit_start"], floor_factor = o["floor_factor"])
            @test isfinite(f.gamma) && f.gamma > 0
            rel = NaN
        else
            id = name == "two-stream" ? "T-215" : "T-216"
            o = r.case["observables"]["growth_rate"]
            if name == "two-stream"
                b = r.case["species"][2]
                v0 = abs(b["drift"])
                γth = two_stream_cold_growth(2π / r.case["grid"]["length"] * v0 / sqrt(2b["density"]); gamma0 = 1 / sqrt(1 - v0^2))
                f = growth_from_window(r.ts, r.h.WEx; window = o["window"])
            else
                b = r.case["species"][1]
                v0 = abs(b["transverse_drift"][1])
                n0 = sum(s -> s["mobile"] ? s["density"] : 0.0, r.case["species"])
                γth = weibel_cold_growth(2π / r.case["grid"]["length"], v0; ωp2 = n0)
                f = growth_from_window(r.ts, r.h.WBz; window = o["window"])
            end
            rel = (f.gamma - γth) / γth
            @test abs(rel - EMTOL[id]["relative_error"]["gamma"]) <= EMTOL[id]["bound"]
        end
        @info "T-994 Galerkin" name backend = backend_label(be) rel drift = d today = EMTOL["T-217"]["measured"][name] momentum = dP seconds = r.seconds
        @test d <= tol[name]
        @test d <= EMTOL["T-217"]["measured"][name] / 5
        if name == "landau"
            plain = haskey(PHYSICS, k) ? momentum_change(PHYSICS[k], name, (; be, ov...)) : nothing
            plain === nothing || (MEASURED["T-994 relative momentum change"]["$k, today's gather"] = plain)
            plain === nothing || @test abs(dP - plain) <= plain / 100
        else
            @test dP <= tol["momentum_cold"]
        end
    end
end

@testset "T-994 the 1D cases with the Galerkin gather: two-stream and Weibel rates within T-215's and T-216's bounds, the total energy kept five times better than today's at least, the momentum as today's [A2, A5, A7, WP-070]" begin
    body_T994(CPU())
end

# --- T-995 ----------------------------------------------------------------------------------

# Where the energy error comes from. The Weibel case (T-216's configuration) at Δt, Δt/2 and Δt/4:
# with the Galerkin gather the largest relative change of the total energy falls by 4 per halving
# (measured 4.03 and 3.97), the leapfrog's second order: what is left is the time integrator's
# error. With today's gather it stays within 3 % (measured 2.83, 2.76 and 2.74 × 10⁻⁴), an error
# in space that a smaller step does not touch.
@testset "T-995 the Galerkin gather's energy error is the time integrator's: it falls by four per halving of Δt in the Weibel case, today's does not fall [A2, A7, WP-070]" begin
    dt0 = load_case("weibel")["time"]["dt"]
    drifts(gather) = [energy_drift(run_case("weibel"; dt = dt0 / 2^k, gather, WEIBEL_OVERRIDES...).h) for k in 0:2]
    dg, dp = drifts(GalerkinGather()), drifts(nothing)
    rg, rp = dg[1:2] ./ dg[2:3], dp[1:2] ./ dp[2:3]
    @info "T-995 energy drift at Δt, Δt/2, Δt/4" galerkin = dg ratios = rg today = dp today_ratios = rp
    MEASURED["T-995 drift ratio per halving"] = Dict("galerkin" => minimum(rg), "today" => maximum(rp))
    @test all(r -> 3.6 <= r <= 4.4, rg)
    @test all(r -> 0.97 <= r <= 1.05, rp)
end
