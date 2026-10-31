# Physics against linear theory: T-141 and T-142. Measured, not thresholded (A7: measure, then
# propose); the relative errors go to tolerances.toml as proposals.

include("cases.jl")

include("digest.jl")

const PHYSICS = Dict{String,Any}()

@testset "T-141 Landau damping rate against linear theory [A5, A7, WP-010]" begin
    st, p, info = landau_case(Float64; LANDAU_KW...)
    initialize!(st, p)
    t = @elapsed run!(st, p, info.nsteps)
    PHYSICS["landau_digest"] = state_digest(st)
    A = mode_amplitude_history(st)
    ts = (0:(info.nsteps - 1)) .* info.dt
    ωth = landau_root(Float64(info.kλD))
    f = fit_damping(ts, A; ω_guess = real(ωth), tmax = LANDAU_FIT_TMAX)
    relγ = (f.γ - imag(ωth)) / imag(ωth)
    relω = (f.ω - real(ωth)) / real(ωth)
    @info "T-141 Landau damping" kλD = info.kλD ω_theory = ωth γ_measured = f.γ ω_measured = f.ω relγ relω peaks = length(f.tpeaks) particles = info.n * info.ppc steps = info.nsteps seconds = t
    PHYSICS["landau"] = (; f.γ, f.ω, ωth, relγ, relω)
    @test isfinite(f.γ) && isfinite(f.ω)
    @test f.γ < 0                       # qualitative: the wave damps
end

@testset "T-142 two-stream growth rate against linear theory [A5, A7, WP-010]" begin
    st, p, info = two_stream_case(Float64; TWO_STREAM_KW...)
    initialize!(st, p)
    t = @elapsed run!(st, p, info.nsteps)
    A = mode_amplitude_history(st)
    ts = (0:(info.nsteps - 1)) .* info.dt
    γth = two_stream_growth(Float64(info.kv0_ωp))
    f = fit_growth(ts, A)
    rel = (f.γ - γth) / γth
    @info "T-142 two-stream" kv0_ωp = info.kv0_ωp γ_theory = γth γ_measured = f.γ rel window = f.window seconds = t
    PHYSICS["two_stream"] = (; f.γ, γth, rel)
    @test isfinite(f.γ)
    @test f.γ > 0                       # qualitative: the mode grows
end
