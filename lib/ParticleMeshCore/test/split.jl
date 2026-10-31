# T-170: moving the electrostatic reference out of the engine changed no bit. Needs the runs of
# physics.jl (T-141, T-142). The digests below were recorded before the move, at the commit
# WP-011 branched from (4ed3667), on the CPU backend with Julia 1.13.3: the FNV digests with
# 1 and 4 threads (identical), the T-136 digest by `determinism_worker.jl` with 1 thread.

# FNV digests (`fnv_state_digest`) of the reference cases after 40 steps, 32 cells, 64 particles
# per cell, 2/ω_p, in Reference mode; for `Dual` the Fast mode falls back to Reference.
const SPLIT_CASES = (
    ("landau", Float32, 1, 0xcee47958109a41e2), ("two_stream", Float32, 1, 0x9f927f27f7c077aa),
    ("landau", Float32, 2, 0xfcaf264c569c57e6), ("two_stream", Float32, 2, 0x972b5f1f40ec2c65),
    ("landau", Float64, 1, 0x1b7287480e1200a6), ("two_stream", Float64, 1, 0x90e1f118e4789e5a),
    ("landau", Float64, 2, 0xbacad7631adabe69), ("two_stream", Float64, 2, 0xcdfd052049af2d22),
    ("landau", D3, 1, 0xdd9925eac8a40da5), ("two_stream", D3, 1, 0x29d53f2f02efe5e2),
    ("landau", D3, 2, 0xfc4209a2f659b33d), ("two_stream", D3, 2, 0xc235c4207311f3ae),
)
const SPLIT_LANDAU_FULL = 0x838abc0fecdc0cef       # LANDAU_KW, the whole run of T-141
const SPLIT_TWO_STREAM_FULL = 0x8bfb13fd25fab0d8   # TWO_STREAM_KW, the whole run of T-142
const SPLIT_T136 = 0xf07021cf9f95d7c0              # T-136's `state_digest` of the T-141 run (Base.hash)

@testset "T-170 the split preserves the bits [A0, A4, WP-011]" begin
    # The spike's recorded rates (reports/spike-electrostatic-1d.md, scripts/es1d/ at 16 threads).
    @test PHYSICS["landau"].γ === -0.1492419736219506
    @test PHYSICS["two_stream"].γ === 0.3545690079405174
    @test PHYSICS["landau_fnv"] == SPLIT_LANDAU_FULL
    @test PHYSICS["two_stream_fnv"] == SPLIT_TWO_STREAM_FULL
    # `hash` is not fixed across Julia versions; the FNV digest above carries the claim there.
    @test PHYSICS["landau_digest"] == SPLIT_T136 skip = VERSION >= v"1.14-"
    for (case, T, P, expected) in SPLIT_CASES, mode in (Reference(), Fast())
        effective_mode(mode, T) isa Reference || continue
        shape = Shape{P}()
        st, p, _ = if case == "landau"
            landau_case(T; n = 32, ppc = 64, α = T === D3 ? mkdual(1 // 20, 1, 0, 0) : T(1 // 20), shape, mode, tmax_ωp = 2)
        else
            two_stream_case(T; n = 32, ppc = 64, shape, mode, tmax_ωp = 2)
        end
        initialize!(st, p)
        run!(st, p, 40)
        @test fnv_state_digest(st) == expected
    end
end
