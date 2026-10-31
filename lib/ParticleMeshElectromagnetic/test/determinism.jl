# T-218: the Reference mode is bitwise reproducible across reruns and thread counts (A4).

@testset "T-218 Reference mode is bitwise identical across reruns and 1 and 4 threads [A4, WP-014]" begin
    first = determinism_digests()
    second = determinism_digests()
    @test first == second
    # two explicit launch shapes give the same bits as the backend default
    for wgs in (32, 256)
        @test determinism_digests(; workgroupsize = wgs) == first
    end
    worker = joinpath(@__DIR__, "determinism_worker.jl")
    project = dirname(Base.active_project())
    for nt in (1, 4)
        out = read(`$(Base.julia_cmd()) --startup-file=no --threads=$nt --project=$project $worker`, String)
        lines = filter(startswith("DIGEST"), split(out, '\n'))
        got = Dict((String(f[3]), String(f[4])) => parse(UInt64, f[5]) for f in split.(lines))
        @test all(f -> parse(Int, f[2]) == nt, split.(lines))
        @test length(got) == length(first)
        for (name, T, d) in first
            @test get(got, (name, type_label(T)), nothing) == d
        end
    end
end
