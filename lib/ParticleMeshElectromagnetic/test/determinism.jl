# T-218: the Reference mode is bitwise reproducible across reruns and thread counts (A4).

foreach(start_determinism_worker, DETERMINISM_THREADS)   # the worker runs (determinism_cases.jl) beside this file's in-process runs

@testset "T-218 Reference mode is bitwise identical across reruns and 1 and 4 threads [A4, WP-014]" begin
    first = determinism_digests()
    second = determinism_digests()
    @test first == second
    # two explicit launch shapes give the same bits as the backend default
    for wgs in (32, 256)
        @test determinism_digests(; workgroupsize = wgs) == first
    end
    for nt in (1, 4)
        out = worker_output(nt)
        lines = filter(startswith("DIGEST"), split(out, '\n'))
        got = Dict((String(f[3]), String(f[4])) => parse(UInt64, f[5]) for f in split.(lines))
        @test all(f -> parse(Int, f[2]) == nt, split.(lines))
        @test length(got) == length(first)
        for (name, T, d) in first
            @test get(got, (name, type_label(T)), nothing) == d
        end
    end
end
