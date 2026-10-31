# Bitwise reproducibility of the Reference mode: T-136 (A4).

@testset "T-136 Reference mode is bitwise reproducible across reruns and thread counts [A4, WP-010]" begin
    @testset "sort by cell is a stable permutation with consistent segments" begin
        for T in TYPES
            grid = PeriodicGrid(9, one(basetype(T)), 2)
            s = test_species(T, grid, 500; seed = 13)
            s.status[17] = ParticleMeshCore.FREE              # a hole in the middle
            before = Dict(Int(value_of(s.u[2][i])) => (s.cell[1][i], s.w[i]) for i in 1:500 if i != 17)
            ws = Workspace(s, grid)
            sort_by_cell!(s, grid, ws)
            @test alive_count(s) == 499
            @test issorted(s.cell[1][1:499])
            @test all(==(ParticleMeshCore.ALIVE), s.status[1:499]) && all(==(ParticleMeshCore.FREE), s.status[500:end])
            @test sum(max.(0, ws.seg_stop .- ws.seg_start .+ 1)) == 499
            @test all(c -> all(==(c), s.cell[1][ws.seg_start[c]:ws.seg_stop[c]]), 1:9)
            @test all(i -> before[Int(value_of(s.u[2][i]))] == (s.cell[1][i], s.w[i]), 1:499)
        end
    end

    # One step, rerun in process (Float32, Float64, Dual)
    for T in TYPES
        digests = map(1:2) do _
            st, p, _ = landau_case(T; n = 64, ppc = 512, mode = Reference(), tmax_ωp = 1)
            initialize!(st, p)
            step!(st, p, 1)
            hash((map(s -> (s.offset[1], s.u[1], s.w), st.species), st.ρ, st.Ehalf))
        end
        @test digests[1] == digests[2]
    end
    # The whole Landau run of T-141 and one Float32 step, in subprocesses with 1 and 4 threads,
    # against each other and against the in-process run of T-141.
    worker = joinpath(@__DIR__, "determinism_worker.jl")
    project = dirname(Base.active_project())
    results = map((1, 4)) do nt
        out = read(`$(Base.julia_cmd()) --startup-file=no --threads=$nt --project=$project $worker`, String)
        line = only(filter(startswith("DIGEST"), split(out, '\n')))
        _, n, one, full = split(line)
        one, full = parse(UInt64, one), parse(UInt64, full)
        (; threads = parse(Int, n), one, full)
    end
    # T-141 stores its digest in PHYSICS; run alone, this file computes it here.
    inprocess = if @isdefined(PHYSICS) && haskey(PHYSICS, "landau_digest")
        PHYSICS["landau_digest"]
    else
        st, p, info = landau_case(Float64; LANDAU_KW...)
        initialize!(st, p)
        run!(st, p, info.nsteps)
        state_digest(st)
    end
    @info "T-136 digests" results inprocess inprocess_threads = Threads.nthreads()
    @test results[1].threads == 1 && results[2].threads == 4
    @test results[1].one == results[2].one
    @test results[1].full == results[2].full
    @test results[1].full == inprocess
end
