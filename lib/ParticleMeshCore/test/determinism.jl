# The sort under the Reference mode: T-136 (A4). The bits of the Reference kernels across
# reruns and thread counts are checked by T-172 to T-177, T-113, T-260 and T-266 (the
# subprocess digests of threads_worker.jl).

@testset "T-136 sort by cell is a stable permutation with consistent segments [A4, WP-010]" begin
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
