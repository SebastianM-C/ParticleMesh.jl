# The time-step check of EMState, from a review of b714331 (#202).

@testset "T-343 EMState refuses dt ≥ Δx, the one-cell-per-step bound of deposit_flux! and the 1D Courant limit [A2, A7]" begin
    case = load_case("two-stream")
    prob = problem_from_case(case, Float64)
    dx = prob.length / prob.cells
    @test prob.dt < dx
    num = numerics_from_case(case; mode = Reference())
    @test EMState(prob, num) isa EMState
    for f in (1.0, 2.0)
        bad = problem_from_case(case, Float64; cells = ceil(Int, f * prob.length / prob.dt))
        @test_throws ArgumentError EMState(bad, num)
    end
end
