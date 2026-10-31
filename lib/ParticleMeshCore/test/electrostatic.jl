# The electrostatic module: T-137 and T-144.

function gauss_residual(st)
    grid = st.grid
    n, g, dx = ncells(grid), guards(grid), cellsize(grid)
    ρ = st.ρ[(g + 1):(g + n)]
    ρ̄ = sum(ρ) / n
    E = st.Ehalf
    r = maximum(absdiff((E[j] - E[mod1(j - 1, n)]) / dx, ρ[j] - ρ̄) for j in 1:n)
    return r, maximum(absmax, ρ), absmax(sum(E) / n)
end

@testset "T-137 discrete Gauss law after every step [A2, A7, A8, WP-010]" begin
    # (Ehalf[j] - Ehalf[j-1])/dx = ρ[j] - mean(ρ), bound C·eps(T)·n·max|ρ|; also mean(Ehalf) = 0.
    C = tolC("T-137")
    for T in TYPES, shape in SHAPES, mode in (Reference(), Fast())
        α = T === D3 ? mkdual(1 // 20, 1, 0, 0) : T(1 // 20)
        st, p, info = landau_case(T; n = 32, ppc = 32, α, shape, mode, tmax_ωp = 5)
        initialize!(st, p)
        worst = 0.0
        for k in 1:info.nsteps
            step!(st, p, k)
            r, ρmax, Ē = gauss_residual(st)
            worst = max(worst, r / (base_eps(T) * ncells(st.grid) * ρmax))
            worst = max(worst, Ē / (base_eps(T) * ncells(st.grid) * ρmax * value_of(cellsize(st.grid))))
        end
        record!("T-137", T, worst)
        @test worst <= C
    end
end

@testset "T-144 allocation of one step does not grow with particle count or grid size [A7, WP-010]" begin
    # After warm-up, bytes allocated by one step at two particle counts and two grid sizes.
    # The CPU backend allocates a constant amount per kernel launch (task bookkeeping); what
    # must not happen is growth with the problem size.
    for mode in (Reference(), Fast()), T in (Float32, Float64)
        bytes = Dict{Tuple{Int,Int},Int}()
        for (n, ppc) in ((32, 16), (32, 256), (128, 16))
            st, p, _ = landau_case(T; n, ppc, mode, tmax_ωp = 1)
            initialize!(st, p)
            step!(st, p, 1); step!(st, p, 2)
            bytes[(n, ppc)] = @allocated step!(st, p, 3)
        end
        @info "T-144 bytes per step" mode T bytes
        # Julia 1.14 allocates per particle in some of these cases, and which ones differs between
        # machines (#58), so the check is skipped there (recorded as broken); the @info above
        # still logs the bytes. Remove the skip when #58 is closed.
        @test bytes[(32, 256)] <= bytes[(32, 16)] skip = VERSION >= v"1.14-"
        @test bytes[(128, 16)] <= bytes[(32, 16)]
    end
end
