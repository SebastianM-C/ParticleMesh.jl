# WP-033 in the module: T-537 and T-539. Which distribution a species loads (the default rule,
# the explicit keyword and the case-file key), that `initialize!` writes the engine's load of
# that distribution and converts only velocity draws, and that the hot drifting plasma of #257,
# which the velocity draw refuses, loads and steps under the default. After subluminal.jl (its
# helpers) and em2d_cases.jl (`plasma_step_2d`).

"The momenta `initialize!` leaves (with `momenta_at = :t_minus_half`, so no kick) and the engine's own load of `dist` for species 1 of `prob`."
function loaded_and_engine(prob, dist; be = CPU())
    st = EMState(prob, EMNumerics(); backend = be)
    initialize!(st, prob; momenta_at = :t_minus_half)
    d = prob.species[1]
    T = typeof(d.charge)
    N = ndims(prob)
    s = Species(be, T, prod(prob.cells) * prod(d.ppc); q = d.charge, m = d.mass, N)
    k = N == 1 ? (d.density, d.amplitude, 2 * T(π) * d.mode / prob.length) :
        (d.density, d.amplitude, ntuple(a -> 2 * T(π) * d.mode[a] / prob.length[a], 2)...)
    load_quiet!(s, st.grid, ParticleMeshElectromagnetic.CosineProfile(), k, d.ppc, dist; seed = prob.seed, stream = d.stream)
    return st.species[1], s
end

function body_T537(be)
    for T in TYPES
        z, v = zero(T), T(1 // 20)
        sd(th; kw...) = SpeciesDescription{T}(:e, -one(T), one(T), one(T), z, 1, 4, (T(1 // 10), z, z), th, 1; kw...)
        # the default rule
        @test sd((v, v, v)).momentum === :maxwell_juttner
        @test sd((v, z, z)).momentum === :gaussian_momentum
        @test sd((v, v, 2v)).momentum === :gaussian_momentum
        @test sd((z, z, z)).momentum === :gaussian_velocity
        for m in (:maxwell_juttner, :gaussian_momentum, :gaussian_velocity)
            @test sd((v, v, v); momentum = m).momentum === m
        end
        @test_throws ArgumentError sd((v, v, v); momentum = :juttner)
        @test_throws ArgumentError sd((v, z, z); momentum = :maxwell_juttner)     # not isotropic
        @test_throws ArgumentError sd((z, z, z); momentum = :maxwell_juttner)     # θ = 0
        # the engine objects
        mj = momentum_distribution(sd((v, v, v)))
        @test mj isa MaxwellJuttner{T} && mj.θ == v^2 && mj.β == (T(1 // 10), z, z)
        gm = momentum_distribution(sd((v, z, z)))
        Γ = one(T) / sqrt(one(T) - T(1 // 10)^2)
        @test gm isa GaussianMomentum{T} && gm.udrift == (Γ * T(1 // 10), z, z) && gm.uth == (v, z, z)
        @test momentum_distribution(sd((z, z, z))) == GaussianVelocity((T(1 // 10), z, z), (z, z, z))
        fast(m) = SpeciesDescription{T}(:e, -one(T), one(T), one(T), z, 1, 4, (T(6 // 10), T(8 // 10), z), (v, v, v), 1; momentum = m)
        @test_throws ArgumentError momentum_distribution(fast(:maxwell_juttner))
        @test_throws ArgumentError momentum_distribution(fast(:gaussian_momentum))
        @test momentum_distribution(fast(:gaussian_velocity)) isa GaussianVelocity      # refused later, by check_subluminal
        @test_throws ArgumentError momentum_distribution(sd((T(NaN), z, z)))
        # initialize! writes the engine's load: momenta as drawn, velocities converted to u = γv
        for N in (1, 2), m in (:maxwell_juttner, :gaussian_momentum, :gaussian_velocity)
            prob = N == 1 ? electrons_1d(T; vth = T(1 // 10), drift = (T(1 // 5), z, T(-1 // 10)), momentum = m) :
                   electrons_2d(T; vth = T(1 // 10), drift = (T(1 // 5), z, T(-1 // 10)), momentum = m)
            dist = momentum_distribution(prob.species[1])
            got, eng = loaded_and_engine(prob, dist; be)
            u = map(Array, got.u)
            e = map(Array, eng.u)
            if draws_momentum(dist)
                @test all(d -> same_bits(u[d], e[d]), 1:3)
            else
                γ = [one(T) / sqrt(one(T) - (e[1][i]^2 + e[2][i]^2 + e[3][i]^2)) for i in eachindex(e[1])]
                @test all(d -> same_bits(u[d], γ .* e[d]), 1:3)
            end
        end
    end
    # the case-file key, both schemas, and the keyword that replaces it
    lan = load_case("landau")
    @test all(s -> s.momentum === :gaussian_momentum, problem_from_case(lan, Float64).species)   # schema 1: spread along x only
    @test all(s -> s.momentum === :maxwell_juttner, problem_from_case(lan, Float64; transverse_thermal = true).species)
    obl = load_case("landau-oblique-2d")
    @test only(problem_from_case(obl, Float64).species).momentum === :maxwell_juttner
    @test only(problem_from_case(obl, Float64; momentum = :gaussian_velocity).species).momentum === :gaussian_velocity
    keyed = deepcopy(obl)
    keyed["species"][1]["momentum_distribution"] = "gaussian_momentum"
    @test only(problem_from_case(keyed, Float64).species).momentum === :gaussian_momentum
    @test only(problem_from_case(keyed, Float64; momentum = :maxwell_juttner).species).momentum === :maxwell_juttner
    keyed1 = deepcopy(lan)
    keyed1["species"][1]["momentum_distribution"] = "gaussian_velocity"
    @test problem_from_case(keyed1, Float64).species[1].momentum === :gaussian_velocity
    keyed1["species"][1]["momentum_distribution"] = "maxwell_juttner"                # spread along x only
    @test_throws ArgumentError problem_from_case(keyed1, Float64)
    keyed["species"][1]["momentum_distribution"] = 1
    @test_throws ArgumentError problem_from_case(keyed, Float64)
    keyed["species"][1]["momentum_distribution"] = "juttner"
    @test_throws ArgumentError problem_from_case(keyed, Float64)
end

function body_T539(be)
    for T in TYPES, mode in (Reference(), Fast())
        # #257's plasma with the electron v_th at 0.3: refused in velocity (T-480), loaded under the default
        hot = plasma_step_2d(T; nsteps = 4, vth = 3 * one(T) / 10)
        @test all(s -> s.momentum === :maxwell_juttner, hot.species)
        @test refusal(hot; mode, be) === nothing
        @test loads_and_steps(hot; mode, be)
        hot1 = electrons_1d(T; vth = 4 * one(T) / 10, momentum = :default)
        @test refusal(hot1; mode, be) === nothing
        @test loads_and_steps(hot1; mode, be)
        # a hot plasma drifting at 0.95 c, and a spread along x only of 0.6 c (Gaussian in momentum)
        fast = electrons_2d(T; vth = T(1 // 2), drift = (zero(T), T(95 // 100), zero(T)), momentum = :default)
        @test loads_and_steps(fast; mode, be)
        aniso = SpeciesDescription{T}(:electron, -one(T), one(T), one(T), zero(T), 1, 16, zero3(T), (T(6 // 10), zero(T), zero(T)), 1)
        @test aniso.momentum === :gaussian_momentum
        L = 2 * T(π)
        @test loads_and_steps(EMProblem("aniso-1d", L, 16, 9 * L / 160, 4, (aniso,), 1, 2 * T(π) / L); mode, be)
    end
end

@testset "T-537 a species' momentum distribution: the default rule, the keyword, the case-file key (both schemas), the refusals, and initialize! writes the engine's load, converting only velocity draws, N = 1 and 2 [A2, A8, WP-033]" begin
    body_T537(CPU())
end

@testset "T-539 under the default distributions the hot and fast-drifting plasmas that the velocity draw refuses (#257) load with a finite γ and step, N = 1 and 2, both modes [A2, A8, WP-033]" begin
    body_T539(CPU())
end
