# T-480, T-481: a Gaussian-velocity load whose drawn speeds reach c is refused (#257). With
# `momentum = :gaussian_velocity` the loader draws each velocity component as drift + thermal·z
# (z standard normal, in velocity), and the module turns velocities into momenta with
# γ = 1/√(1 − v²), which is not finite for |v| ≥ 1; `initialize!` refuses such a load with an
# ArgumentError naming the species and the largest drawn |v|, before any step. The default
# distributions draw momenta and never reach c (WP-033, T-539); a cold species keeps the
# velocity draw, so its drift is checked here too.

using ParticleMeshElectromagnetic: check_subluminal

"A cold or thermal electron species at N = 1 on a 2π box of `cells` cells (zero perturbation), drawn in velocity unless `momentum` says otherwise."
function electrons_1d(::Type{T}; drift = zero3(T), vth = zero(T), cells = 16, ppc = 16, seed = 1, momentum = :gaussian_velocity) where {T}
    L = 2 * T(π)
    sp = (SpeciesDescription{T}(:electron, -one(T), one(T), one(T), zero(T), 1, ppc, drift, (T(vth), T(vth), T(vth)), 1; momentum),)
    return EMProblem("electrons-1d", L, cells, 9 * L / cells / 10, 4, sp, seed, 2 * T(π) / L)
end

"The same at N = 2: a 2π × 2π box of `cells` cells."
function electrons_2d(::Type{T}; drift = zero3(T), vth = zero(T), cells = (6, 6), ppc = (2, 2), seed = 1, momentum = :gaussian_velocity) where {T}
    L = (2 * T(π), 2 * T(π))
    dt = 9 * inv(sqrt(sum(d -> (cells[d] / L[d])^2, 1:2))) / 10
    sp = (SpeciesDescription{T}(:electron, -one(T), one(T), one(T), zero(T), (1, 0), ppc, drift, (T(vth), T(vth), T(vth)), 1; momentum),)
    return EMProblem("electrons-2d", L, cells, T(dt), 4, sp, seed, (1, 0))
end

"The largest |v|² the loader draws for species `d` of `prob`, recomputed on the host from a fresh load (§0.4 allows host reads in tests)."
function drawn_v2_max(prob, d; be)
    T = typeof(d.charge)
    N = ndims(prob)
    grid = EMState(prob, EMNumerics(); backend = be).grid
    s = Species(be, T, prod(prob.cells) * prod(d.ppc); q = d.charge, m = d.mass, N)
    k = N == 1 ? (d.density, d.amplitude, 2 * T(π) * d.mode / prob.length) :
        (d.density, d.amplitude, ntuple(a -> 2 * T(π) * d.mode[a] / prob.length[a], 2)...)
    load_quiet!(s, grid, ParticleMeshElectromagnetic.CosineProfile(), k, d.ppc; vdrift = d.drift, vth = d.thermal,
                seed = prob.seed, stream = d.stream)
    u1, u2, u3 = Array(s.u[1]), Array(s.u[2]), Array(s.u[3])
    return maximum(i -> u1[i] * u1[i] + u2[i] * u2[i] + u3[i] * u3[i], eachindex(u1))
end

"The message of the ArgumentError that `initialize!` of `prob` in `mode` throws, or `nothing` when it throws none."
function refusal(prob; mode, be)
    st = EMState(prob, EMNumerics(; mode); backend = be)
    try
        initialize!(st, prob)
    catch err
        err isa ArgumentError && return err.msg
        rethrow()
    end
    return nothing
end

"`initialize!` and one step of `prob` in `mode` leave every momentum finite."
function loads_and_steps(prob; mode, be)
    st = EMState(prob, EMNumerics(; mode); backend = be)
    initialize!(st, prob)
    step!(st, params(prob), 1)
    return all(s -> all(a -> all(isfinite, Array(a)), s.u), st.species)
end

function body_T480(be)
    for T in TYPES, mode in (Reference(), Fast())
        # #257's reproduction: the WP-023 plasma with the electron v_th raised from 0.1 to 0.3
        hot = plasma_step_2d(T; nsteps = 4, vth = 3 * one(T) / 10, momentum = :gaussian_velocity)
        v2 = drawn_v2_max(hot, hot.species[1]; be)
        @test v2 >= one(T)                                          # the case does draw past c
        @test drawn_v2_max(hot, hot.species[2]; be) < one(T)        # the positive species does not
        msg = refusal(hot; mode, be)
        @test msg !== nothing
        if msg !== nothing
            @test occursin("species electron", msg) && !occursin("positive", msg)
            @test occursin("largest drawn |v| is $(sqrt(v2))", msg)
            @test occursin("drawn in velocity", msg)
        end
        @test loads_and_steps(plasma_step_2d(T; nsteps = 4, momentum = :gaussian_velocity); mode, be)  # the ordinary case still loads
        # the same at N = 1
        hot1 = electrons_1d(T; vth = 4 * one(T) / 10)
        v21 = drawn_v2_max(hot1, hot1.species[1]; be)
        @test v21 >= one(T)
        msg1 = refusal(hot1; mode, be)
        @test msg1 !== nothing && occursin("species electron", msg1) && occursin("largest drawn |v| is $(sqrt(v21))", msg1)
        @test loads_and_steps(electrons_1d(T; vth = one(T) / 10); mode, be)
    end
end

function body_T481(be)
    below = T -> T(prevfloat(one(Float32)))         # |v|² = 1 − 2⁻²³ (rounded): γ finite, about 2900
    for T in TYPES
        z = zero(T)
        for d in 1:3                                 # a cold drift at exactly c along each axis: |v|² = 1 exactly
            c = ntuple(a -> a == d ? one(T) : z, 3)
            @test refusal(electrons_1d(T; drift = c); mode = Reference(), be) !== nothing
            @test refusal(electrons_2d(T; drift = c); mode = Reference(), be) !== nothing
            b = ntuple(a -> a == d ? below(T) : z, 3)
            @test loads_and_steps(electrons_1d(T; drift = b); mode = Reference(), be)
            @test loads_and_steps(electrons_2d(T; drift = b); mode = Reference(), be)
        end
        @test refusal(electrons_2d(T; vth = T(NaN)); mode = Reference(), be) !== nothing   # a NaN velocity too
        # only the alive slots count: free slots past the load holding |v| = 2 are not refused
        prob = electrons_2d(T; vth = one(T) / 10)
        d = prob.species[1]
        grid = EMState(prob, EMNumerics(); backend = be).grid
        s = Species(be, T, prod(prob.cells) * prod(d.ppc) + 3; q = d.charge, m = d.mass, N = 2)
        load_quiet!(s, grid, ParticleMeshElectromagnetic.CosineProfile(), (d.density, d.amplitude, z, z), d.ppc;
                    vth = d.thermal, seed = 1, stream = 1)
        s.u[1] .= ifelse.(s.status .== ParticleMeshCore.ALIVE, s.u[1], 2 * one(T))
        @test check_subluminal(s, d) === s
        s.status .= ParticleMeshCore.ALIVE                     # the same slots alive: refused
        @test_throws ArgumentError check_subluminal(s, d)
        # a NaN in one slot only, the others subluminal: the reduction's `max` must carry it through
        # (for the Dual element type too, where `max` compares values; the all-NaN load above would pass either way)
        s2 = Species(be, T, prod(prob.cells) * prod(d.ppc); q = d.charge, m = d.mass, N = 2)
        load_quiet!(s2, grid, ParticleMeshElectromagnetic.CosineProfile(), (d.density, d.amplitude, z, z), d.ppc;
                    vth = d.thermal, seed = 1, stream = 1)
        @test check_subluminal(s2, d) === s2
        for slot in (1, 7, length(s2.w))
            u1 = Array(s2.u[1])
            u1[slot] = T(NaN)
            copyto!(s2.u[1], u1)
            @test_throws ArgumentError check_subluminal(s2, d)
            u1[slot] = z
            copyto!(s2.u[1], u1)
        end
    end
end

@testset "T-480 initialize! refuses a Gaussian-velocity load whose drawn speeds reach c with an ArgumentError naming the species and the largest |v|, and the ordinary loads still step, at N = 1 and 2 [A2, A8]" begin
    body_T480(CPU())
end

@testset "T-481 the refusal of superluminal loads is exact at |v| = c on every axis, refuses NaN velocities and counts only alive slots [A2, A8]" begin
    body_T481(CPU())
end
