# T-100 to T-106: the SciML harness (WP-008), the module's step as a `DiscreteProblem` solved
# with `FunctionMap`. The harness is an adapter: in `Reference()` mode a solve gives the same
# bits as `initialize!` plus `run!` (the comparisons are on bits, so `-0.0` and the partials of
# duals count). Time counts steps (`t = k` after step `k`, `dt = 1`).

using SciMLBase: SciMLBase, DiscreteProblem, ReturnCode, remake, solve, init, solve!
using RecursiveArrayTools: ArrayPartition
using OrdinaryDiffEqFunctionMap: FunctionMap

const HARNESS_KW = (; ppc = 64, t_end = 1)        # Landau: 107 steps, particles cross cells and the seam

"The problem description and numerics of a case file, with no state allocated."
function case_problem(name, ::Type{T}; mode = Reference(), kw...) where {T}
    case = load_case(name)
    return problem_from_case(case, T; kw...), numerics_from_case(case; mode)
end

"The plain loop: a new state, `initialize!`, then `run!` over `n` steps."
function loop_state(prob, num, n = prob.nsteps)
    st = EMState(prob, num; backend)
    initialize!(st, prob)
    run!(st, params(prob), n)
    return st
end

partition_bits(u::ArrayPartition) = map(bits, u.x)
state_bits(st::EMState) = partition_bits(state_partition(st))
history_bits(st::EMState, n = length(st.hist.K)) = map(h -> bits(view(h, 1:n)), values(st.hist))

"Bytes of the arrays of a partition."
partition_bytes(u::ArrayPartition) = sum(sizeof, u.x)

"T-100 for one element type: a solve over `n` steps against the loop, bitwise."
function harness_equals_loop(::Type{T}; n = nothing) where {T}
    prob, num = case_problem("landau", T; HARNESS_KW...)
    n = n === nothing ? prob.nsteps : n
    dprob = DiscreteProblem(prob, num; backend, nsteps = n)
    u0 = partition_bits(dprob.u0)
    sol = solve(dprob, FunctionMap(); save_everystep = false)
    ref = loop_state(prob, num, n)
    again = solve(dprob, FunctionMap(); save_everystep = false)
    return (; ok = sol.retcode == ReturnCode.Success, t = sol.t[end], n,
            state = partition_bits(sol.u[end]) == state_bits(ref),
            hist = history_bits(harness_state(sol), n) == history_bits(ref, n),
            u0 = partition_bits(dprob.u0) == u0,
            again = partition_bits(again.u[end]) == state_bits(ref) &&
                    history_bits(harness_state(again), n) == history_bits(ref, n))
end

@testset "T-100 harness equals the loop: solve over n steps is bitwise initialize! plus run! [A9, WP-008]" begin
    for n in (nothing, 40)
        r = harness_equals_loop(Float64; n)
        @test r.ok
        @test r.t == r.n
        @test r.state             # particles, fields and K
        @test r.hist              # the diagnostic histories in the cache
        @test r.u0                # solving does not touch the problem's u0
        @test r.again             # a second solve of the same problem gives the same bits
    end
    # the state mapping: zero-copy over the state's arrays, and copyto! back
    prob, num = case_problem("landau", Float64; HARNESS_KW...)
    st = EMState(prob, num; backend)
    initialize!(st, prob)
    u = state_partition(st)
    @test u.x[1] === st.species[1].cell[1] && u.x[end] === st.K
    @test map(eltype, u.x)[1:8] == (Int32, Float64, Float64, Float64, Float64, Float64, UInt8, Int32)
    saved = copy(u)
    run!(st, params(prob), 5)
    copyto!(st, saved)
    @test state_bits(st) == partition_bits(saved)
    @test_throws ArgumentError copyto!(st, ArrayPartition(u.x[1:(end - 1)]...))
    @test_throws ArgumentError DiscreteProblem(prob, num; backend, nsteps = prob.nsteps + 1)
    # every step from step 1: dt ≠ 1 (integer or not) and a tspan not starting at 0 are refused,
    # at construction and, when a solve keyword overrides them, at the map's first call
    for dt in (1 // 2, 2)
        @test_throws ArgumentError DiscreteProblem(prob, num; backend, dt)
        @test_throws ArgumentError solve(DiscreteProblem(prob, num; backend), FunctionMap(); dt)
    end
    dprob = DiscreteProblem(prob, num; backend)
    @test_throws ArgumentError remake_reload(dprob; p = prob, tspan = (5.0, 10.0))
    @test_throws ArgumentError solve(remake(dprob; tspan = (5.0, 10.0)), FunctionMap())
    # nor after a solve that ended at that step: a new solve does not continue the last one
    solve(remake(dprob; tspan = (0.0, 5.0)), FunctionMap())
    @test_throws ArgumentError solve(remake(dprob; tspan = (5.0, 10.0)), FunctionMap())
    # a run continues on its own integrator: init, then single steps, equal the loop
    integ = init(dprob, FunctionMap(); save_everystep = false)
    foreach(_ -> SciMLBase.step!(integ), 1:3)
    @test integ.t == 3
    @test partition_bits(integ.u) == state_bits(loop_state(prob, num, 3))
end

@testset "T-101 remake_reload equals a problem built from scratch for p′, bitwise [A3, A9, WP-008]" begin
    prob, num = case_problem("landau", Float64; HARNESS_KW...)
    dprob = DiscreteProblem(prob, num; backend, save_everystep = false)
    p′, _ = case_problem("landau", Float64; HARNESS_KW..., amplitude = 0.02, seed = 7)
    reloaded = remake_reload(dprob; p = p′)
    fresh = DiscreteProblem(p′, num; backend, save_everystep = false)
    @test reloaded.p === p′
    @test harness_state(reloaded) !== harness_state(dprob)     # its own cache
    @test partition_bits(reloaded.u0) == partition_bits(fresh.u0)
    @test partition_bits(reloaded.u0) != partition_bits(dprob.u0)
    a = solve(reloaded, FunctionMap())
    b = solve(fresh, FunctionMap())
    @test length(a.u) == 2                                     # the problem keyword survived
    @test partition_bits(a.u[end]) == partition_bits(b.u[end])
    @test history_bits(harness_state(a)) == history_bits(harness_state(b))
    @test partition_bits(a.u[end]) == state_bits(loop_state(p′, num))
    # the plain remake keeps u0, and with it the weights loaded from the old parameters (A3)
    plain = remake(dprob; p = p′)
    @test partition_bits(plain.u0) == partition_bits(dprob.u0)
    # the element type follows p′
    p32, _ = case_problem("landau", Float32; HARNESS_KW...)
    @test eltype(remake_reload(dprob; p = p32).u0.x[2]) == Float32
end

@testset "T-102 states saved at saveat match snapshots of run!, bitwise [A9, WP-008]" begin
    prob, num = case_problem("landau", Float64; HARNESS_KW...)
    steps = [0, 1, 17, 50, prob.nsteps]
    sol = solve(DiscreteProblem(prob, num; backend, saveat = steps), FunctionMap())
    @test sol.t == steps
    st = EMState(prob, num; backend)
    initialize!(st, prob)
    p = params(prob)
    done = 0
    for (i, k) in enumerate(steps)
        run!(st, p, k - done; first = done + 1)
        done = k
        @test partition_bits(sol.u[i]) == state_bits(st)
    end
    # saved states are copies of the arrays, with their element types
    @test all(map(!==, sol.u[1].x, sol.u[2].x))
    @test map(eltype, sol.u[1].x) == map(eltype, state_partition(st).x)
end

@testset "T-103 harness overhead per step against run!, reported [A7, WP-008]" begin
    prob, num = case_problem("landau", Float64; ppc = 256, t_end = 1)
    n = prob.nsteps
    dprob = DiscreteProblem(prob, num; backend, save_everystep = false)
    st = EMState(prob, num; backend)
    initialize!(st, prob)
    p = params(prob)
    # per step: the map (copy uprev into u, then step! on u) against step! alone, minima over
    # repetitions, since one step varies by more than the overhead at several threads
    m = dprob.f.f
    u, uprev = copy(dprob.u0), copy(dprob.u0)
    m(u, uprev, prob, 1.0)
    step!(st, p, 1)
    reps = 30
    map_s = minimum(_ -> @elapsed(m(u, uprev, prob, 1.0)), 1:reps)
    step_s = minimum(_ -> @elapsed(step!(st, p, 1)), 1:reps)
    copy_s = minimum(_ -> @elapsed(copyto!(st, uprev)), 1:reps)
    extra_bytes = @allocated(m(u, uprev, prob, 1.0)) - @allocated(step!(st, p, 1))
    # whole runs, for context: n steps through solve against run!
    run!(st, p, n)
    solve(dprob, FunctionMap())
    loop = minimum(_ -> @elapsed(run!(st, p, n)), 1:3)
    harness = minimum(_ -> @elapsed(solve(dprob, FunctionMap())), 1:3)
    particles = sum(capacity, st.species)
    @info "T-103 harness overhead" particles state_bytes = partition_bytes(dprob.u0) map_us = 1e6 * map_s step_us = 1e6 * step_s copy_us = 1e6 * copy_s extra_bytes n loop_s = loop harness_s = harness threads = Threads.nthreads()
    d = get!(MEASURED, "T-103 harness overhead", Dict{String,Float64}())
    d["copy/step"] = copy_s / step_s
    d["map/step"] = map_s / step_s
    d["solve/run!"] = harness / loop
    @test isfinite(map_s) && isfinite(copy_s)
end

@testset "T-104 memory the harness holds with save_everystep = false, reported [A7, WP-008]" begin
    prob, num = case_problem("landau", Float64; HARNESS_KW...)
    dprob = DiscreteProblem(prob, num; backend, save_everystep = false)
    state = partition_bytes(dprob.u0)
    integ = init(dprob, FunctionMap())
    # every partition the integrator, its cache, its solution and the problem hold, counted once
    held = Any[dprob.u0, integ.u, integ.uprev, integ.cache.u, integ.cache.uprev, integ.cache.tmp, integ.sol.u...]
    distinct = IdDict{Any,Nothing}()
    for u in held, a in u.x
        distinct[a] = nothing
    end
    copies_init = sum(sizeof, keys(distinct)) / state
    solve!(integ)
    for u in integ.sol.u, a in u.x
        distinct[a] = nothing
    end
    copies_end = sum(sizeof, keys(distinct)) / state
    cache = harness_state(dprob)
    ws = Base.summarysize(cache.ws)
    hist = Base.summarysize(cache.hist)
    extras = sum(s -> Base.summarysize(s.extras), cache.species)
    total = Base.summarysize(integ) / state
    @info "T-104 harness memory" state_bytes = state copies_init copies_end total_over_state = total workspaces_bytes = ws histories_bytes = hist extras_bytes = extras rho_bytes = sizeof(cache.ρ)
    get!(MEASURED, "T-104 state copies held", Dict{String,Float64}())["init"] = copies_init
    MEASURED["T-104 state copies held"]["end"] = copies_end
    @test copies_init >= 1 && isfinite(copies_end)
end

"""
The Ex field energy at the last step of the T-219 run, through `solve` of a problem built by
`remake_reload` from a Float64 base problem (so a dual α gets a dual cache).
"""
function harness_field_energy(α, mode, seed)
    p, _ = case_problem("landau", typeof(α); mode, seed, amplitude = α, GRADIENT_KW...)
    sol = solve(remake_reload(HARNESS_BASE[]; p), FunctionMap(); save_everystep = false)
    return Array(harness_state(sol).hist.WEx)[end]
end
const HARNESS_BASE = Ref{Any}()

@testset "T-105 ForwardDiff through solve against T-219 and the fd_scan floor [A3, A7, A8, WP-008]" begin
    prob, num = case_problem("landau", Float64; GRADIENT_KW...)
    HARNESS_BASE[] = DiscreteProblem(prob, num; backend)
    seed = 20261031
    α0 = 0.01
    ad = ForwardDiff.derivative(α -> harness_field_energy(α, Reference(), seed), α0)
    ad_loop = ForwardDiff.derivative(α -> landau_field_energy(α, Reference(), seed), α0)
    hs = α0 .* 2.0 .^ (-2:-2:-20)
    r = fd_scan(harness_field_energy, α0, hs; seed, ad)
    ratio = r.rel[r.istar] / r.floor_rel
    @info "T-105 gradient through solve" ad ad_loop fd = r.fd[r.istar] h = hs[r.istar] rel = r.rel[r.istar] floor = r.floor_rel ratio
    get!(MEASURED, "T-105 rel/floor", Dict{String,Float64}())["Float64"] = ratio
    @test isfinite(ad) && ad != 0
    @test ad == ad_loop                    # the harness is bitwise the loop, through the duals too
    @test r.rel[r.istar] <= emC("T-105") * r.floor_rel
end

@testset "T-106 T-100 for every element type of the A8 matrix [A8, WP-008]" begin
    for T in TYPES
        r = harness_equals_loop(T)
        @test r.ok
        @test r.state
        @test r.hist
        @test r.u0
        @test r.again
    end
end
