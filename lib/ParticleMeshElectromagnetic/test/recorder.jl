# WP-062, the recorder's cadence and selection: T-810 to T-812 (T-813 reruns T-810 and T-811 on
# CUDA, gpu.jl). The checks are exact: what a recorder with options records has the bits of the
# same entries of the recorder that records every step and every group, and the run's evolved
# arrays do not depend on the observer.

"The problems of the recorder's tests: a 1D plasma, a periodic 2D plasma with current, a 2D moving window that shifts."
recorder_problems(::Type{T}) where {T} =
    ((beams_1d(T; nsteps = 13), EMNumerics()),
     (plasma_step_2d(T; nsteps = 13), EMNumerics(; current = Esirkepov())),
     (window_plasma(T; nsteps = 13, thermal = (1 // 20, 1 // 25, 1 // 50)), EMNumerics(; headroom = 1 / 4)))

function recorder_run(prob, num, be; observer = :default, kw...)
    st = EMState(prob, num; backend = be, kw...)
    initialize!(st, prob)
    run!(st, params(prob), prob.nsteps; observer = observer === :default ? st.recorder : observer)
    return st
end

same_evolved(a, b) = all(map((x, y) -> same_bits(Array(x), Array(y)), evolved(a), evolved(b)))

const RECORDER_OPTIONS = ((every = 1, record = :kinetic), (every = 1, record = (:field_energy,)), (every = 1, record = (:modes, :kinetic)),
                          (every = 3, record = RECORD_GROUPS), (every = 4, record = (:kinetic, :field_energy)), (every = 5, record = :modes),
                          (every = 13, record = RECORD_GROUPS), (every = 20, record = :kinetic), (every = 2, record = ()))

function body_T810(be)
    for T in (Float32, Float64, D3), (prob, num) in recorder_problems(T)
        full = recorder_run(prob, num, be)
        hf = histories(full)
        N = ndims(full)
        @test keys(hf) == (N == 1 ? ParticleMeshElectromagnetic.HISTORY_KEYS : ParticleMeshElectromagnetic.HISTORY_KEYS_2D) && recorded_steps(full) == 1:1:13
        @test any(!iszero, hf.K) && any(!iszero, hf.WEx) && any(!iszero, hf.Ex_re)
        N == 2 && prob.window !== nothing && @test ParticleMeshElectromagnetic.window_shifts(full, prob.nsteps) > 2      # the window case shifts
        for opt in RECORDER_OPTIONS
            st = recorder_run(prob, num, be; opt...)
            h = histories(st)
            groups = opt.record isa Symbol ? (opt.record,) : opt.record
            want = filter(keys(hf)) do k
                k in ParticleMeshElectromagnetic.ENERGY_KEYS ? :field_energy in groups : k === :K ? :kinetic in groups : :modes in groups
            end
            steps = 1:opt.every:prob.nsteps
            @test keys(h) == want
            @test recorded_steps(st) == steps && length(st.recorder) == prob.nsteps
            @test all(k -> length(h[k]) == length(steps), keys(h))
            @test all(k -> same_bits(h[k], hf[k][steps]), keys(h))        # the entries of the full recorder, bit for bit
            @test same_evolved(st, full)                                  # the run does not depend on what is recorded
        end
    end
    return nothing
end

@testset "T-810 a recorder with a cadence and a selection holds the bits of the full recorder's entries, and the run is the same [A4, A7, A8, WP-062]" begin
    body_T810(CPU())
end

"The kinetic energy an observer reads in `state.K`, summed as the recorder sums it."
kinetic_seen(st) = foldl(+, Array(st.K); init = zero(eltype(st))) * prod(cellsize(st.grid))

function body_T811(be)
    for T in (Float32, Float64), (prob, num) in recorder_problems(T)
        p = params(prob)
        full = recorder_run(prob, num, be)
        hf = histories(full)
        # a function as observer reads the cache of its own step at every step
        seen = T[]
        fn = recorder_run(prob, num, be; observer = (s, k) -> push!(seen, kinetic_seen(s)))
        @test same_bits(seen, hf.K) && same_evolved(fn, full)
        # a function next to a sparse recorder: the function still reads a current cache
        empty!(seen)
        st = EMState(prob, num; backend = be, every = 4, record = :kinetic)
        initialize!(st, prob)
        run!(st, p, prob.nsteps; observer = (st.recorder, (s, k) -> push!(seen, kinetic_seen(s))))
        @test same_bits(seen, hf.K) && same_bits(histories(st).K, hf.K[1:4:end])
        # no observer: the cache stays current, as a function's does
        none = recorder_run(prob, num, be; observer = nothing)
        @test same_bits(Array(none.K), Array(full.K)) && same_evolved(none, full)
        # a recorder of nothing: the step leaves the cache of `initialize!`, and `refresh_caches!` gives that of the run's end
        silent = EMState(prob, num; backend = be, record = ())
        initialize!(silent, prob)
        K0 = Array(silent.K)
        run!(silent, p, prob.nsteps)
        @test same_bits(Array(silent.K), K0) && !same_bits(K0, Array(full.K)) && same_evolved(silent, full)
        ParticleMeshElectromagnetic.refresh_caches!(silent)
        @test same_bits(Array(silent.K), Array(full.K))
        # a sparse recorder: the cache is computed by the step before a recorded step only
        sparse = EMState(prob, num; backend = be, every = 4)
        initialize!(sparse, prob)
        for k in 1:prob.nsteps
            before = Array(sparse.K)
            step!(sparse, p, k)
            fresh = Array(ParticleMeshElectromagnetic.state_view(sparse, evolved(sparse)).K)       # a new buffer, computed from the particles
            @test same_bits(Array(sparse.K), k % 4 == 0 ? fresh : before)
            k % 4 == 0 && @test !same_bits(fresh, before)
        end
        @test same_bits(histories(sparse).K, hf.K[1:4:end])
    end
    return nothing
end

@testset "T-811 the kinetic-energy cache is computed where the next step's observer reads it, and is then the full run's [A4, A7, WP-062]" begin
    body_T811(CPU())
    prob, num = recorder_problems(Float64)[2]
    st = EMState(prob, num)
    full, sparse, nokin = st.recorder, HistoryRecorder(prob, st; every = 3), HistoryRecorder(prob, st; record = (:modes, :field_energy))
    none = HistoryRecorder(prob, st; record = ())
    @test all(k -> reads_kinetic(full, k), 1:20) && all(k -> reads_kinetic(nothing, k), 1:20)
    @test !any(k -> reads_kinetic(nokin, k), 1:20) && !any(k -> reads_kinetic(none, k), 1:20)
    @test [k for k in 1:10 if reads_kinetic(sparse, k)] == [1, 4, 7, 10]
    @test [k for k in 1:10 if reads_kinetic((nokin, sparse, none), k)] == [1, 4, 7, 10] && !reads_kinetic((), 1) && reads_kinetic((none, nothing), 2)
    @test reads_kinetic((s, k) -> nothing, 2) && reads_kinetic((nokin, identity), 2)
end

@testset "T-812 a recorder refuses a cadence, a group or a step it does not have, and its copies keep its options [A4, WP-062]" begin
    for (prob, num) in recorder_problems(Float64)
        st = EMState(prob, num; every = 3, record = :kinetic)
        initialize!(st, prob)
        r = st.recorder
        @test r.every == 3 && ParticleMeshElectromagnetic.recorded_groups(r) == (:kinetic,) && keys(r.hist) == (:K,) && length(r.hist.K) == 5
        @test_throws ArgumentError EMState(prob, num; every = 0)
        @test_throws ArgumentError EMState(prob, num; record = :energy)
        @test_throws ArgumentError HistoryRecorder(prob, st; record = (:kinetic, :fields))
        @test_throws ArgumentError HistoryRecorder(prob, st; every = -2)
        @test_throws ArgumentError observe!(r, st, prob.nsteps + 1)          # a recorded step past the length (14 = 1 + 13)
        @test_throws ArgumentError observe!(r, st, prob.nsteps + 2)          # a step it would not record, past the length
        @test_throws ArgumentError observe!(r, st, 0)
        observe!(r, st, 2)                                                   # not a recorded step: nothing written
        @test all(iszero, Array(r.hist.K))
        observe!(r, st, 4)
        @test findall(!iszero, Array(r.hist.K)) == [2]
        e = ParticleMeshElectromagnetic.empty_recorder(r)
        @test e.every == 3 && e.nsteps == r.nsteps && keys(e.hist) == (:K,) && all(iszero, Array(e.hist.K)) && e.hist.K !== r.hist.K
        v = ParticleMeshElectromagnetic.state_view(st, evolved(st))
        @test v.recorder.every == 3 && keys(v.recorder.hist) == (:K,) && recorded_steps(v) == 1:3:13
        none = HistoryRecorder(prob, st; record = ())
        @test isempty(none.hist) && histories(none) == NamedTuple() && length(none) == prob.nsteps
        observe!(none, st, 1)
        @test_throws ArgumentError observe!(none, st, prob.nsteps + 1)
        short = HistoryRecorder(prob, st; nsteps = 7, every = 3)
        @test recorded_steps(short) == 1:3:7 && length(short.hist.K) == 3 && length(HistoryRecorder(prob, st; nsteps = 0).hist.K) == 0
    end
end
