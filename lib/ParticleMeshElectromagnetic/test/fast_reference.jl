# T-290 (WP-037): the module's two deposition modes give the same histories to roundoff. The
# engine checks Fast() against Reference() per call (T-110 to T-113, T-265, T-300); this checks
# that the module's step uses both the same way, over a short run of each 1D case. Fast() sums in
# the order of its atomics, so the two runs differ by roundoff from the first deposit on, and the
# instabilities amplify it: the runs stop at t = 1 (93 to 131 steps), before that growth shows.
# Each history is compared relative to its own largest value, in units of eps(T)·steps.

"The histories T-290 compares in a case: the field energies that are not zero and the kinetic energy."
fast_reference_keys(name) = name == "weibel" ? (:WEy, :WBz, :K) : (:WEx, :K)

function body_T290(be)
    C = emC("T-290")
    for name in ("landau", "two-stream", "weibel"), T in (Float32, Float64)
        runs = map((Reference(), Fast())) do mode
            st, p, prob, _ = case_state(name, T; mode, ppc = 64, t_end = 1, be)
            run!(st, p, prob.nsteps)
            (; h = histories(st), nsteps = prob.nsteps)
        end
        r, f = runs
        worst = 0.0
        for key in fast_reference_keys(name)
            a, b = getfield(f.h, key), getfield(r.h, key)
            @test all(isfinite, a) && maximum(abs, b) > 0
            worst = max(worst, Float64(maximum(abs, a .- b) / (eps(T) * maximum(abs, b) * r.nsteps)))
        end
        record!("T-290 $name", T, worst, be)
        @test worst <= C
    end
end

@testset "T-290 Fast and Reference histories of the module agree to roundoff [A4, A7, WP-014, WP-037]" begin
    body_T290(CPU())
end
