# T-360: the time of the loaded momenta in `initialize!`. With `momenta_at = :t0` (the default)
# the loaded momenta are u⁰ and are kicked back half a step in E⁰; with `:t_minus_half` they are
# taken as u⁻¹ᐟ², the convention of Smilei and PIConGPU, and kept as loaded. Everything else the
# two starts share bit for bit in Reference mode (Fast's atomic charge makes E⁰ differ in its last
# bits between two loads), so there the `:t_minus_half` state followed by the same kick and the
# kinetic deposit is the default state, bit for bit.

"The half step back of `initialize!` applied to the momenta of `st`, then the kinetic energy at the kicked momenta."
function kick_back!(st::EMState, prob::EMProblem)
    (; numerics, grid) = st
    g = Int32(guards(grid))
    for s in st.species
        kick_kernel!(get_backend(s))(s, st.fields, numerics.pusher, numerics.shape, g, -s.q / s.m * prob.dt / 4;
                                     ndrange = capacity(s))
    end
    ParticleMeshElectromagnetic.kinetic!(st)
    return st
end

momenta_bits(st) = [bits(a) for s in st.species for a in s.u]

function body_T360(be)
    for name in ("landau", "two-stream"), T in TYPES
        amplitude = T === D3 ? seeded(T, name == "landau" ? 0.01 : 0.001) : nothing
        kw = (; mode = Reference(), ppc = 16, t_end = 1, amplitude, be)
        a, p, prob, _ = case_state(name, T; momenta_at = :t_minus_half, kw...)
        b, _, _, _ = case_state(name, T; kw...)
        @test momenta_bits(a) != momenta_bits(b)          # E⁰ ≠ 0: the kick changes the momenta
        for (sa, sb) in zip(a.species, b.species)
            @test all(x -> bits(getfield(sa, x)[1]) == bits(getfield(sb, x)[1]), (:cell, :offset))
            @test bits(sa.w) == bits(sb.w) && bits(sa.status) == bits(sb.status)
        end
        @test all(k -> bits(a.fields[k]) == bits(b.fields[k]), keys(a.fields))
        @test state_digest(kick_back!(a, prob)) == state_digest(b)
        @test bits(a.K) == bits(b.K)
        step!(a, p, 1)                                    # the kept momenta step like the kicked ones
        @test all(s -> all(isfinite, Array(s.u[1])), a.species)
    end
    st, _, prob, _ = case_state("two-stream", Float64; ppc = 4, t_end = 1, be)
    @test_throws ArgumentError initialize!(st, prob; momenta_at = :t_half)
end

@testset "T-360 initialize! takes the loaded momenta at t = 0 (kicked back half a step) or at −Δt/2 (kept) [A2, A4, A8, WP-017]" begin
    body_T360(CPU())
end
