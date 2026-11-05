# T-420: the time of the loaded momenta in `initialize!`. With `momenta_at = :t0` (the default)
# the loaded momenta are u⁰ and are kicked back half a step in E⁰; with `:t_minus_half` they are
# taken as u⁻¹ᐟ², the convention of Smilei and PIConGPU, and kept as loaded. Everything else the
# two starts share bit for bit in Reference mode (Fast's atomic charge makes E⁰ differ in its last
# bits between two loads), so there the `:t_minus_half` state followed by the same kick and the
# kinetic deposit is the default state, bit for bit. The same holds at N = 2.

"The half step back of `initialize!` (the same launch) applied to the momenta of `st`, then the kinetic energy at the kicked momenta."
function kick_back!(st::EMState, prob::EMProblem)
    (; numerics, grid) = st
    g = Int32(guards(grid))
    for s in st.species
        ParticleMeshCore.launcher(kick_kernel!, get_backend(s), numerics.workgroupsize)(
            s, st.fields, numerics.pusher, numerics.shape, g, -s.q / s.m * prob.dt / 4; ndrange = capacity(s))
    end
    ParticleMeshElectromagnetic.kinetic!(st)
    return st
end

"The same at N = 2."
function kick_back_2d!(st::EMState{2}, prob::EMProblem{2})
    (; numerics, grid) = st
    g = Int32(guards(grid))
    for s in st.species
        ParticleMeshCore.launcher(ParticleMeshElectromagnetic.kick_2d_kernel!, get_backend(s), numerics.workgroupsize)(
            s, st.fields, numerics.pusher, numerics.shape, g, -s.q / s.m * prob.dt / 4; ndrange = capacity(s))
    end
    ParticleMeshElectromagnetic.kinetic!(st)
    return st
end

momenta_bits(st) = [bits(a) for s in st.species for a in s.u]

function body_T420(be)
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
    for T in TYPES
        prob = plasma_2d(T; amplitude = T === D3 ? seeded(T, 0.05) : one(T) / 20)
        a, b = (EMState(prob, EMNumerics(); backend = be) for _ in 1:2)
        initialize!(a, prob; momenta_at = :t_minus_half)
        initialize!(b, prob)
        @test momenta_bits(a) != momenta_bits(b)
        @test all(k -> bits(a.fields[k]) == bits(b.fields[k]), keys(a.fields))
        @test state_digest_2d(kick_back_2d!(a, prob)) == state_digest_2d(b)
    end
    st, _, prob, _ = case_state("two-stream", Float64; ppc = 4, t_end = 1, be)
    @test_throws ArgumentError initialize!(st, prob; momenta_at = :t_half)
    prob2 = plasma_2d(Float64)
    @test_throws ArgumentError initialize!(EMState(prob2, EMNumerics(); backend = be), prob2; momenta_at = :t_half)
end

@testset "T-420 initialize! takes the loaded momenta at t = 0 (kicked back half a step) or at −Δt/2 (kept), at N = 1 and 2 [A2, A4, A8, WP-017]" begin
    body_T420(CPU())
end

# T-522: where `initialize!` samples the density profile (#226, item 2). `density_at = :particle`
# (the default) is today's load bit for bit; `:cell_centre` evaluates the profile at the centre of
# each particle's cell, as Smilei and PIConGPU do, so every particle of a cell carries the same
# weight. With the momenta kept as loaded (`:t_minus_half`), positions, momenta and status are the
# same in both, and only the weights, the charge and the field from it differ.

"Whether every particle of a cell carries the same weight (host copies; the cell as a tuple over the axes)."
function weights_per_cell_constant(s)
    w, st = Array(s.w), Array(s.status)
    cells = map(Array, s.cell)
    seen = Dict{Any,eltype(w)}()
    for i in eachindex(w)
        st[i] == ParticleMeshCore.ALIVE || continue
        c = map(a -> a[i], cells)
        bits([get!(seen, c, w[i])]) == bits([w[i]]) || return false
    end
    return true
end

function body_T522(be)
    for name in ("landau", "two-stream"), T in TYPES
        amplitude = T === D3 ? seeded(T, name == "landau" ? 0.01 : 0.001) : nothing
        kw = (; mode = Reference(), ppc = 16, t_end = 1, amplitude, be)
        a, _, _, _ = case_state(name, T; kw...)
        b, _, _, _ = case_state(name, T; density_at = :particle, kw...)
        @test state_digest(a) == state_digest(b)          # the default is :particle, bit for bit
        p, _, _, _ = case_state(name, T; momenta_at = :t_minus_half, kw...)
        c, pc, _, _ = case_state(name, T; momenta_at = :t_minus_half, density_at = :cell_centre, kw...)
        for (sp, sc) in zip(p.species, c.species)
            @test all(x -> bits(getfield(sp, x)[1]) == bits(getfield(sc, x)[1]), (:cell, :offset))
            @test momenta_bits(p) == momenta_bits(c) && bits(sp.status) == bits(sc.status)
            @test bits(sp.w) != bits(sc.w)
            @test weights_per_cell_constant(sc) && !weights_per_cell_constant(sp)
        end
        @test bits(p.fields.Ex) != bits(c.fields.Ex)
        step!(c, pc, 1)
        @test all(s -> all(isfinite, Array(s.u[1])), c.species)
    end
    for T in TYPES
        prob = plasma_2d(T; amplitude = T === D3 ? seeded(T, 0.05) : one(T) / 20)
        a, b, c = (EMState(prob, EMNumerics(); backend = be) for _ in 1:3)
        initialize!(a, prob; momenta_at = :t_minus_half)
        initialize!(b, prob; momenta_at = :t_minus_half, density_at = :particle)
        initialize!(c, prob; momenta_at = :t_minus_half, density_at = :cell_centre)
        @test state_digest_2d(a) == state_digest_2d(b)
        @test momenta_bits(a) == momenta_bits(c)
        @test all(s -> weights_per_cell_constant(s), c.species) && !all(s -> weights_per_cell_constant(s), a.species)
        @test bits(a.fields.Ex) != bits(c.fields.Ex)
    end
    st, _, prob, _ = case_state("two-stream", Float64; ppc = 4, t_end = 1, be)
    @test_throws ArgumentError initialize!(st, prob; density_at = :cell_center)
    prob2 = plasma_2d(Float64)
    @test_throws ArgumentError initialize!(EMState(prob2, EMNumerics(); backend = be), prob2; density_at = :node)
end

@testset "T-522 initialize! samples the density at each particle (default, bit for bit) or at its cell centre, with positions and momenta unchanged, at N = 1 and 2 [A3, A4, A8]" begin
    body_T522(CPU())
end
