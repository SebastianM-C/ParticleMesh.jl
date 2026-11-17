# WP-042, the module at N = 3 outside the push and the current: T-880 to T-883. The field layout,
# the initial field of a loaded charge (the engine's 3D quiet load, sort, deposition, fold and
# periodic solve in one chain, against its closed form), the recorder's diagnostics, and the
# state's methods that are written over N. The Yee solver at N = 3 is in yee_nd.jl (T-320, T-322,
# T-323, T-327 to T-329). After yee_nd.jl, whose BigFloat helpers and `random_fields` it uses.

const PME = ParticleMeshElectromagnetic

"The grid of the 3D module tests: 12 × 8 × 10 cells of 1/4 × 3/8 × 1/2, with the guards of the module's step."
em3_grid(::Type{T}, shape = Quadratic; n = (12, 8, 10)) where {T} =
    PeriodicGrid(n, (basetype(T)(1 // 4), basetype(T)(3 // 8), basetype(T)(1 // 2)), em_guard_width(shape))

"The shifts, in cells, of each component's nodes in the Yee cell (written out, not read from `field_layout`)."
const YEE_SHIFTS_3D = (Ex = (1 // 2, 0, 0), Ey = (0, 1 // 2, 0), Ez = (0, 0, 1 // 2),
                       Bx = (0, 1 // 2, 1 // 2), By = (1 // 2, 0, 1 // 2), Bz = (1 // 2, 1 // 2, 0))

"The cosine density of the 3D load: `p[1]·(1 + p[2]·cos(p[3]·x₁ + p[4]·x₂ + p[5]·x₃))`."
struct Cosine3 end
@inline (::Cosine3)(x, p) = p[1] * (one(p[1]) + p[2] * cos(p[3] * x[1] + p[4] * x[2] + p[5] * x[3]))

const EM3_MODE = (1, 2, -1)
const EM3_PPC = (2, 1, 3)
const EM3_AMPLITUDE = 1 // 8
em3_params(::Type{T}, grid) where {T} =
    (seeded(T, 1), T(EM3_AMPLITUDE), ntuple(d -> T(2π * EM3_MODE[d] / (ncells(grid)[d] * cellsize(grid)[d])), 3)...)

"A species of charge −1 on `grid` loaded with the cosine density, momenta from a Gaussian in `u`."
function em3_species(::Type{T}, grid; q = -1, extra = 5) where {T}
    s = Species(backend, T, prod(ncells(grid)) * prod(EM3_PPC) + extra; q, m = 1, N = 3)
    dist = GaussianMomentum(T.((1 // 10, 0, -1 // 5)), T.((1 // 4, 1 // 3, 1 // 5)))
    return load_quiet!(s, grid, Cosine3(), em3_params(T, grid), EM3_PPC, dist; seed = 7, stream = 1)
end

"The folded charge of `s` on ρ's nodes, deposited in `mode`."
function em3_charge(mode, s, shape, grid)
    T = eltype(s)
    ws = Workspace(s, grid)
    effective_mode(mode, T) isa Reference && sort_by_cell!(s, grid, ws)
    ρ = deposit!(mode, field(backend, T, grid), s, shape, PME.rho_stagger(Val(3)), grid, ws)
    return fold_guards!(ρ, grid)
end

"The B-spline of order `P` at the distance `s` (in cells) from its centre."
bspline(P, s) = P == 1 ? max(zero(s), 1 - abs(s)) : abs(s) <= 1 // 2 ? 3 // 4 - s^2 : abs(s) <= 3 // 2 ? (3 // 2 - abs(s))^2 / 2 : zero(s)

"""
    lattice_response(P, ppc, m, n) -> BigFloat

What one axis of the quiet lattice (`ppc` particles per cell at the offsets (k − 1/2)/ppc) with a
shape of order `P` makes of the mode `m` of `n` cells: the charge at a node of ρ is
Σ_p w_p·W(s_p) over the particles at the distances s_p = (j − 1/2)/ppc, which for the weights
cos(2π m x/n)/ppc is R·cos(2π m x_node/n) with R = Σ_j W(s_j)·cos(2π m s_j/n)/ppc (the sine terms
cancel, the lattice being symmetric about the node).
"""
lattice_response(P, ppc, m, n) =
    sum(big(bspline(P, (2j - 1) // (2ppc))) * cospi(BigFloat(2 * m * (2j - 1) // (2ppc * n))) for j in (-2ppc + 1):(2ppc)) / ppc

"`2 sin(π m/n)/Δ` in BigFloat, one axis of the Yee-consistent symbol."
symbol_big(m, n, Δ) = 2 * sinpi(BigFloat(m // n)) / BigFloat(Δ)

"""
The closed form of the initial field of the loaded cosine charge, per unit `q·n₀`: the amplitude
`A = a·Π_d R_d` of the deposited charge's mode and the three components
`E_d[j] = A·S_d/K²·sin(θ_j + π m_d/n_d)`, `θ_j = 2π Σ_d m_d(j_d − 1)/n_d`, as Float64 interiors.
"""
function initial_field_reference(shape, grid)
    n, h = grid.n, grid.dx
    A = big(EM3_AMPLITUDE) * prod(d -> lattice_response(order(shape), EM3_PPC[d], EM3_MODE[d], n[d]), 1:3)
    S = ntuple(d -> symbol_big(EM3_MODE[d], n[d], h[d]), 3)
    K2 = sum(abs2, S)
    θ(j) = 2 * sum(d -> EM3_MODE[d] * (j[d] - 1) // n[d], 1:3)                       # θ/π, exact
    E = ntuple(d -> [Float64(A * S[d] / K2 * sinpi(BigFloat(θ(j) + EM3_MODE[d] // n[d]))) for j in CartesianIndices(n)], 3)
    ρ = [Float64(1 + A * cospi(BigFloat(θ(j)))) for j in CartesianIndices(n)]
    return (; A, S, K2, E, ρ)
end

"The backward-difference divergence of (Ex, Ey, Ez) minus (ρ − mean ρ) at every interior node, periodically."
function gauss_residual_3d(f, ρ, grid)
    n, h = grid.n, grid.dx
    E = map(a -> collect(interior_view(a, grid)), (f.Ex, f.Ey, f.Ez))
    ρi = collect(interior_view(ρ, grid))
    ρ̄ = sum(ρi) / prod(n)
    return [sum(d -> (E[d][j] - E[d][CartesianIndex(ntuple(e -> e == d ? mod1(j[e] - 1, n[e]) : j[e], 3))]) / h[d], 1:3) - (ρi[j] - ρ̄)
            for j in CartesianIndices(ρi)]
end

"`a` is, guards included, the periodic image of its interior, bitwise."
is_periodic_image(a, grid) = same_bits(a, fill_guards!(copy(a), grid))

@testset "T-880 the field layout at N = 3 is the Yee cell [A2, WP-042]" begin
    layout = field_layout(Val(3))
    @test keys(layout) == (:Ex, :Ey, :Ez, :Bx, :By, :Bz, :Jx, :Jy, :Jz)
    for (key, shift) in pairs(YEE_SHIFTS_3D)
        @test getfield(layout, key) === map(x -> x == 0 ? Node() : Center(), shift)
    end
    @test (layout.Jx, layout.Jy, layout.Jz) === (layout.Ex, layout.Ey, layout.Ez)
    @test PME.rho_stagger(Val(3)) === (Node(), Node(), Node())
    for T in TYPES
        grid = em3_grid(T)
        f = em_fields(backend, T, grid)
        @test keys(f) == keys(layout)
        @test all(a -> size(a) == ncells(grid) .+ 2guards(grid) && eltype(a) === T && all(iszero, a), f)
    end
end

@testset "T-881 the 3D initial field of a loaded cosine charge: Gauss's law at every node and the closed form [A2, A3, A7, A8, WP-042]" begin
    C, Cρ, CE = emC("T-881"), emC("T-881", "C_charge"), emC("T-881", "C_field")
    for T in TYPES, shape in SHAPES, mode in (Reference(), Fast())
        ε = base_eps(T)
        grid = em3_grid(T, shape)
        n = grid.n
        s = em3_species(T, grid)
        ρ = em3_charge(mode, s, shape, grid)
        f = em_fields(backend, T, grid)
        φ = field(backend, T, grid)
        PME.initial_field!(f, φ, ρ, grid, SpectralWorkspace(backend, T, grid))
        ρmax = maximum(absmax, interior_view(ρ, grid))
        φmax = maximum(absmax, interior_view(φ, grid))
        L = sum(4 ./ Float64.(grid.dx) .^ 2)
        r = maximum(absmax, gauss_residual_3d(f, ρ, grid)) / (ε * sum(n) * (ρmax + L * φmax))
        record!("T-881", T, r)
        @test r <= C
        # the deposited charge and the field against the closed form; q·n₀ = −(1 + ε-parts for duals)
        ref = initial_field_reference(shape, grid)
        qn = -1 .* Float64.(components(seeded(T, 1)))
        ρi = interior_view(ρ, grid)
        rρ = maximum(maximum(abs.(Float64.(components(ρi[j])) .- qn .* ref.ρ[j])) for j in eachindex(ref.ρ)) / (ε * maximum(abs, qn))
        record!("T-881 charge", T, rρ)
        @test rρ <= Cρ
        Emax = maximum(E -> maximum(abs, E), ref.E)
        for (a, E) in zip((f.Ex, f.Ey, f.Ez), ref.E)
            ai = interior_view(a, grid)
            rE = maximum(maximum(abs.(Float64.(components(ai[j])) .- qn .* E[j])) for j in eachindex(E)) / (ε * sum(n) * maximum(abs, qn) * Emax)
            record!("T-881 field", T, rE)
            @test rE <= CE
            @test is_periodic_image(a, grid)
        end
        @test Emax > 1e-3                                             # the reference is a field, not zero
        # nothing else is written
        @test all(a -> all_plus_zero(a), (f.Bx, f.By, f.Bz, f.Jx, f.Jy, f.Jz))
        T === D3 && @test any(x -> !iszero(partials(x)), f.Ez)
    end
end

"`Σ over the interior of a·e^{−ik·x}` at the component's Yee positions, times 2/prod(n), in BigFloat: (re, im)."
function fourier_big(a, grid, kmode, shift)
    n = grid.n
    v = interior_view(a, grid)
    c = sum(big(Float64(value_of(v[j]))) * cispi(-2 * big(sum(d -> kmode[d] * (j[d] - 1 + shift[d]) // n[d], 1:3))) for j in CartesianIndices(v))
    return 2 * real(c) / prod(n), 2 * imag(c) / prod(n)
end

"The host reference of one recorded entry: every history key of a 3D recorder, in BigFloat, from the fields' values."
function recorder_reference(f, K, grid, box, kmode)
    dV = prod(big.(Float64.(grid.dx)))
    out = Dict{Symbol,BigFloat}()
    for key in (:Ex, :Ey, :Ez, :Bx, :By, :Bz)
        v = interior_view(getfield(f, key), grid)
        out[Symbol(:W, key)] = sum(x -> big(Float64(value_of(x)))^2, v) * dV / 2
        out[Symbol(key, :_re)], out[Symbol(key, :_im)] = fourier_big(getfield(f, key), grid, kmode, getfield(YEE_SHIFTS_3D, key))
    end
    out[:K] = sum(x -> big(Float64(value_of(x))), K; init = big(0)) * dV
    k = ntuple(d -> big(kmode[d]) / big(Float64(box[d])), 3)
    k̂ = k ./ sqrt(sum(abs2, k))
    out[:El_re] = sum(d -> k̂[d] * out[Symbol((:Ex, :Ey, :Ez)[d], :_re)], 1:3)
    out[:El_im] = sum(d -> k̂[d] * out[Symbol((:Ex, :Ey, :Ez)[d], :_im)], 1:3)
    return out
end

const EM3_KMODE = (1, -2, 3)
em3_box(grid) = ncells(grid) .* cellsize(grid)

@testset "T-882 the 3D recorder: energies, kinetic energy and the mode's amplitudes at each component's own stagger [A7, A8, WP-042]" begin
    C, Cm = emC("T-882"), emC("T-882", "C_modes")
    @test PME.HISTORY_KEYS_3D == (:WEx, :WEy, :WEz, :WBx, :WBy, :WBz, :K, :Ex_re, :Ex_im, :Ey_re, :Ey_im, :Ez_re, :Ez_im,
                                  :Bx_re, :Bx_im, :By_re, :By_im, :Bz_re, :Bz_im, :El_re, :El_im)
    for T in TYPES
        ε = base_eps(T)
        grid = em3_grid(T; n = (12, 10, 8))                        # a box of three different lengths, (3, 15/4, 4)
        n, box = ncells(grid), em3_box(grid)
        f = random_fields(T, grid; seed = 8820)
        K = [seeded(T, 0.3 + 0.1i) for i in 1:7]
        r = HistoryRecorder(backend, T, grid, box, EM3_KMODE, 5)
        @test r isa HistoryRecorder{3} && length(r) == 5
        PME.record_entry!(r, f, K, grid, 2)
        h = histories(r)
        @test keys(h) == PME.HISTORY_KEYS_3D && all(a -> length(a) == 5, h)
        @test all(a -> all(iszero, a[[1, 3, 4, 5]]), h)                # the other entries are not written
        ref = recorder_reference(f, K, grid, box, EM3_KMODE)
        mean_abs = sum(key -> sum(x -> abs(Float64(value_of(x))), interior_view(getfield(f, key), grid)), EB_KEYS) / (6 * prod(n))
        for key in PME.HISTORY_KEYS_3D
            got, want = big(Float64(value_of(getfield(h, key)[2]))), ref[key]
            if key in PME.ENERGY_KEYS || key === :K
                record!("T-882", T, Float64(abs(got - want) / want) / ε)
                @test abs(got - want) <= C * ε * want
            else
                record!("T-882 modes", T, Float64(abs(got - want)) / (ε * 2 * mean_abs))
                @test abs(got - want) <= Cm * ε * 2 * mean_abs
            end
        end
        # A pure mode: component c holds A_c·cos(k·x − φ_c) at its own Yee positions, so its amplitude
        # is A_c·e^{−iφ_c}; a component read at another stagger would turn by k_dΔ_d/2 on that axis.
        amps = (Ex = (1.0, 0.3), Ey = (-0.5, 1.1), Ez = (0.75, -2.0), Bx = (1.25, 0.7), By = (0.6, 2.9), Bz = (-0.9, -0.4))
        g = em_fields(backend, T, grid)
        for (key, (A, φ0)) in pairs(amps)
            v = interior_view(getfield(g, key), grid)
            sh = getfield(YEE_SHIFTS_3D, key)
            for j in CartesianIndices(v)
                v[j] = seeded(T, A * Float64(cos(2 * big(π) * sum(d -> EM3_KMODE[d] * (j[d] - 1 + sh[d]) // n[d], 1:3) - big(φ0))))
            end
        end
        PME.record_entry!(r, g, K, grid, 5)
        h = histories(r)
        worst = 0.0
        for (key, (A, φ0)) in pairs(amps)
            re, im = Float64(value_of(getfield(h, Symbol(key, :_re))[5])), Float64(value_of(getfield(h, Symbol(key, :_im))[5]))
            worst = max(worst, abs(re - A * cos(φ0)), abs(im + A * sin(φ0)))
        end
        record!("T-882 pure mode", T, worst / ε)
        @test worst <= emC("T-882", "C_pure") * ε
        k = EM3_KMODE ./ Float64.(box)
        k̂ = k ./ sqrt(sum(abs2, k))
        El = sum(d -> k̂[d] * amps[d][1] * cis(-amps[d][2]), 1:3)
        @test abs(Float64(value_of(h.El_re[5])) - real(El)) <= emC("T-882", "C_pure") * ε
        @test abs(Float64(value_of(h.El_im[5])) - imag(El)) <= emC("T-882", "C_pure") * ε
        @test value_of(mode_amplitude_history(h, :Ez)[5]) ≈ 0.75 rtol = 100ε
        T === D3 && @test Tuple(partials(h.K[2])) != (0, 0, 0)
        # A selection records the same bits and nothing else; an empty one records nothing.
        full = histories(PME.record_entry!(HistoryRecorder(backend, T, grid, box, EM3_KMODE, 5), f, K, grid, 2))
        for (record, kept) in ((:field_energy, PME.ENERGY_KEYS), (:kinetic, (:K,)), (:modes, PME.HISTORY_KEYS_3D[8:end]),
                               ((:kinetic, :modes), PME.HISTORY_KEYS_3D[7:end]))
            rs = HistoryRecorder(backend, T, grid, box, EM3_KMODE, 5; record)
            PME.record_entry!(rs, f, K, grid, 2)
            hs = histories(rs)
            @test keys(hs) == kept
            @test all(key -> same_bits(getfield(hs, key), getfield(full, key)), kept)
        end
        @test isempty(histories(PME.record_entry!(HistoryRecorder(backend, T, grid, box, EM3_KMODE, 5; record = ()), f, K, grid, 2)))
        # launch shapes give the same bits; an empty recorder has its own scratch and shares the tables
        for wg in (7, 64)
            rw = HistoryRecorder(backend, T, grid, box, EM3_KMODE, 5; workgroupsize = wg)
            PME.record_entry!(rw, f, K, grid, 2; workgroupsize = wg)
            @test all(same_bits.(values(histories(rw)), values(full)))
        end
        e = PME.empty_recorder(r)                                      # r holds the entries 2 and 5
        @test e.aux.tables === r.aux.tables && e.aux.partials !== r.aux.partials && e.aux.lines !== r.aux.lines
        @test all(a -> all(iszero, a), e.hist) && !all(a -> all(iszero, a), r.hist)
        PME.record_entry!(e, f, K, grid, 2)
        @test all(same_bits.(values(histories(e)), values(full)))
        # input checks
        @test_throws ArgumentError HistoryRecorder(backend, T, grid, box, (0, 0, 0), 5)
        @test_throws ArgumentError PME.record_entry!(r, f, K, em3_grid(T; n = (12, 8, 12)), 2)
        @test_throws ArgumentError PME.record_entry!(r, f, K, grid, 0)
        @test_throws ArgumentError PME.record_entry!(r, f, K, grid, 6)
    end
end

"""
    em3_state(T; shape, record, every) -> EMState{3}

A 3D state put together from its parts (the module has no 3D problem or step yet): one loaded
species with its workspace, the fields, ρ, the kinetic-energy chunks, the recorder of
`EM3_KMODE` for 6 steps, no modifiers, and the spectral workspace and φ in `aux`.
"""
function em3_state(::Type{T}; shape = Quadratic, record = RECORD_GROUPS, every = 1) where {T}
    grid = em3_grid(T, shape)
    s = em3_species(T, grid)
    numerics = EMNumerics(; shape, current = NoCurrent())
    K = KernelAbstractions.zeros(backend, T, cld(capacity(s), PME.KINETIC_CHUNK))
    recorder = HistoryRecorder(backend, T, grid, em3_box(grid), EM3_KMODE, 6; record, every)
    aux = (spectral = SpectralWorkspace(backend, T, grid), φ = field(backend, T, grid))
    return EMState(numerics, grid, T(1 // 10), (s,), (Workspace(s, grid),), em_fields(backend, T, grid), field(backend, T, grid), K,
                   recorder, (), (), aux, Ref(false))
end

@testset "T-883 a 3D state: the kinetic-energy cache, the initial field and observe! [A7, A8, WP-042]" begin
    C, Ck = emC("T-883"), emC("T-883", "C_kinetic")
    for T in TYPES
        ε = base_eps(T)
        st = em3_state(T)
        @test st isa EMState{3,T} && ndims(st) == 3
        s, grid = st.species[1], st.grid
        dV = prod(Float64.(grid.dx))
        np = alive_count(s)
        # the kinetic energy of fixed chunks of slots, Σ m·w·(γ − 1), against the host's sum in Float64
        @test PME.kinetic!(st) === st && st.fresh[]
        kin(i) = (u2 = sum(d -> Float64(value_of(s.u[d][i]))^2, 1:3); Float64(value_of(s.w[i])) * u2 / (sqrt(1 + u2) + 1))
        chunks = [sum(kin, lo:min(lo + PME.KINETIC_CHUNK - 1, np); init = 0.0) for lo in 1:PME.KINETIC_CHUNK:capacity(s)]
        @test length(st.K) == length(chunks)
        rk = maximum(abs.(Float64.(value_of.(st.K)) .- chunks)) / (ε * maximum(chunks))
        record!("T-883 kinetic", T, rk)
        @test rk <= Ck
        @test sum(chunks) > 0.1                                        # warm particles: the sum is not trivially zero
        # the state's initial field is the function on its arrays, bit for bit
        ρ = em3_charge(Reference(), s, st.numerics.shape, grid)
        copyto!(st.ρ, ρ)
        @test PME.initial_field!(st) === st
        f = em_fields(backend, T, grid)
        PME.initial_field!(f, field(backend, T, grid), ρ, grid, SpectralWorkspace(backend, T, grid))
        @test all(key -> same_bits(getfield(st.fields, key), getfield(f, key)), keys(f))
        @test any(!iszero, st.fields.Ez)
        # observe! at step 3 writes entry 3: the function on the arrays, and the closed form of the
        # loaded mode at the recorder's wave vector (another mode than the charge's: amplitude zero)
        @test observe!(st.recorder, st, 3) === nothing
        h = histories(st)
        r2 = HistoryRecorder(backend, T, grid, em3_box(grid), EM3_KMODE, 6)
        PME.record_entry!(r2, st.fields, st.K, grid, 3)
        @test all(same_bits.(values(h), values(histories(r2))))
        @test Float64(value_of(h.K[3])) ≈ sum(chunks) * dV rtol = 100ε
        ref = initial_field_reference(st.numerics.shape, grid)
        rm = HistoryRecorder(backend, T, grid, em3_box(grid), EM3_MODE, 6)
        observe!(rm, st, 1)
        hm = histories(rm)
        scale = Float64(ref.A * sqrt(sum(abs2, ref.S)) / ref.K2)
        for (d, key) in enumerate((:Ex, :Ey, :Ez))
            # E_d = q·n₀·A·S_d/K²·sin(k·x) at its own stagger, q·n₀ = −1: amplitude +i·A·S_d/K²
            err = max(abs(Float64(value_of(getfield(hm, Symbol(key, :_re))[1]))),
                      abs(Float64(value_of(getfield(hm, Symbol(key, :_im))[1])) - Float64(ref.A * ref.S[d] / ref.K2)))
            record!("T-883", T, err / (ε * sum(grid.n) * scale))
            @test err <= C * ε * sum(grid.n) * scale
        end
        @test Float64(value_of(hypot(hm.El_re[1], hm.El_im[1]))) > scale / 2    # the field is longitudinal up to the grid's dispersion
        @test abs(Float64(value_of(hypot(h.Ex_re[3], h.Ex_im[3])))) <= C * ε * sum(grid.n) * scale
        # a stale cache is rebuilt before it is recorded
        s.u[1] .*= 2
        st.fresh[] = false
        observe!(st.recorder, st, 4)
        @test st.fresh[] && value_of(histories(st).K[4]) > value_of(h.K[3])
        @test PME.refresh_caches!(st) === st
        # a recorder that does not record the step, or of another N, another grid or another length
        re = HistoryRecorder(backend, T, grid, em3_box(grid), EM3_KMODE, 6; every = 2)
        observe!(re, st, 2)
        @test all(a -> all(iszero, a), re.hist)
        observe!(re, st, 3)
        @test !iszero(re.hist.WEz[2]) && iszero(re.hist.WEz[1]) && histories(re).steps == [1, 3, 5]
        g2 = PeriodicGrid((12, 8), (basetype(T)(1 // 4), basetype(T)(3 // 8)), 3)
        @test_throws ArgumentError observe!(HistoryRecorder(backend, T, g2, (3.0, 3.0), (1, 2), 6), st, 1)
        other = em3_grid(T; n = (12, 8, 12))
        @test_throws ArgumentError observe!(HistoryRecorder(backend, T, other, em3_box(other), EM3_KMODE, 6; every = 2), st, 2)
        @test_throws ArgumentError observe!(st.recorder, st, 7)
    end
end
