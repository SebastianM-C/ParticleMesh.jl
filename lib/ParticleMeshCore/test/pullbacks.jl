# WP-077: the hand-written pullbacks of the transfers and the sort (T-1060 to T-1064), and
# Enzyme's reverse mode through them (the reverse rules of ParticleMeshCoreEnzymeCoreExt).
#
# The checks are chain-rule identities against forward-mode duals, near roundoff: for a direction
# v of the species' floating-point arrays (and of the field, for the gather) and an adjoint ȳ of the
# transfer's output, the pullback x̄ satisfies ⟨x̄, v⟩ = ⟨ȳ, J·v⟩, with J·v the partials of the
# transfer run on a species of `Dual{Float64,1}` with partials v. Each floating-point array of the
# species is a direction of its own, so a dropped or misplaced term fails by name. Float64 on the
# CPU; Float32 against Float64 for the element type.

using Enzyme: Enzyme, autodiff, Reverse, Active, Const, Duplicated

const DP1 = Dual{Nothing,Float64,1}
const PMC = ParticleMeshCore

# The pullbacks of the test quantities (q·w·u_y, q·w·u_z; ChargeUy2 is the 2D q·w·u₂).
for Q in (:ChargeUy, :ChargeUy2, :ChargeUz)
    k = Q === :ChargeUz ? 3 : 2
    @eval @inline function ParticleMeshCore.pullback_quantity!(s̄, ::$Q, s, i, q̄)
        @inbounds begin
            s̄.w[i] += s.q * s.u[$k][i] * q̄
            s̄.u[$k][i] += s.q * s.w[i] * q̄
        end
        return nothing
    end
end

"A species with the layout of `s` (element type `T`) and every array zero."
function zero_like(s::Species{N}, ::Type{T} = eltype(s)) where {N,T}
    z = Species(backend, T, capacity(s); q = s.q, m = s.m, N, extras = map(a -> eltype(a) <: AbstractFloat ? T : eltype(a), s.extras))
    for (a, b) in zip(PMC.species_arrays(z), PMC.species_arrays(s))
        eltype(a) <: AbstractFloat || copyto!(a, b)
    end
    copyto!(z.count, s.count)
    return z
end

"The floating-point arrays of a species, in `species_arrays` order, with their names."
function float_arrays(s::Species{N}) where {N}
    names = (ntuple(d -> "cell$d", N)..., ntuple(d -> "offset$d", N)..., "u1", "u2", "u3", "w", "status", map(string, keys(s.extras))...)
    return [(n, a) for (n, a) in zip(names, PMC.species_arrays(s)) if eltype(a) <: AbstractFloat]
end

"Deterministic values in (-1, 1) for an array of `len` entries, from `seed`."
function lcg_values(len, seed)
    state = UInt64(seed) * 0x9e3779b97f4a7c15 + 0x2545f4914f6cdd1d
    return [(state = state * 0x5851f42d4c957f2d + 0x14057b7ef767814f; 2 * Float64(state >> 11) / 2.0^53 - 1) for _ in 1:len]
end
lcg_field(a, seed) = reshape(lcg_values(length(a), seed), size(a))

"A direction: zero everywhere but the floating-point array `k` of `s`, which takes values in (-1, 1)."
function direction(s, k; seed = 7)
    v = zero_like(s, Float64)
    _, a = float_arrays(v)[k]
    a .= lcg_values(length(a), seed + k)
    return v
end

"`s` with every floating-point array a `Dual{Float64,1}` whose partial is the matching array of `v`."
function dualize(s::Species{N}, v::Species{N}) where {N}
    d = zero_like(s, DP1)
    for (a, b, c) in zip(PMC.species_arrays(d), PMC.species_arrays(s), PMC.species_arrays(v))
        eltype(a) === DP1 && (a .= Dual{Nothing}.(Float64.(b), c))
    end
    return d
end

"⟨x̄, v⟩ over the floating-point arrays of two species of one layout."
pair(x̄::Species, v::Species) = sum(sum(Float64.(a) .* b) for ((_, a), (_, b)) in zip(float_arrays(x̄), float_arrays(v)))
"⟨ȳ, ∂y⟩ and Σ|ȳ·∂y| for fields of duals and their adjoints."
function jvp_pair(ȳs, ys)
    terms = reduce(vcat, [vec(ȳ .* map(y -> partials(y)[1], yd)) for (ȳ, yd) in zip(ȳs, ys)])
    return sum(terms), sum(abs, terms)
end

"The ratio |⟨x̄, v⟩ - ⟨ȳ, J·v⟩| / (eps·scale), recorded under `id`."
chain_ratio(id, lhs, rhs, scale, T = Float64) = record!(id, T, abs(lhs - rhs) / (eps(T) * max(scale, floatmin(Float64))))

const FLUX1D_N = 12
flux_grid_1d(::Type{T}, shape, st) where {T} = PeriodicGrid(FLUX1D_N, T(1 // 4), flux_guard_width(shape, st))

@testset "T-1060 shape and flux-frame slopes are the derivatives of the weights [A2, A7, WP-077]" begin
    C = tolC("T-1060")
    for shape in (Shape{0}(), SHAPES...), st in (Node(), Center()), δ in (0.0, 0.1, 0.25, 0.5, 0.7, prevfloat(1.0))
        f, w = shape_weights(shape, st, Dual{Nothing}(δ, 1.0))
        fs, dw = PMC.shape_slopes(shape, st, δ)
        @test fs == f
        @test record!("T-1060", Float64, maximum(abs.(dw .- map(x -> partials(x)[1], w))) / eps(Float64)) <= C
        @test abs(sum(dw)) <= C * eps(Float64)
    end
    # the frame's weights and slopes, for old cells one to the left, the same and one to the right
    for shape in SHAPES, st in (Node(), Center()), (δ, δ0) in ((0.2, 0.9), (0.6, 0.1), (0.05, 0.95)), e in (-1, 0, 1)
        c, n = Int32(5), Int32(12)
        c0 = c + Int32(e)
        _, Wo, Wn = PMC.flux_axis_frame(shape, st, c, Dual{Nothing}(δ, 1.0, 0.0), c0, Dual{Nothing}(δ0, 0.0, 1.0), n)
        dWo, dWn = PMC.flux_axis_slopes(shape, st, c, δ, c0, δ0, n)
        @test record!("T-1060", Float64, maximum(abs.(dWn .- map(x -> partials(x)[1], Wn))) / eps(Float64)) <= C
        @test record!("T-1060", Float64, maximum(abs.(dWo .- map(x -> partials(x)[2], Wo))) / eps(Float64)) <= C
        @test all(x -> partials(x)[2] == 0, Wn) && all(x -> partials(x)[1] == 0, Wo)
    end
end

"The deposit cases of T-1061: (N, species, grid, staggers, quantities) for N = 1, 2, 3."
function deposit_cases(::Type{T}) where {T}
    s1 = moved_species(T, PeriodicGrid(FLUX1D_N, T(1 // 4), 2), 300; seed = 5, dead = (40, 41))
    s2 = flux_species_2d(T, grid2(T, 2), 300; seed = 5)
    s3 = test_species_3d(T, grid3(T, 2), 200; seed = 5)
    return ((1, s1, PeriodicGrid(FLUX1D_N, T(1 // 4), 2), (Node(), Center()), (ChargeWeight(), ChargeUy())),
            (2, s2, grid2(T, 2), STAGGER_PAIRS, (ChargeWeight(), ChargeUy2())),
            (3, s3, grid3(T, 2), STAGGER_TRIPLES[1:2], (ChargeWeight(),)))
end

@testset "T-1061 deposit_pullback! and gather_pullback! against duals, N = 1, 2, 3, and the gather with one shape per axis [A3, A7, WP-077]" begin
    C = tolC("T-1061")
    for (N, s, grid, staggers, quantities) in deposit_cases(Float64), shape in SHAPES, st in staggers, quantity in quantities
        ρ̄ = lcg_field(field(backend, Float64, grid), 11)
        x̄ = zero_like(s)
        deposit_pullback!(x̄, ρ̄, s, shape, st, grid; quantity)
        for k in eachindex(float_arrays(s))
            v = direction(s, k)
            sD = dualize(s, v)
            ρD = deposit!(Fast(), field(backend, DP1, grid), sD, shape, st, grid, Workspace(sD, grid); quantity)
            rhs, scale = jvp_pair((ρ̄,), (ρD,))
            name = float_arrays(s)[k][1]
            r = chain_ratio("T-1061", pair(x̄, v), rhs, scale)
            r <= C || @info "T-1061 deposit" N shape st quantity name r
            @test r <= C
        end
        # the gather: a direction in the field and one in each offset; ō is zeroed where alive
        a = lcg_field(field(backend, Float64, grid), 13)
        ȧ = lcg_field(a, 17)
        ō = lcg_values(capacity(s), 19)
        ā = zero(a)
        x̄ = zero_like(s)
        ō1 = copy(ō)
        gather_pullback!(ā, x̄, ō1, a, s, shape, st, grid)
        @test all(i -> s.status[i] == PMC.ALIVE ? ō1[i] == 0 : ō1[i] == ō[i], 1:capacity(s))
        for k in 1:N
            v = direction(s, findfirst(t -> t[1] == "offset$k", float_arrays(s)))
            sD = dualize(s, v)
            aD = Dual{Nothing}.(a, ȧ)
            outD = gather!(zeros(DP1, capacity(s)), aD, sD, shape, st, grid)
            alive = [s.status[i] == PMC.ALIVE for i in 1:capacity(s)]
            rhs, scale = jvp_pair((ō .* alive,), (outD,))
            r = chain_ratio("T-1061", sum(ā .* ȧ) + pair(x̄, v), rhs, scale)
            r <= C || @info "T-1061 gather" N shape st k r
            @test r <= C
        end
    end
    # one shape per axis, as a Galerkin gather reads (WP-070), order 0 included
    for (N, s, grid, staggers, _) in deposit_cases(Float64)[2:3],
        shape in (N == 2 ? ((Shape{0}(), Quadratic), (Linear, Quadratic), (Quadratic, Shape{0}())) : ((Quadratic, Linear, Shape{0}()),)),
        st in staggers
        a = lcg_field(field(backend, Float64, grid), 13)
        ȧ = lcg_field(a, 17)
        ō = lcg_values(capacity(s), 19)
        ā = zero(a)
        x̄ = zero_like(s)
        gather_pullback!(ā, x̄, copy(ō), a, s, shape, st, grid)
        alive = [s.status[i] == PMC.ALIVE for i in 1:capacity(s)]
        for k in 1:N
            v = direction(s, findfirst(t -> t[1] == "offset$k", float_arrays(s)))
            outD = gather!(zeros(DP1, capacity(s)), Dual{Nothing}.(a, ȧ), dualize(s, v), shape, st, grid)
            rhs, scale = jvp_pair((ō .* alive,), (outD,))
            r = chain_ratio("T-1061", sum(ā .* ȧ) + pair(x̄, v), rhs, scale)
            r <= C || @info "T-1061 per-axis gather" N shape st k r
            @test r <= C
        end
    end
end

@testset "T-1062 flux_pullback! against duals at N = 1 and 2, every input array, Float32 against Float64, node bins with filled guards [A2, A3, A7, WP-077]" begin
    C = tolC("T-1062")
    C32 = tolC("T-1062", "C32")
    for shape in SHAPES, st in (Node(), Center())
        grid = flux_grid_1d(Float64, shape, st)
        s = moved_species(Float64, grid, 300; seed = 9, dead = (40, 41, 250))
        J̄ = ntuple(k -> lcg_field(field(backend, Float64, grid), 20 + k), 3)
        x̄ = zero_like(s)
        flux_pullback!(x̄, J̄..., s, shape, st, grid; dt = 0.125, quantity_y = ChargeUy(), quantity_z = ChargeUz())
        for k in eachindex(float_arrays(s))
            v = direction(s, k)
            sD = dualize(s, v)
            JD = ntuple(_ -> field(backend, DP1, grid), 3)
            deposit_flux!(Fast(), JD..., sD, shape, st, grid, Workspace(sD, grid); dt = DP1(0.125), quantity_y = ChargeUy(), quantity_z = ChargeUz())
            rhs, scale = jvp_pair(J̄, JD)
            r = chain_ratio("T-1062", pair(x̄, v), rhs, scale)
            r <= C || @info "T-1062 N = 1" shape st float_arrays(s)[k][1] r
            @test r <= C
        end
        # Float32: the same pullback within Float32 roundoff of the Float64 one
        s32 = moved_species(Float32, flux_grid_1d(Float32, shape, st), 300; seed = 9, dead = (40, 41, 250))
        x̄32 = zero_like(s32)
        flux_pullback!(x̄32, map(a -> Float32.(a), J̄)..., s32, shape, st, flux_grid_1d(Float32, shape, st); dt = 0.125f0,
                       quantity_y = ChargeUy(), quantity_z = ChargeUz())
        for ((_, a32), (_, a64)) in zip(float_arrays(x̄32), float_arrays(x̄))
            @test record!("T-1062 Float32", Float32, maximum(abs.(a32 .- a64)) / (eps(Float32) * max(maximum(abs, a64), 1))) <= C32
        end
    end
    for shape in SHAPES, st in STAGGER_PAIRS
        grid = grid2(Float64, flux_g2(shape, st))
        s = flux_species_2d(Float64, grid, 300; seed = 9)
        J̄ = ntuple(k -> lcg_field(field(backend, Float64, grid), 30 + k), 3)
        x̄ = zero_like(s)
        flux_pullback!(x̄, J̄..., s, shape, st, grid; dt = 0.125, quantity_z = QZ2)
        for k in eachindex(float_arrays(s))
            v = direction(s, k)
            sD = dualize(s, v)
            JD = flux2d(Reference(), sD, shape, st, grid; dt = 0.125)[1:3]
            rhs, scale = jvp_pair(J̄, JD)
            r = chain_ratio("T-1062", pair(x̄, v), rhs, scale)
            r <= C || @info "T-1062 N = 2" shape st float_arrays(s)[k][1] r
            @test r <= C
        end
        g32 = grid2(Float32, flux_g2(shape, st))
        s32 = flux_species_2d(Float32, g32, 300; seed = 9)
        x̄32 = zero_like(s32)
        flux_pullback!(x̄32, map(a -> Float32.(a), J̄)..., s32, shape, st, g32; dt = 0.125f0, quantity_z = QZ2)
        for ((_, a32), (_, a64)) in zip(float_arrays(x̄32), float_arrays(x̄))
            @test record!("T-1062 Float32", Float32, maximum(abs.(a32 .- a64)) / (eps(Float32) * max(maximum(abs, a64), 1))) <= C32
        end
    end
    # nearest-node bins: a Reference deposit writes some contributions into the guard on the other
    # side, so the pullback is its transpose for an adjoint equal on a guard and its periodic image,
    # as fold_guards!'s adjoint (fill_guards!) gives it
    for (shape, st) in ((Linear, (Center(), Center())), (Quadratic, (Node(), Node())))
        grid = grid2(Float64, flux_g2(shape, st) + 1)
        s = flux_species_2d(Float64, grid, 300; seed = 9)
        J̄ = ntuple(k -> fill_guards!(lcg_field(field(backend, Float64, grid), 30 + k), grid), 3)
        x̄ = zero_like(s)
        flux_pullback!(x̄, J̄..., s, shape, st, grid; dt = 0.125, quantity_z = QZ2)
        for k in eachindex(float_arrays(s))
            v = direction(s, k)
            JD = flux2d(Reference(), dualize(s, v), shape, st, grid; dt = 0.125, bins = NodeBins(shape, st))[1:3]
            rhs, scale = jvp_pair(J̄, JD)
            r = chain_ratio("T-1062", pair(x̄, v), rhs, scale)
            r <= C || @info "T-1062 NodeBins" shape st float_arrays(s)[k][1] r
            @test r <= C
        end
    end
end

@testset "T-1063 sort_pullback! is the inverse permutation of the sort, bitwise, Float64 and Float32 [A4, WP-077]" begin
    for (N, s0, grid) in ((1, moved_species(Float64, PeriodicGrid(FLUX1D_N, 0.25, 2), 300; seed = 5, dead = (40, 41)), PeriodicGrid(FLUX1D_N, 0.25, 2)),
                          (2, flux_species_2d(Float64, grid2(Float64, 2), 300; seed = 5), grid2(Float64, 2)),
                          (2, flux_species_2d(Float32, grid2(Float32, 2), 300; seed = 5), grid2(Float32, 2))),
        resort in (false, true)
        s = deepcopy(s0)
        ws = Workspace(s, grid; append = resort ? 8 : nothing)
        resort && sort_by_cell!(s, grid, ws)
        x = [copy(a) for (_, a) in float_arrays(s)]
        resort ? resort_by_cell!(s, grid, ws) : sort_by_cell!(s, grid, ws)
        perm = copy(ws.perm)
        ȳ = zero_like(s)
        for (k, (_, a)) in enumerate(float_arrays(ȳ))
            a .= lcg_values(length(a), 40 + k)
        end
        ints = [copy(a) for a in PMC.species_arrays(ȳ) if !(eltype(a) <: AbstractFloat)]
        ȳ0 = [copy(a) for (_, a) in float_arrays(ȳ)]
        sort_pullback!(ȳ, perm)
        # the transpose: ⟨Pᵀȳ, x⟩ = ⟨ȳ, Px⟩ term by term, so the arrays are equal entry by entry
        for ((_, a), b, xa, (_, ya)) in zip(float_arrays(ȳ), ȳ0, x, float_arrays(s))
            @test a[perm] == b
            @test same_bits(xa[perm], ya)
        end
        @test all(a == b for (a, b) in zip((a for a in PMC.species_arrays(ȳ) if !(eltype(a) <: AbstractFloat)), ints))
    end
end

"`Σ_k ⟨J̄_k, J_k⟩`, written as a loop for Enzyme."
function inner3(J̄, J)
    acc = 0.0
    for k in 1:3, j in eachindex(J[k])
        acc += J̄[k][j] * J[k][j]
    end
    return acc
end

function flux_loss(mode, s, ws, J, J̄, shape, st, grid, sorts, overwrite)
    sorts && sort_by_cell!(s, grid, ws)
    foreach(a -> fill!(a, 0.0), J)
    deposit_flux!(mode, J..., s, shape, st, grid, ws; dt = 0.25, quantity_z = QZ2)
    overwrite && overwrite_species!(s)
    return inner3(J̄, J)
end

"Change the species after a transfer, as a push does: the rules must read their tape."
function overwrite_species!(s)
    s.w .*= 2
    s.offset[1] .*= 0.5
    s.u[1] .*= 3
    return nothing
end

struct FluxHolder{S,W,T}
    s::S
    ws::W
    dt::T
end

function holder_loss(h, J, J̄, shape, st, grid)
    foreach(a -> fill!(a, 0.0), J)
    deposit_flux!(Fast(), J..., h.s, shape, st, grid, h.ws; dt = h.dt, quantity_z = QZ2)
    return inner3(J̄, J)
end

function flux_loss_xy(s, ws, Jx, Jy, Jz, J̄, shape, st, grid)
    fill!(Jx, 0.0)
    fill!(Jy, 0.0)
    fill!(Jz, 0.0)
    deposit_flux!(Fast(), Jx, Jy, Jz, s, shape, st, grid, ws; dt = 0.25, quantity_z = QZ2)
    acc = 0.0
    for j in eachindex(Jx)
        acc += J̄[1][j] * Jx[j] + J̄[2][j] * Jy[j]
    end
    return acc
end

function deposit_loss(mode, s, ws, ρ, ρ̄, shape, st, grid, quantity, overwrite)
    effective_mode(mode, Float64) isa Reference && sort_by_cell!(s, grid, ws)
    fill!(ρ, 0.0)
    deposit!(mode, ρ, s, shape, st, grid, ws; quantity)
    overwrite && overwrite_species!(s)
    acc = 0.0
    for j in eachindex(ρ)
        acc += ρ̄[j] * ρ[j]
    end
    return acc
end

function gather_loss(out, ō, a, s, shape, st, grid, overwrite)
    gather!(out, a, s, shape, st, grid)
    if overwrite                       # the field and the species change after the gather
        a .*= 2
        overwrite_species!(s)
    end
    acc = 0.0
    for i in eachindex(out)
        acc += ō[i] * out[i]
    end
    return acc
end

@testset "T-1064 Enzyme's reverse mode through the rules: deposit_flux!, deposit! and gather! in both modes, with the sort [A4, A7, WP-077]" begin
    C = tolC("T-1064")
    shape, st = Quadratic, (Node(), Node())
    grid = grid2(Float64, flux_g2(shape, st))
    s0 = flux_species_2d(Float64, grid, 300; seed = 9)
    J̄ = ntuple(k -> lcg_field(field(backend, Float64, grid), 30 + k), 3)
    for mode in (Fast(), Reference()), overwrite in (false, true)
        s = deepcopy(s0)
        ws = Workspace(s, grid; flux = (shape, st))
        J = fields3(Float64, grid)
        x̄ = zero_like(s)
        autodiff(Reverse, flux_loss, Active, Const(mode), Duplicated(s, x̄), Const(ws), Duplicated(J, map(zero, J)), Const(J̄),
                 Const(shape), Const(st), Const(grid), Const(mode isa Reference), Const(overwrite))
        for k in eachindex(float_arrays(s0))
            v = direction(s0, k)
            JD = flux2d(Reference(), dualize(s0, v), shape, st, grid)[1:3]
            rhs, scale = jvp_pair(J̄, JD)
            # with the overwrite, the loss also sees nothing of the doubled weights: J was deposited before
            r = chain_ratio("T-1064", pair(x̄, v), rhs, scale)
            r <= C || @info "T-1064 flux" mode overwrite float_arrays(s0)[k][1] r
            @test r <= C
        end
    end
    # dt read from a differentiated structure, as a step reads it from its state: a constant
    # keyword (inactive_kwarg), the same gradient, and no adjoint for dt
    sh = deepcopy(s0)
    h = FluxHolder(sh, Workspace(sh, grid; flux = (shape, st)), 0.25)
    dh = Enzyme.make_zero(h)
    J = fields3(Float64, grid)
    autodiff(Reverse, holder_loss, Active, Duplicated(h, dh), Duplicated(J, map(zero, J)), Const(J̄), Const(shape), Const(st), Const(grid))
    @test dh.dt == 0
    for k in eachindex(float_arrays(s0))
        v = direction(s0, k)
        rhs, scale = jvp_pair(J̄, flux2d(Reference(), dualize(s0, v), shape, st, grid)[1:3])
        @test chain_ratio("T-1064", pair(dh.s, v), rhs, scale) <= C
    end
    # a constant Jz: its adjoint is zero, so only Jx and Jy reach the species
    sc = deepcopy(s0)
    x̄ = zero_like(sc)
    J = fields3(Float64, grid)
    autodiff(Reverse, flux_loss_xy, Active, Duplicated(sc, x̄), Const(Workspace(sc, grid; flux = (shape, st))), Duplicated(J[1], zero(J[1])),
             Duplicated(J[2], zero(J[2])), Const(J[3]), Const(J̄), Const(shape), Const(st), Const(grid))
    J̄xy = (J̄[1], J̄[2], zero(J̄[3]))
    for k in eachindex(float_arrays(s0))
        v = direction(s0, k)
        rhs, scale = jvp_pair(J̄xy, flux2d(Reference(), dualize(s0, v), shape, st, grid)[1:3])
        @test chain_ratio("T-1064", pair(x̄, v), rhs, scale) <= C
    end
    # deposit! and gather! at N = 2, ρ's own stagger
    s0 = flux_species_2d(Float64, grid2(Float64, 2), 300; seed = 5)
    g2 = grid2(Float64, 2)
    ρ̄ = lcg_field(field(backend, Float64, g2), 11)
    for mode in (Fast(), Reference()), quantity in (ChargeWeight(), ChargeUy2()), overwrite in (false, true)
        s = deepcopy(s0)
        x̄ = zero_like(s)
        ρ = field(backend, Float64, g2)
        autodiff(Reverse, deposit_loss, Active, Const(mode), Duplicated(s, x̄), Const(Workspace(s, g2)), Duplicated(ρ, zero(ρ)),
                 Const(ρ̄), Const(shape), Const(st), Const(g2), Const(quantity), Const(overwrite))
        for k in eachindex(float_arrays(s0))
            v = direction(s0, k)
            sD = dualize(s0, v)
            ρD = deposit!(Fast(), field(backend, DP1, g2), sD, shape, st, g2, Workspace(sD, g2); quantity)
            rhs, scale = jvp_pair((ρ̄,), (ρD,))
            r = chain_ratio("T-1064", pair(x̄, v), rhs, scale)
            r <= C || @info "T-1064 deposit" mode quantity overwrite float_arrays(s0)[k][1] r
            @test r <= C
        end
    end
    a = lcg_field(field(backend, Float64, g2), 13)
    ȧ = lcg_field(a, 17)
    ō = lcg_values(capacity(s0), 19)
    alive = [s0.status[i] == PMC.ALIVE for i in 1:capacity(s0)]
    for overwrite in (false, true)
        ā = zero(a)
        x̄ = zero_like(s0)
        autodiff(Reverse, gather_loss, Active, Duplicated(zeros(capacity(s0)), zeros(capacity(s0))), Const(ō), Duplicated(copy(a), ā),
                 Duplicated(deepcopy(s0), x̄), Const(shape), Const(st), Const(g2), Const(overwrite))
        for d in 1:2
            v = direction(s0, findfirst(t -> t[1] == "offset$d", float_arrays(s0)))
            outD = gather!(zeros(DP1, capacity(s0)), Dual{Nothing}.(a, ȧ), dualize(s0, v), shape, st, g2)
            rhs, scale = jvp_pair((ō .* alive,), (outD,))
            r = chain_ratio("T-1064", sum(ā .* ȧ) + pair(x̄, v), rhs, scale)
            r <= C || @info "T-1064 gather" overwrite d r
            @test r <= C
        end
    end
    # a constant species that moves after the gather: the field's adjoint reads the taped positions
    ā = zero(a)
    autodiff(Reverse, gather_loss, Active, Duplicated(zeros(capacity(s0)), zeros(capacity(s0))), Const(ō), Duplicated(copy(a), ā),
             Const(deepcopy(s0)), Const(shape), Const(st), Const(g2), Const(true))
    āref = zero(a)
    gather_pullback!(āref, nothing, copy(ō), a, s0, shape, st, g2)
    @test record!("T-1064", Float64, maximum(abs.(ā .- āref)) / (eps(Float64) * maximum(abs, āref))) <= C
end

# WP-080: the flux pullback at N = 3 and Enzyme through its rule (T-1110, T-1111)

@testset "T-1110 flux_pullback! against duals at N = 3, every input array, both orders and four stagger triples, Float32 against Float64 [A2, A3, A7, WP-080]" begin
    C = tolC("T-1110")
    C32 = tolC("T-1110", "C32")
    for shape in SHAPES, st in FLUX3D_TRIPLES
        grid = flux_grid3(Float64, shape, st)
        s = flux_species_3d(Float64, grid, 200; seed = 9)
        J̄ = ntuple(k -> lcg_field(field(backend, Float64, grid), 40 + k), 3)
        x̄ = zero_like(s)
        flux_pullback!(x̄, J̄..., s, shape, st, grid; dt = 0.125)
        # it adds to the adjoint it is given (in a step the push's and the gather's are there first)
        x̄1 = zero_like(s)
        foreach(((_, a),) -> fill!(a, 1.0), float_arrays(x̄1))
        flux_pullback!(x̄1, J̄..., s, shape, st, grid; dt = 0.125)
        for ((_, a1), (_, a0)) in zip(float_arrays(x̄1), float_arrays(x̄))
            @test maximum(abs.((a1 .- 1) .- a0)) <= 4 * eps(Float64) * (1 + maximum(abs, a0))
        end
        for k in eachindex(float_arrays(s))
            v = direction(s, k)
            JD = flux3d(Reference(), dualize(s, v), shape, st, grid; dt = 0.125)[1:3]
            rhs, scale = jvp_pair(J̄, JD)
            r = chain_ratio("T-1110", pair(x̄, v), rhs, scale)
            r <= C || @info "T-1110" shape st float_arrays(s)[k][1] r
            @test r <= C
        end
        g32 = flux_grid3(Float32, shape, st)
        s32 = flux_species_3d(Float32, g32, 200; seed = 9)
        x̄32 = zero_like(s32)
        flux_pullback!(x̄32, map(a -> Float32.(a), J̄)..., s32, shape, st, g32; dt = 0.125f0)
        for ((_, a32), (_, a64)) in zip(float_arrays(x̄32), float_arrays(x̄))
            @test record!("T-1110 Float32", Float32, maximum(abs.(a32 .- a64)) / (eps(Float32) * max(maximum(abs, a64), 1))) <= C32
        end
    end
    grid = flux_grid3(Float64, Quadratic, NODES3)
    s = flux_species_3d(Float64, grid, 20)
    J̄ = fields3(Float64, grid)
    @test_throws ArgumentError flux_pullback!(zero_like(s), J̄..., s, Quadratic, NODES3, grid; dt = 0.125, quantity_z = ChargeWeight())
end

function flux_loss_3d(mode, s, ws, J, J̄, shape, st, grid, overwrite)
    effective_mode(mode, Float64) isa Reference && sort_by_cell!(s, grid, ws)
    foreach(a -> fill!(a, 0.0), J)
    deposit_flux!(mode, J..., s, shape, st, grid, ws; dt = 0.25)
    overwrite && overwrite_species!(s)
    return inner3(J̄, J)
end

@testset "T-1111 Enzyme's reverse mode through the N = 3 flux rule in both modes, with the sort [A4, A7, WP-080]" begin
    C = tolC("T-1111")
    for (shape, st) in ((Quadratic, NODES3), (Linear, (Center(), Node(), Center())))
        grid = flux_grid3(Float64, shape, st)
        s0 = flux_species_3d(Float64, grid, 200; seed = 11)
        J̄ = ntuple(k -> lcg_field(field(backend, Float64, grid), 50 + k), 3)
        for mode in (Fast(), Reference()), overwrite in (false, true)
            s = deepcopy(s0)
            ws = Workspace(s, grid; flux = (shape, st))
            J = fields3(Float64, grid)
            x̄ = zero_like(s)
            autodiff(Reverse, flux_loss_3d, Active, Const(mode), Duplicated(s, x̄), Const(ws), Duplicated(J, map(zero, J)), Const(J̄),
                     Const(shape), Const(st), Const(grid), Const(overwrite))
            for k in eachindex(float_arrays(s0))
                v = direction(s0, k)
                JD = flux3d(Reference(), dualize(s0, v), shape, st, grid)[1:3]
                rhs, scale = jvp_pair(J̄, JD)
                r = chain_ratio("T-1111", pair(x̄, v), rhs, scale)
                r <= C || @info "T-1111" shape st mode overwrite float_arrays(s0)[k][1] r
                @test r <= C
            end
        end
    end
end
