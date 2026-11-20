# WP-071, the flux gather `gather_flux!`, the transpose of `deposit_flux!`: T-1000 to T-1003 and T-1005 (T-1004,
# its CUDA rerun, is in gpu.jl). The cases are the flux tests' own: the 1D, 2D and 3D species of
# flux_cases.jl, flux2d_cases.jl and flux3d_cases.jl on their grids, every particle moved by less than
# a cell, faces and seams crossed.

"The ρ stagger of `N` axes from one stagger per axis (`N = 1` takes the bare `Stagger`)."
gather_staggers(N) = N == 1 ? STAGGERS : N == 2 ? STAGGER_PAIRS : FLUX3D_TRIPLES

"A field value of `T` from a real `x`; for duals the partials vary with `x` too."
gval(::Type{T}, x) where {T<:AbstractFloat} = T(x)
gval(::Type{D3}, x) = mkdual(x, sin(3x), cos(5x), x / 3)

"The interior indices of a field on `grid`, as Cartesian indices of the full array."
interior_indices(grid) = CartesianIndices(map(m -> (guards(grid) + 1):(guards(grid) + m), Tuple(ncells(grid))))

"A host field on `grid` for component `c`, interior values from a fixed smooth-free sequence, guards filled."
function gather_field(::Type{T}, grid, c) where {T}
    E = field(backend, T, grid)
    for (k, I) in enumerate(interior_indices(grid))
        E[I] = gval(T, sin(1.37k + 0.9c) + c / 4)
    end
    fill_guards!(E, grid)
    return E
end

"""
    transpose_case(T, N, mode, shape, st; bins, be, maxstep, workgroupsize) -> NamedTuple

One step of the flux tests' species at `N` (400 particles, faces and seams crossed): `deposit_flux!`
in `mode` (by `bins` where it sorts; at ρ's stagger `dep_st`, which the control sets to a wrong one) into fields of zeros, guards folded, and `gather_flux!` of three
fields `E` with filled guards into outputs that hold a sentinel, every engine call on `be`. Also the
quantity each component deposits per particle, `q·w` in space and the module's quantity out of it,
and the factor `Δx_d/dt` (in space) or 1 (out of it) that `gather_flux!` puts on the weights.
"""
function transpose_case(::Type{T}, N, mode, shape, st; bins = CellBins(), be = CPU(), maxstep = 9 / 10, workgroupsize = nothing,
                        seed = 3, dep_st = st) where {T}
    dt = basetype(T)(1 // 4)
    g = max(maximum(flux_guard_width(shape, st)), maximum(flux_guard_width(shape, dep_st)))
    if N == 1
        grid = PeriodicGrid(10, basetype(T)(1 // 2), g)
        s = moved_species(T, grid, 400; seed, maxstep, dead = (40, 41, 300))
        ws = Workspace(s, grid; bins)
        effective_mode(mode, T) isa Reference && on(be, sort_by_cell!, s, grid, ws)
        J = fields3(T, grid)
        on(be, deposit_flux!, mode, J..., s, shape, dep_st, grid, ws; dt = T(dt), quantity_y = ChargeUy(), quantity_z = ChargeUz())
        quantities = (ChargeWeight(), ChargeUy(), ChargeUz())
        space = (true, false, false)
    elseif N == 2
        grid = grid2(T, g)
        s = flux_species_2d(T, grid, 400; seed, maxstep)
        J = flux2d(mode, s, shape, dep_st, grid; dt, be, bins)[1:3]
        quantities = (ChargeWeight(), ChargeWeight(), QZ2)
        space = (true, true, false)
    else
        grid = grid3(T, g; n = N3F)
        s = flux_species_3d(T, grid, 400; seed, maxstep)
        J = flux3d(mode, s, shape, dep_st, grid; dt, be, bins)[1:3]
        quantities = (ChargeWeight(), ChargeWeight(), ChargeWeight())
        space = (true, true, true)
    end
    foreach(a -> on(be, fold_guards!, a, grid), J)
    E = ntuple(c -> gather_field(T, grid, c), 3)
    o = ntuple(_ -> fill(gval(T, 7.0), capacity(s)), 3)
    on(be, gather_flux!, o..., E..., s, shape, st, grid; dt = T(dt), workgroupsize)
    h = Tuple(cellsize(grid))
    factor = ntuple(d -> space[d] ? Float64(h[d] / dt) : 1.0, 3)
    return (; grid, s, J, E, o, quantities, factor, N, P = order(shape), T)
end

alive_slots(s) = [i for i in 1:capacity(s) if s.status[i] == ParticleMeshCore.ALIVE]

"Per component `d`, `(⟨J_d, E_d⟩ over the interior, Σ_i quantity_d(i)·o_d[i] over the alive particles, scale)`, the scale as in `transpose_mismatch`."
function transpose_sums(c)
    (; grid, s, J, E, o, quantities, factor, N, P) = c
    alive = alive_slots(s)
    I = interior_indices(grid)
    return ntuple(3) do d
        lhs = sum(J[d][k] * E[d][k] for k in I)
        rhs = sum(quantities[d](s, i) * o[d][i] for i in alive)
        scale = maximum(absmax, E[d]) * sum(i -> absmax(quantities[d](s, i)), alive) * factor[d] * (P + 2)^N
        (lhs, rhs, scale)
    end
end

"""
    transpose_mismatch(c) -> NTuple{3,Float64}

Per component `d`, |⟨J_d, E_d⟩ over the interior − Σ_i quantity_d(i)·o_d[i]| over the alive particles,
in units of `eps(T)` times the scale `max|E_d| · Σ_i |quantity_d(i)| · factor_d · (P + 2)^N`, which
bounds the sum of the absolute terms of either side (a particle's weights on a component are at
most one per node, on at most `(P + 2)^N` nodes).
"""
transpose_mismatch(c) = map(((l, r, sc),) -> absdiff(l, r) / (base_eps(c.T) * sc), transpose_sums(c))

function body_T1000(be)
    C = tolC("T-1000")
    for N in 1:3, T in TYPES, shape in SHAPES, st in gather_staggers(N), mode in (Fast(), Reference())
        for bins in (mode isa Reference ? (CellBins(), NodeBins(shape, st)) : (CellBins(),))
            c = transpose_case(T, N, mode, shape, st; bins, be)
            r = record!("T-1000", T, maximum(transpose_mismatch(c)), be)
            @test r <= C
            # Dead slots keep what they held.
            @test all(i -> all(o -> same_bits([o[i]], [gval(T, 7.0)]), c.o), (i for i in 1:capacity(c.s) if c.s.status[i] != ParticleMeshCore.ALIVE))
        end
        # Control, as T-980's: the deposit at another stagger of axis 1 is not the gather's transpose.
        # Measured against the sum itself, since the roundoff scale is loose and Float32's eps large.
        lhs, rhs, _ = transpose_sums(transpose_case(T, N, Reference(), shape, st; be, dep_st = flipped_axis1(st)))[1]
        @test absdiff(lhs, rhs) > absmax(lhs) / 100
    end
end

"`st` with axis 1's stagger exchanged (Node ↔ Center): a wrong partner for T-1000's control."
flipped_axis1(st::ParticleMeshCore.Stagger) = st isa Node ? Center() : Node()
flipped_axis1(st::Tuple) = (flipped_axis1(st[1]), Base.tail(st)...)

@testset "T-1000 gather_flux! is the transpose of deposit_flux! in both modes, N = 1, 2, 3 [A3, A7, A8, WP-071]" begin
    body_T1000(CPU())
    # The case moves particles across faces and seams on every axis.
    c = transpose_case(Float64, 3, Fast(), Quadratic, (Node(), Node(), Node()))
    s = c.s
    @test all(d -> count(i -> s.cell[d][i] != getfield(s.extras, Symbol(:old_cell_, d))[i], alive_slots(s)) > 100, 1:3)
    @test all(d -> count(i -> abs(s.cell[d][i] - getfield(s.extras, Symbol(:old_cell_, d))[i]) > 1, alive_slots(s)) > 5, 1:3)
end

# Host closed forms of the B-splines of order 0, 1 and 2 and the integrals of the first two.
bspline(::Val{0}, ξ) = abs(ξ) < 1 / 2 ? 1.0 : abs(ξ) == 1 / 2 ? 0.5 : 0.0
bspline(::Val{1}, ξ) = max(0.0, 1 - abs(ξ))
bspline(::Val{2}, ξ) = (a = abs(ξ); a < 1 / 2 ? 3 / 4 - a^2 : a < 3 / 2 ? (3 / 2 - a)^2 / 2 : 0.0)
bspline_integral(::Val{0}, ξ) = clamp(ξ + 1 / 2, 0.0, 1.0)
bspline_integral(::Val{1}, ξ) = ξ <= -1 ? 0.0 : ξ <= 0 ? (1 + ξ)^2 / 2 : ξ <= 1 ? 1 - (1 - ξ)^2 / 2 : 1.0

"The periodic difference `x − y` on `n` cells, in `[-n/2, n/2)`."
pdiff(x, y, n) = mod(x - y + n / 2, n) - n / 2
soff(::Node) = 0.0
soff(::Center) = 0.5

"Particle `i`'s old cell, old offset and move in cells along axis `d` (`d = 0` at N = 1), from the extras (the move unwrapped across the seam)."
function old_and_move_parts(s, i, d, n)
    c0, δ0 = d == 0 ? (s.extras.old_cell[i], s.extras.old_offset[i]) :
             (getfield(s.extras, Symbol(:old_cell_, d))[i], getfield(s.extras, Symbol(:old_offset_, d))[i])
    a = max(d, 1)
    e = mod(s.cell[a][i] - c0 + 1, n) - 1
    return (c0, Float64(value_of(δ0)), e + Float64(value_of(s.offset[a][i])) - Float64(value_of(δ0)))
end

"Particle `i`'s old position (cells) and its move along axis `d`."
old_and_move(s, i, d, n) = ((c0, δ0, Δ) = old_and_move_parts(s, i, d, n); (c0 + δ0, Δ))

function body_T1001()
    C = tolC("T-1001")
    for T in (Float32, Float64), shape in SHAPES, st in STAGGERS
        c = transpose_case(T, 1, Fast(), shape, st)
        (; grid, s, E, o) = c
        n, g, h = ncells(grid), guards(grid), Float64(cellsize(grid))
        dt = 0.25
        Emax = maximum(abs, E[1])
        err = maximum(alive_slots(s)) do i
            c0, δ0, Δ = old_and_move_parts(s, i, 0, n)
            ref = sum(1:n) do j
                ξ0 = pdiff(c0, j, n) + (δ0 - soff(st) - 1 / 2)     # the cell difference wrapped in integers, exact
                Float64(E[1][g + j]) * (bspline_integral(Val(order(shape) - 1), ξ0 + Δ) - bspline_integral(Val(order(shape) - 1), ξ0))
            end
            abs(Float64(o[1][i]) - h / dt * ref)
        end
        r = record!("T-1001", T, err / (base_eps(T) * h / 0.25 * Emax * (order(shape) + 1)))
        @test r <= C
    end
end

@testset "T-1001 at N = 1 the flux gather is the field integrated along the move with the shape one order lower at the faces [A2, A7, WP-071]" begin
    body_T1001()
end

"""
    nudged_case(N, shape, st, ε) -> NamedTuple

`transpose_case` at rest in Float64 (Fast), then every alive particle moved by `ε` times a factor
of magnitude 1/2 to 1 and alternating sign per axis, and the flux gather run again on the moved species.
"""
function nudged_case(N, shape, st, ε)
    c = transpose_case(Float64, N, Fast(), shape, st; maxstep = 0)
    (; grid, s) = c
    n = Int32.(Tuple(ncells(grid)))
    for (k, i) in enumerate(alive_slots(s))
        Δ = ntuple(d -> ε * (isodd(k + d) ? 1 : -1) * (1 / 2 + mod(0.618k + 0.31d, 1.0) / 2), N)
        if N == 1
            s.cell[1][i], s.offset[1][i] = advance(s.cell[1][i], s.offset[1][i], Δ[1], n[1])
        else
            cn, δn = advance(ntuple(d -> s.cell[d][i], N), ntuple(d -> s.offset[d][i], N), Δ, n)
            foreach(d -> (s.cell[d][i] = cn[d]; s.offset[d][i] = δn[d]), 1:N)
        end
    end
    o = ntuple(_ -> zeros(capacity(s)), 3)
    gather_flux!(o..., c.E..., s, shape, st, grid; dt = 0.25)
    return (; c..., o)
end

"""
    galerkin_point(E, grid, X, st, C, space) -> Float64

The host closed form of the energy-conserving (Galerkin) gather that `gather_flux!` tends to for a
vanishing move: field component `C` at the position `X` (cells, per axis), read with the quadratic
shape on every axis but `C`'s own, where it is the linear shape one order lower centred on the
faces half a cell past ρ's nodes, if the component is in space (`space`); with the quadratic shape
on every axis if it is not.
"""
function galerkin_point(E, grid, X, st, C, space)
    n, g = Tuple(ncells(grid)), guards(grid)
    N = length(n)
    sts = st isa Tuple ? st : (st,)
    acc = 0.0
    for J in CartesianIndices(n)
        w = prod(1:N) do d
            own = space && d == C
            ξ = pdiff(X[d], J[d] + soff(sts[d]) + (own ? 1 / 2 : 0.0), n[d])
            own ? bspline(Val(1), ξ) : bspline(Val(2), ξ)
        end
        w == 0 || (acc += w * Float64(E[CartesianIndex(Tuple(J) .+ g)]))
    end
    return acc
end

"The largest |o_C/(velocity) − galerkin_point| over the alive particles and the components, over max|E|, at the move's midpoint."
function galerkin_gap(c, st)
    (; grid, s, E, o, N) = c
    n = Tuple(ncells(grid))
    gap = 0.0
    for i in alive_slots(s), C in 1:3
        space = N == 3 || C <= N
        om = ntuple(d -> old_and_move(s, i, N == 1 ? 0 : d, n[d]), N)
        X = ntuple(d -> om[d][1] + om[d][2] / 2, N)
        avg = space ? o[C][i] / (om[C][2] * Float64(Tuple(cellsize(grid))[C]) / 0.25) : o[C][i]
        gap = max(gap, abs(avg - galerkin_point(E[C], grid, X, st, C, space)) / maximum(abs, E[C]))
    end
    return gap
end

@testset "T-1002 for a vanishing move the flux gather is the energy-conserving gather, order 2, N = 1, 2, 3 [A2, A7, WP-071]" begin
    C = tolC("T-1002")
    for N in 1:3, st in gather_staggers(N)
        g4, g5 = galerkin_gap(nudged_case(N, Quadratic, st, 1e-4), st), galerkin_gap(nudged_case(N, Quadratic, st, 1e-5), st)
        record!("T-1002", Float64, max(g4 / 1e-4, g5 / 1e-5))
        @test g4 <= C * 1e-4 && g5 <= C * 1e-5
        @test g5 <= g4 / 5            # at least linear convergence
    end
end

function body_T1003(be)
    C = tolC("T-1003")
    for N in 1:3, T in TYPES, shape in SHAPES, st in gather_staggers(N)
        c = transpose_case(T, N, Fast(), shape, st; maxstep = 0, be)
        (; s, E, o, grid) = c
        alive = alive_slots(s)
        # At rest every component in space is exactly zero (a signed zero), partials included.
        inspace = N == 1 ? (1,) : N == 2 ? (1, 2) : (1, 2, 3)
        @test all(d -> all(i -> all(iszero, components(o[d][i])), alive), inspace)
        # Out of space, at rest, the flux gather is the plain gather with the same shape: the same
        # bits at N = 1 (the frame's extra node has weight zero), to roundoff at N = 2.
        for d in setdiff(1:3, inspace)
            p = on(be, gather!, fill(gval(T, 7.0), capacity(s)), E[d], s, shape, st, grid)
            if N == 1
                @test same_bits(o[d][alive], p[alive])
            else
                r = record!("T-1003", T, maximum(i -> absdiff(o[d][i], p[i]), alive) / (base_eps(T) * maximum(absmax, E[d])), be)
                @test r <= C
            end
        end
        # The bits do not depend on the launch shape.
        m = transpose_case(T, N, Fast(), shape, st; be)
        for wg in (1, 7)
            w = transpose_case(T, N, Fast(), shape, st; be, workgroupsize = wg)
            @test all(d -> same_bits(m.o[d], w.o[d]), 1:3)
        end
    end
end

@testset "T-1003 gather_flux!: zero in space at rest, the plain gather out of it, launch shapes bitwise, argument checks [A4, A7, A8, WP-071]" begin
    body_T1003(CPU())
    st = Node()
    o = ntuple(_ -> zeros(53), 3)
    grid = PeriodicGrid(10, 0.5, 2)                 # one guard too few for order 2 at Node()
    s = moved_species(Float64, grid, 50; seed = 1)
    E = ntuple(_ -> field(backend, Float64, grid), 3)
    @test_throws ArgumentError gather_flux!(o..., E..., s, Quadratic, st, grid; dt = 0.25)
    grid = PeriodicGrid(10, 0.5, 3)
    E = ntuple(_ -> field(backend, Float64, grid), 3)
    @test_throws ArgumentError gather_flux!(o..., E..., test_species(Float64, grid, 50), Quadratic, st, grid; dt = 0.25)   # no extras
    @test_throws ArgumentError gather_flux!(o..., E..., moved_species(Float64, grid, 50; seed = 1), Quadratic, (st,), grid; dt = 0.25)
    @test gather_flux!(o..., E..., moved_species(Float64, grid, 50; seed = 1), Quadratic, st, grid; dt = 0.25) === o
    g2 = grid2(Float64, 3)
    s2 = flux_species_2d(Float64, g2, 20)
    E2 = ntuple(_ -> field(backend, Float64, g2), 3)
    @test_throws ArgumentError gather_flux!(ntuple(_ -> zeros(capacity(s2)), 3)..., E2..., s2, Linear, st, g2; dt = 0.25)
    @test gather_flux!(ntuple(_ -> zeros(capacity(s2)), 3)..., E2..., s2, Linear, (st, st), g2; dt = 0.25) isa Tuple
end

# --- T-1005 (WP-070's cross-test, landed after WP-071) ---------------------------------------------

"""
    midpoint_species(s, n) -> Species

A copy of the flux tests' species `s` with every alive particle at the midpoint of its move: its
old cell and offset (the extras) advanced by half the move.
"""
function midpoint_species(s, n)
    N = length(n)
    m = deepcopy(s)
    for i in alive_slots(s)
        if N == 1
            c0, δ0, Δ = old_and_move_parts(s, i, 0, n[1])
            m.cell[1][i], m.offset[1][i] = advance(Int32(c0), Float64(δ0), Δ / 2, Int32(n[1]))
        else
            p = ntuple(d -> old_and_move_parts(s, i, d, n[d]), N)
            cn, δn = advance(ntuple(d -> Int32(p[d][1]), N), ntuple(d -> Float64(p[d][2]), N), ntuple(d -> p[d][3] / 2, N), Int32.(n))
            foreach(d -> (m.cell[d][i] = cn[d]; m.offset[d][i] = δn[d]), 1:N)
        end
    end
    return m
end

# The flux gather and the Galerkin gather of WP-070 are the same weights for a vanishing move. For
# order 2, every ρ stagger the flux tests use and every component C in space whose axis has ρ at
# `Node()` (there the current's face j is the field's `Center()` index j; at a `Center()` ρ the face
# is the next node, a shift of the array, which T-1002's closed form covers instead), the engine's
# per-axis `gather!` at the move's midpoint, with the linear shape at `Center()` along C and the
# quadratic shape at ρ's stagger across it, is (1) T-1002's host closed form `galerkin_point` to
# roundoff (of a position of up to n cells, which the closed form holds as one number) and (2) the flux gather's o_C/(Δ_C·Δx_C/Δt) to within C·ε at moves ε = 1e-4 and 1e-5.
@testset "T-1005 for a vanishing move the flux gather of WP-071 reproduces the per-axis gather! of WP-070's Galerkin weights, order 2, N = 1, 2, 3 [A2, A7, WP-070, WP-071]" begin
    C = tolC("T-1005")
    worst_gap, worst_point, cases = 0.0, 0.0, 0
    for N in 1:3, st in gather_staggers(N)
        sts = st isa Tuple ? st : (st,)
        for ε in (1e-4, 1e-5)
            c = nudged_case(N, Quadratic, st, ε)
            (; grid, s, E, o) = c
            n = Tuple(ncells(grid))
            mid = midpoint_species(s, n)
            for Cc in 1:N
                sts[Cc] isa Node || continue
                stc = ntuple(d -> d == Cc ? Center() : sts[d], N)
                shp = ntuple(d -> d == Cc ? Linear : Quadratic, N)
                out = zeros(capacity(s))
                N == 1 ? gather!(out, E[Cc], mid, Linear, Center(), grid) : gather!(out, E[Cc], mid, shp, stc, grid)
                Emax = maximum(abs, E[Cc])
                for i in alive_slots(s)
                    _, _, Δ = old_and_move_parts(s, i, N == 1 ? 0 : Cc, n[Cc])
                    X = ntuple(d -> ((c0, δ0, Δd) = old_and_move_parts(s, i, N == 1 ? 0 : d, n[d]); c0 + δ0 + Δd / 2), N)
                    avg = o[Cc][i] / (Δ * Float64(Tuple(cellsize(grid))[Cc]) / 0.25)
                    worst_point = max(worst_point, abs(out[i] - galerkin_point(E[Cc], grid, X, st, Cc, true)) / (eps(Float64) * Emax * maximum(n)))
                    worst_gap = max(worst_gap, abs(avg - out[i]) / Emax / ε)
                end
                cases += 1
            end
        end
    end
    record!("T-1005", Float64, worst_gap)
    @info "T-1005 flux gather against the per-axis Galerkin gather!" worst_gap worst_point cases
    @test cases > 0
    @test worst_point <= 4          # the same weights; the closed form takes the position as one number up to n cells
    @test worst_gap <= C
end
