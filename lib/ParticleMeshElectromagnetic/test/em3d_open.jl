# WP-072: the open transverse boundaries of the 3D step, T-1010 to T-1018 (T-1019 is their rerun
# on CUDA, in gpu.jl). Each open axis (y, z or both) carries its own absorber (`AbsorbingLayer`
# or `PML`, the same kernels as at N = 2, placed per axis), and the particles that cross an open
# face are retired through the alive mask. The 2D theory of the layers (T-450, T-540) carries
# over through T-1010: a 3D run that does not vary along a periodic axis is the 2D open run. The
# plasma runs are `open_plasma_3d` (em3d_cases.jl), whose particles leave through every open
# face. After em3d_step.jl, whose helpers (`invariant_error_3d`, the worker) this file reuses.

using ParticleMeshElectromagnetic: LayerFactors, PMLFields, pml_b!, pml_e!, place_absorber, open_absorbers

"The number of alive particles of every species of `st`."
alive_3d(st) = sum(s -> count(==(ParticleMeshCore.ALIVE), Array(s.status)), st.species)

"The interior rows along axis `d` of a state whose axes in `opens` (axis => cells) are open: the node rows off the layers, and the centre rows likewise (T-451's rule per axis)."
function free_rows(g, n, d, opens)
    c = get(opens, d, 0)
    c == 0 && return ((g + 1):(g + n[d]), (g + 1):(g + n[d]))
    return ((g + 1 + c + 1):(g + 1 + n[d] - c - 1), (g + 1 + c):(g + 1 + n[d] - c - 1))
end

"""
    open_residuals_3d(st, ρ̄₀) -> (; free, layer, scale, divB, bscale, ρbar)

Gauss's law `Σ_d D⁻_d E_d − (ρ − ρ̄₀)` of the 3D layout on the nodes whose rows are off every
open axis' layer (`free`) and its maximum on the other nodes (`layer`), ρ deposited now at the
primal nodes (the species sorted first in Reference mode, since retired particles stay in their
segments until the next sort), `scale = (n₁ + n₂ + n₃)·Σ_s max|ρ_s|` (T-950's species scale);
∇·B on the cell centres off every layer with T-950's `bscale`; `ρbar` the current mean of ρ.
"""
function open_residuals_3d(st, ρbar0)
    (; grid, numerics) = st
    T = eltype(st.ρ)
    g, n, h = guards(grid), ncells(grid), cellsize(grid)
    opens = Dict(d => open_absorber(st, d).cells for d in 2:3 if open_absorber(st, d) !== nothing)
    I = CartesianIndices(ntuple(d -> (g + 1):(g + n[d]), 3))
    free = CartesianIndices(ntuple(d -> free_rows(g, n, d, opens)[1], 3))
    cfree = CartesianIndices(ntuple(d -> free_rows(g, n, d, opens)[2], 3))
    e = ntuple(d -> CartesianIndex(ntuple(a -> Int(a == d), 3)), 3)
    ρ = zeros(T, size(st.ρ))
    species_scale = 0.0
    for (s, ws) in zip(st.species, st.ws)
        numerics.mode isa Reference && sort_by_cell!(s, grid, ws)
        ρs = field(get_backend(st.ρ), T, grid)
        deposit!(numerics.mode, ρs, s, numerics.shape, (Node(), Node(), Node()), grid, ws)
        fold_guards!(ρs, grid)
        a = Array(ρs)
        species_scale += Float64(maximum(absmax, a[I]))
        ρ .+= a
    end
    E = map(Array, (st.fields.Ex, st.fields.Ey, st.fields.Ez))
    B = map(Array, (st.fields.Bx, st.fields.By, st.fields.Bz))
    gauss(j) = absmax(sum(d -> (E[d][j] - E[d][j - e[d]]) / h[d], 1:3) - (ρ[j] - ρbar0))
    res = maximum(gauss, free)
    resl = maximum(j -> j in free ? 0.0 : Float64(gauss(j)), I)
    divB = maximum(absmax(sum(d -> (B[d][j + e[d]] - B[d][j]) / h[d], 1:3)) for j in cfree)
    bmax = maximum(b -> maximum(absmax, b[cfree]), B)
    hv = map(value_of, h)
    return (; free = res, layer = resl, scale = sum(n) * species_scale, divB, bscale = sum(d -> 2 / hv[d], 1:3) * bmax,
            ρbar = sum(ρ[I]) / prod(n))
end

# --- T-1010 ---------------------------------------------------------------------------------

# The 3D step with an open axis is the 2D open step where the problem does not vary along the
# third, periodic axis: T-953's cold counter-streaming beams on a plane of 12 × 16 cells, open
# along the plane's second axis with each absorber of OPEN3_ABSORBERS, and the same problem in 3D
# invariant along z (the open axis y) or along x (the plane's axes on y and z, the open axis z).
# The absorbers are placed per axis from the axis' cells and cell size alone, so the 3D layer
# multiplies by the 2D factors and the 3D PML advances its convolution fields with the 2D
# coefficients; the particles that cross the open edge leave on every line of the invariant axis
# at once (the alive count stays 12 times the 2D one: 6 cells and 2 particles per cell along the
# invariant axis). Compared as T-953 after initialize! and every step, over 30 steps in which the
# beams' charge reaches the layers and leaves.
const OPEN3_PLANE = (cells = (12, 16), len = (2, 2), boundaries = (:periodic, :open))

function body_T1010(be)
    C = emC("T-1010")
    for T in TYPES, mode in (Reference(), Fast()), axis in (3, 1), (label, ab) in OPEN3_ABSORBERS
        amplitude = seeded(T, 1 / 20)
        p2, p3 = beams_plane_2d(T; amplitude, nsteps = 30, OPEN3_PLANE...), beams_3d(T, axis; amplitude, nsteps = 30, OPEN3_PLANE...)
        s2 = EMState(p2, EMNumerics(; mode, absorber = ab); backend = be)
        s3 = EMState(p3, EMNumerics(; mode, absorber = ab); backend = be)
        @test p3.boundaries[inv3_axes(axis)[2]] === :open && open_absorber(s3, inv3_axes(axis)[2]) isa Union{LayerFactors,PMLFields}
        initialize!(s2, p2)
        initialize!(s3, p3)
        n0 = alive_3d(s2)
        @test alive_3d(s3) == 12 * n0
        scale = sum(ncells(s2.grid)) * sqrt(prod(p2.species[1].ppc))
        worst = invariant_error_3d(s2, s3, axis) / (base_eps(T) * scale)
        for k in 1:p2.nsteps
            step!(s2, params(p2), k)
            step!(s3, params(p3), k)
            worst = max(worst, invariant_error_3d(s2, s3, axis) / (base_eps(T) * k * scale))
            @test alive_3d(s3) == 12 * alive_3d(s2)
        end
        @test alive_3d(s2) < n0                   # particles left through the open edges
        @test all(c -> maximum(absmax, Array(getfield(s2.fields, c))) > 0, keys(invariant_map_3d(axis)))
        record!("T-1010 $label axis $axis $(nameof(typeof(mode)))", T, worst, be)
        @test worst <= C
    end
end

@testset "T-1010 a 3D run with an open axis, invariant along a periodic one, is the 2D open run, each absorber, y and z open [A2, A7, A8, WP-072]" begin
    body_T1010(CPU())
end

# --- T-1011 ---------------------------------------------------------------------------------

# Gauss's law with particles leaving through the open faces of y and z at once, as T-451 and
# T-543 at N = 2: each absorber touches E only on its own axis' layer rows (the layer damps every
# component there, the PML corrects the two components across its axis), so on the nodes whose
# rows are off every layer D⁻·E = ρ − ρ̄₀ to roundoff after every step, ρ̄₀ the mean charge at t = 0,
# on T-950's species scale; ∇·B likewise on the centres off every layer. The edges where the two
# layers overlap belong to the layers. Both are far from roundoff in the layers, which shows the
# check sees them.
function body_T1011(be)
    C, CB = emC("T-1011"), emC("T-1011", "C_divB")
    for T in TYPES, mode in (Reference(), Fast()), (label, ab) in OPEN3_ABSORBERS
        prob = open_plasma_3d(T; amplitude = seeded(T, 1 / 10))
        st = open_state_3d(prob, ab; mode, be)
        ρbar0 = open_residuals_3d(st, zero(T)).ρbar
        r = open_residuals_3d(st, ρbar0)
        worst = r.free / (base_eps(T) * r.scale)
        worstB = worst_layer = 0.0
        n0 = alive_3d(st)
        p = params(prob)
        for k in 1:prob.nsteps
            step!(st, p, k)
            r = open_residuals_3d(st, ρbar0)
            worst = max(worst, r.free / (base_eps(T) * r.scale * sqrt(k + 1)))
            r.bscale > 0 && (worstB = max(worstB, r.divB / (base_eps(T) * r.bscale * sqrt(k))))
            worst_layer = max(worst_layer, Float64(r.layer / (base_eps(T) * r.scale * sqrt(k + 1))))
        end
        left = n0 - alive_3d(st)
        @info "T-1011" label T mode worst worstB worst_layer left of = n0
        @test left > n0 ÷ 10                       # particles left through the open faces
        @test worst_layer > 1e4                    # the layers are not charge conserving
        record!("T-1011 $label $(nameof(typeof(mode)))", T, worst, be)
        record!("T-1011 divB $label $(nameof(typeof(mode)))", T, worstB, be)
        @test worst <= C
        @test worstB <= CB
    end
end

@testset "T-1011 Gauss's law and ∇·B to roundoff off the layers with particles leaving through y and z, each absorber, all three types, both modes [A2, A7, A8, WP-072]" begin
    body_T1011(CPU())
end

# --- T-1012 ---------------------------------------------------------------------------------

# The retirement, as T-452: exactly the particles whose cell along an open axis jumped across
# its seam in this step are marked free, the particles that crossed only periodic seams stay, the
# alive charge falls by the retired particles' charge; for y open, z open and both. Fast mode in
# Float64, which does not permute the slots. The current of the last move is deposited before
# the retirement: the first step's J is the periodic box's bit for bit (Reference mode, each type).
@testset "T-1012 particles that cross an open face are retired after their current is deposited, the periodic seams keep them [A2, WP-072]" begin
    for bd in ((:periodic, :open, :periodic), (:periodic, :periodic, :open), (:periodic, :open, :open))
        opens = filter(d -> bd[d] === :open, (2, 3))
        prob = open_plasma_3d(Float64; boundaries = bd)
        st = open_state_3d(prob, OPEN3_ABSORBERS[1][2]; mode = Fast())
        p = params(prob)
        retired = crossed = 0
        for k in 1:prob.nsteps
            before = map(s -> (status = Array(s.status), c = map(Array, s.cell), qw = s.q .* Array(s.w)), st.species)
            step!(st, p, k)
            for (s, b) in zip(st.species, before)
                status, c = Array(s.status), map(Array, s.cell)
                was = b.status .== ParticleMeshCore.ALIVE
                wrap = ntuple(d -> was .& (abs.(c[d] .- b.c[d]) .> 1), 3)
                wrap_open = reduce(.|, map(d -> wrap[d], opens))
                wrap_periodic = reduce(.|, map(d -> wrap[d], filter(∉(opens), (1, 2, 3))))
                @test (was .& (status .!= ParticleMeshCore.ALIVE)) == wrap_open
                @test all(status[wrap_periodic .& .!wrap_open] .== ParticleMeshCore.ALIVE)
                @test sum(b.qw[was .& (status .== ParticleMeshCore.ALIVE)]) ≈ sum(b.qw[was]) - sum(b.qw[wrap_open])
                retired += count(wrap_open)
                crossed += count(wrap_periodic .& .!wrap_open)
            end
        end
        @info "T-1012" bd retired crossed
        @test retired > 50 && crossed > 50
        for T in TYPES
            po = open_plasma_3d(T; boundaries = bd)
            pp = EMProblem(po.name, po.length, po.cells, po.dt, po.nsteps, po.species, po.seed, po.kmode)
            so, sp = open_state_3d(po, OPEN3_ABSORBERS[2][2]), state_3d(pp)
            step!(so, params(po), 1)
            step!(sp, params(pp), 1)
            @test all(c -> bits(Array(getfield(so.fields, c))) == bits(Array(getfield(sp.fields, c))), (:Jx, :Jy, :Jz))
            @test alive_3d(so) < alive_3d(sp)
            @test Array(so.fields.Ey) != Array(sp.fields.Ey)
        end
    end
end

# --- T-1013 ---------------------------------------------------------------------------------

# With a zero profile (a layer of zero strength; a PML of σ = α = 0, κ = 1) the open box is the
# periodic one bit for bit: random vacuum fields through 20 steps with y and z open. A real
# absorber on z is the one on y with the axes exchanged: its factors (layer) or coefficients
# (PML) are the y absorber's of the grid whose y and z are swapped, bit for bit, its
# convolution fields have 2·cells rows along z and are named by the components that cross z
# (Bx, By, Ex, Ey); each PML correction changes only those components, only on the z layer
# rows; and after a solve with both axes' absorbers every guard is its interior image.
@testset "T-1013 zero-profile absorbers are the periodic 3D step bit for bit; the z absorber is the y one with the axes exchanged [A2, A4, A8, WP-072]" begin
    zero_layer, zero_pml = AbsorbingLayer(; cells = 4, strength = 0), PML(; cells = 4, sigma_max = 0, kappa_max = 1, alpha_max = 0)
    for T in TYPES, ab in (zero_layer, zero_pml)
        L, n = (2 * T(π), 3 * T(π), 5 * T(π) / 2), (8, 14, 12)
        dt = 9 * inv(sqrt(sum(d -> (n[d] / L[d])^2, 1:3))) / 10
        po = EMProblem("vac-open", L, n, T(dt), 20, (), 1, (1, 1, 1); boundaries = (:periodic, :open, :open))
        pp = EMProblem("vac", L, n, T(dt), 20, (), 1, (1, 1, 1))
        so = EMState(po, EMNumerics(; current = NoCurrent(), absorber = ab))
        sp = EMState(pp, EMNumerics(; current = NoCurrent()))
        for (s, pr) in ((so, po), (sp, pp))
            initialize!(s, pr)
            foreach(a -> (a .= T.(randn(Xoshiro(5), size(a))); fill_guards!(a, s.grid)), values(s.fields))
        end
        run!(so, params(po), po.nsteps)
        run!(sp, params(pp), pp.nsteps)
        @test all(c -> bits(Array(getfield(so.fields, c))) == bits(Array(getfield(sp.fields, c))), keys(so.fields))
    end
    T = Float64
    n, h = (8, 14, 12), (T(1) / 2, T(2) / 5, T(1) / 3)
    grid, swapped = PeriodicGrid(n, h, 3), PeriodicGrid((n[1], n[3], n[2]), (h[1], h[3], h[2]), 3)
    dt = T(1) / 10
    for ab in OPEN3_ABSORBERS
        Lz = place_absorber(ab[2], grid, dt; axis = 3, backend = CPU())
        Ly = place_absorber(ab[2], swapped, dt; axis = 2, backend = CPU())
        if Lz isa LayerFactors
            @test all(f -> bits(Array(getfield(Lz, f))) == bits(Array(getfield(Ly, f))), (:node_half, :center_half, :node_full, :center_full))
        else
            @test all(k -> bits(Array(Lz.node[k])) == bits(Array(Ly.node[k])) && bits(Array(Lz.center[k])) == bits(Array(Ly.center[k])), 1:3)
            @test keys(Lz.ψ) == (:Bx, :By, :Ex, :Ey) && keys(Ly.ψ) == (:Bx, :Bz, :Ex, :Ez)
            @test all(a -> size(a) == (n[1], n[2], 8) && all(iszero, Array(a)), values(Lz.ψ))
            @test Lz.sigma_max == Ly.sigma_max
        end
    end
    # each PML correction of the z axis on its own: B changes Bx, By, E changes Ex, Ey, each on the
    # z layer rows only, every other entry and component bit for bit
    prob = open_plasma_3d(T; boundaries = (:periodic, :open, :open))
    st = open_state_3d(prob, OPEN3_ABSORBERS[2][2])
    foreach(a -> (a .= randn(Xoshiro(9), size(a)); fill_guards!(a, st.grid)), values(st.fields))
    Lz = open_absorber(st, 3)
    g, nz = guards(st.grid), ncells(st.grid)[3]
    layer_z = vcat((g + 1):(g + 4), (g + nz - 3):(g + nz))
    off_z = setdiff(axes(st.fields.Ex, 3), vcat(layer_z, 1:g, (g + nz + 1):(nz + 2g)))
    for (fn, changed) in ((pml_b!, (:Bx, :By)), (pml_e!, (:Ex, :Ey)))
        Lc = PMLFields(OPEN3_ABSORBERS[2][2], st.grid, prob.dt; axis = 3, backend = CPU())
        foreach(a -> (a .= randn(Xoshiro(13), size(a))), values(Lc.ψ))
        f = map(copy, st.fields)
        fn === pml_b! ? fn(Lc, f, st.grid, prob.dt / 2, true, nothing) : fn(Lc, f, st.grid, prob.dt, nothing)
        for c in keys(f)
            a, b = Array(getfield(f, c)), Array(getfield(st.fields, c))
            if c in changed
                @test all(k -> a[:, :, k] != b[:, :, k], layer_z)
                @test bits(a[:, :, off_z]) == bits(b[:, :, off_z])
            else
                @test bits(a) == bits(b)
            end
        end
    end
    # every guard is its interior image after a solve with both axes' absorbers
    for (_, ab) in OPEN3_ABSORBERS
        sl = open_state_3d(open_plasma_3d(T), ab)
        foreach(a -> (a .= randn(Xoshiro(7), size(a)); fill_guards!(a, sl.grid)), values(sl.fields))
        solve_fields!(Yee(), sl.fields, sl.grid, prob.dt, sl.modifiers, nothing, 1)
        refilled = map(a -> fill_guards!(copy(a), sl.grid), sl.fields)
        @test all(c -> bits(Array(getfield(sl.fields, c))) == bits(Array(getfield(refilled, c))), keys(sl.fields))
    end
end

# --- T-1014 ---------------------------------------------------------------------------------

@testset "T-1014 the open 3D step is bitwise across reruns, launch shapes and 1 and 4 threads in Reference mode [A4, WP-072]" begin
    ref = open3d_step_digests()
    @test length(ref) == 2 * length(TYPES) && allunique(last.(ref))
    @test open3d_step_digests() == ref
    for wgs in (32, 256)
        @test open3d_step_digests(; workgroupsize = wgs) == ref
    end
    for nt in (1, 4)
        fields = split.(em3d_worker_lines(nt, "EM3DO "))
        got = Dict((String(f[3]), String(f[4])) => parse(UInt64, f[5]) for f in fields)
        @test all(f -> parse(Int, f[2]) == nt, fields)
        @test length(got) == length(ref)
        for (label, T, d) in ref
            @test get(got, (label, type_label(T)), nothing) == d
        end
    end
end

# --- T-1015 ---------------------------------------------------------------------------------

# A vacuum pulse into every face and corner: B = ∇×(x̂·a) of a Gaussian a of width 2 cells at the
# centre of a box of 8 × 40 × 40 cells, open along y and z with 8 cells of each absorber, E = 0.
# It has no Fourier component with k_y = k_z = 0, so its energy leaves through the open faces,
# the corners and edges included (the periodic box keeps it, which shows the check sees the
# absorbers). The field energy on the cells off the layers falls below
# a fraction of its start once the pulse has crossed the box, and over a long run (2000 steps,
# 25 crossings of the box at c) it does not grow again: the corners where the two absorbers
# overlap stay stable.
function corner_pulse(::Type{T}, ab; nsteps) where {T}
    n = (8, 40, 40)
    h = (T(1) / 2, T(1) / 2, T(1) / 2)
    dt = 9 * inv(sqrt(sum(d -> inv(h[d])^2, 1:3))) / 10
    bd = ab === nothing ? (:periodic, :periodic, :periodic) : (:periodic, :open, :open)
    prob = EMProblem("corner-pulse", n .* h, n, T(dt), nsteps, (), 1, (1, 1, 1); boundaries = bd)
    st = EMState(prob, EMNumerics(; current = NoCurrent(), absorber = ab))
    initialize!(st, prob)
    g = guards(st.grid)
    w = 2 * h[2]
    c = (n[2] * h[2] / 2, n[3] * h[3] / 2)
    a(y, z) = exp(-((y - c[1])^2 + (z - c[2])^2) / w^2)
    By, Bz = Array(st.fields.By), Array(st.fields.Bz)
    for I in CartesianIndices(By)
        j, k = I[2] - g - 1, I[3] - g - 1
        # By = ∂z a on (Center, Node, Center), Bz = −∂y a on (Center, Center, Node), as differences of a
        y, z = j * h[2], (k + 1 / 2) * h[3]
        By[I] = (a(y, z + h[3] / 2) - a(y, z - h[3] / 2)) / h[3]
        y, z = (j + 1 / 2) * h[2], k * h[3]
        Bz[I] = -(a(y + h[2] / 2, z) - a(y - h[2] / 2, z)) / h[2]
    end
    copyto!(st.fields.By, By)
    copyto!(st.fields.Bz, Bz)
    foreach(f -> fill_guards!(f, st.grid), values(st.fields))
    return prob, st
end

"The field energy on the cells whose rows are off the 8-cell layers of y and z (on every interior cell with `edge = 0`)."
function corner_energy(st; edge = 8)
    g, n = guards(st.grid), ncells(st.grid)
    I = (g + 1):(g + n[1]), (g + 1 + edge):(g + n[2] - edge), (g + 1 + edge):(g + n[3] - edge)
    return sum(c -> sum(abs2, Array(getfield(st.fields, c))[I...]), (:Ex, :Ey, :Ez, :Bx, :By, :Bz))
end

@testset "T-1015 a vacuum pulse leaves through the faces, edges and corners of y and z, and the energy stays down over a long run [A2, A7, WP-072]" begin
    bound = emC("T-1015", "fraction")
    for (label, ab) in (("layer", AbsorbingLayer(; cells = 8, strength = 4)), ("pml", PML(; cells = 8, kappa_max = 2, alpha_max = 0.1)),
                        ("periodic", nothing))
        prob, st = corner_pulse(Float64, ab; nsteps = 2000)
        p = params(prob)
        W0, Wbox = corner_energy(st), corner_energy(st; edge = 0)
        Ws = Float64[]
        for k in 1:prob.nsteps
            step!(st, p, k)
            k % 50 == 0 && push!(Ws, corner_energy(st) / W0)
        end
        half = length(Ws) ÷ 2
        @info "T-1015" label box = corner_energy(st; edge = 0) / Wbox at_250 = Ws[5] at_1000 = Ws[half] at_2000 = Ws[end] late_max = maximum(Ws[half:end])
        if ab === nothing
            @test corner_energy(st; edge = 0) / Wbox > 1 / 2     # the periodic box keeps the pulse's energy
        else
            record!("T-1015 $label", Float64, Ws[end])
            @test Ws[end] <= bound
            @test maximum(Ws[half:end]) <= max(2 * Ws[half], bound)   # no late growth (corner instability)
        end
    end
end

# --- T-1016 ---------------------------------------------------------------------------------

# Dual{Float32,3} through 10 open 3D steps with each absorber, as T-955: the primal against the
# Float32 run, the partials ∂/∂(amplitude) finite, nonzero, the unseeded directions zero, and the
# seeded one against the central difference of Float64 runs at amplitude 1/10 ± 10⁻⁴; and the
# convolution fields' partials likewise.
@testset "T-1016 Dual through the open 3D step, each absorber [A3, A8, WP-072]" begin
    C, CP = emC("T-1016"), emC("T-1016", "C_partial")
    a = mkdual(1 / 10, 1, 0, 0)
    for (label, ab) in OPEN3_ABSORBERS
        pd, pf = open_plasma_3d(D3; amplitude = a, nsteps = 10), open_plasma_3d(Float32; nsteps = 10)
        sd, sf = open_state_3d(pd, ab), open_state_3d(pf, ab)
        run!(sd, params(pd), pd.nsteps)
        run!(sf, params(pf), pf.nsteps)
        δ = 1e-4
        fd = map((1 / 10 + δ, 1 / 10 - δ)) do amp
            p = open_plasma_3d(Float64; amplitude = amp, nsteps = 10)
            s = open_state_3d(p, ab)
            run!(s, params(p), p.nsteps)
            s
        end
        arrays(s) = (map(c -> getfield(s.fields, c), (:Ex, :Ey, :Ez, :Bx, :By, :Bz))..., evolved_arrays(open_absorber(s, 2))..., evolved_arrays(open_absorber(s, 3))...)
        worst, worst_partial = 0.0, 0.0
        for (A, B, P, M) in zip(map(Array, arrays(sd)), map(Array, arrays(sf)), map(Array, arrays(fd[1])), map(Array, arrays(fd[2])))
            worst = max(worst, maximum(abs, value_of.(A) .- B) / (eps(Float32) * maximum(abs, B) * pd.nsteps))
            @test all(x -> all(isfinite, partials(x)), A)
            @test any(x -> !iszero(partials(x)[1]), A)
            @test all(x -> iszero(partials(x)[2]) && iszero(partials(x)[3]), A)
            slope = (P .- M) ./ (2δ)
            worst_partial = max(worst_partial, maximum(abs, map(x -> Float64(partials(x)[1]), A) .- slope) /
                                               (eps(Float32) * maximum(abs, slope) * pd.nsteps))
        end
        @info "T-1016" label worst worst_partial
        record!("T-1016 $label", D3, worst)
        record!("T-1016 partial $label", D3, worst_partial)
        @test worst <= C
        @test worst_partial <= CP
    end
end

# --- T-1017 ---------------------------------------------------------------------------------

function step_bytes_open_3d(mode, ab; ppc, cells)
    prob = open_plasma_3d(Float64; ppc, cells, nsteps = 4)
    st = open_state_3d(prob, ab; mode)
    p = params(prob)
    step!(st, p, 1)
    step!(st, p, 2)
    bytes = @allocated step!(st, p, 3)
    return bytes, count(s -> ParticleMeshElectromagnetic.sorts(st, s), st.species)
end

@testset "T-1017 allocation per open 3D step does not grow with particle count or grid size [A7, WP-072]" begin
    for mode in (Reference(), Fast()), (label, ab) in OPEN3_ABSORBERS
        small, sorts = step_bytes_open_3d(mode, ab; ppc = (1, 1, 1), cells = (6, 14, 12))
        more_particles, _ = step_bytes_open_3d(mode, ab; ppc = (2, 2, 2), cells = (6, 14, 12))
        larger_grid, _ = step_bytes_open_3d(mode, ab; ppc = (1, 1, 1), cells = (8, 20, 18))
        @info "T-1017 bytes per open 3D step" label mode small more_particles larger_grid sorts threads = Threads.nthreads()
        # at more than one thread the Reference sort allocates per particle (#82, as T-457); CI runs one thread
        bound = small + sorts * SORT_COUNTS_BYTES
        if Threads.nthreads() == 1 || mode isa Fast
            @test_alloc more_particles <= bound
            @test_alloc larger_grid <= bound
        end
    end
end

# --- T-1018 ---------------------------------------------------------------------------------

# The checks of the open 3D problem and state: an absorber with no open axis, an open axis with
# no absorber, an open axis 1, a layer that leaves no interior rows along one of the open axes
# (each axis checks its own fit), and a back-edge absorber (the 3D window is not there yet) are
# refused; a state finds each axis' absorber, in the order y, z, and its run evolves both axes'
# convolution fields, each once.
@testset "T-1018 input checks and the absorbers of the open 3D state [A7, A10, WP-072]" begin
    T = Float64
    ok = open_plasma_3d(T)
    sp = ok.species
    mk(bd; cells = ok.cells) = EMProblem("bd", ok.length, cells, ok.dt, 2, sp, 1, ok.kmode; boundaries = bd)
    @test_throws ArgumentError mk((:open, :periodic, :periodic))
    @test_throws ArgumentError mk((:open, :open, :open))
    @test_throws ArgumentError EMState(mk((:periodic, :open, :periodic)), EMNumerics())                        # no absorber
    @test_throws ArgumentError EMState(mk((:periodic, :periodic, :periodic)), EMNumerics(; absorber = PML(; cells = 4)))
    @test_throws ArgumentError EMState(mk((:periodic, :open, :open); cells = (6, 14, 8)), EMNumerics(; absorber = PML(; cells = 4)))   # 2·4 ≥ 8 along z
    @test_throws ArgumentError EMState(mk((:periodic, :open, :open); cells = (6, 8, 12)), EMNumerics(; absorber = AbsorbingLayer(; cells = 4, strength = 1)))
    @test_throws ArgumentError EMState(mk((:periodic, :open, :open)), EMNumerics(; absorber = PML(; cells = 2)))   # thinner than the guards
    @test_throws ArgumentError EMState(ok, EMNumerics(; absorber = PML(; cells = 4), back_absorber = PML(; cells = 4)))
    for (label, ab) in OPEN3_ABSORBERS
        st = EMState(ok, EMNumerics(; absorber = ab))
        Ly, Lz = open_absorber(st, 2), open_absorber(st, 3)
        @test Ly !== nothing && Lz !== nothing && Ly !== Lz
        @test @inferred(open_absorbers(st)) === (Ly, Lz)
        @test open_absorber(st) === Ly
        ψ = (evolved_arrays(Ly)..., evolved_arrays(Lz)...)
        @test length(ψ) == (ab isa PML ? 8 : 0)
        @test all(a -> count(b -> a === b, evolved(st)) == 1, ψ)
        only_y = EMState(open_plasma_3d(T; boundaries = (:periodic, :open, :periodic)), EMNumerics(; absorber = ab))
        @test open_absorber(only_y, 3) === nothing && length(open_absorbers(only_y)) == 1
        only_z = EMState(open_plasma_3d(T; boundaries = (:periodic, :periodic, :open)), EMNumerics(; absorber = ab))
        @test open_absorber(only_z, 2) === nothing && open_absorber(only_z, 3) !== nothing
    end
end
