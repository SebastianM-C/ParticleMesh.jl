# WP-075: the field ionization of the 3D step, T-1040 to T-1048 (T-1049 is their rerun on CUDA, in
# gpu.jl). The dopant of WP-059 and WP-063 is one method over N: the atoms are the lattice of
# `ppc` points per cell across the transverse cells of its `rows`, in the engine's sorted order, the
# rates are each level's law in the field gathered at the atom, the births write into the slots
# behind the electrons' count, and a moving window shifts the atoms and refills the injection
# column's at their global column. The 2D theory carries over through T-1040: a 3D run that does
# not vary along y or z is the 2D run. The problems are in em3d_cases.jl (`ion_box_3d`,
# `ion_window_3d`, the invariant pair `ion_plane_2d`, `ion_inv_3d`); the Reference digests run in
# a worker of their own (em3d_ion_worker.jl) at 1 and 4 threads.

using ParticleMeshElectromagnetic: ionization_frame, injection_column, window_frame, state_view

"Start em3d_ion_worker.jl at `nt` threads, its output going to a file; once per suite."
function start_ion3d_worker(nt)
    get!(ION3D_WORKERS, nt) do
        worker = joinpath(@__DIR__, "em3d_ion_worker.jl")
        out = tempname()
        cmd = `$(Base.julia_cmd()) --startup-file=no --threads=$nt --project=$(dirname(Base.active_project())) $worker`
        (run(pipeline(cmd; stdout = out, stderr = stderr); wait = false), out)
    end
end
const ION3D_WORKERS = Dict{Int,Tuple{Base.Process,String}}()

"The lines of the ionization worker at `nt` threads that start with `prefix`, after it finished."
function ion3d_worker_lines(nt, prefix)
    proc, out = start_ion3d_worker(nt)
    wait(proc)
    success(proc) || error("em3d_ion_worker.jl with $nt threads failed: $proc")
    return filter(startswith(prefix), split(read(out, String), '\n'))
end

haskey(ENV, "PM_NO_ION3D_WORKER") || foreach(start_ion3d_worker, (1, 4))   # beside this file's in-process runs

"The alive electrons of a 3D state's ionization, on the host."
function born_3d(st)
    e = ionization_electrons(st)
    alive = Array(e.status) .== ParticleMeshCore.ALIVE
    pick(a) = Array(a)[alive]
    return (; n = count(alive), cell = map(pick, e.cell), offset = map(pick, e.offset), u = map(pick, e.u), w = pick(e.w))
end

"True when the value and every partial of every entry is zero."
all_zero_3d(a) = all(x -> absmax(x) == 0, a)

"The C¹ rise of a dopant section, from its definition: sin²(π·s/(2·ramp)) between 0 for s ≤ 0 and 1 for s ≥ ramp."
rise_3d(s, ramp) = s <= 0 ? zero(s) : s >= ramp ? one(s) : sin(oftype(s, π) * s / (2 * ramp))^2

"The ADK rate from its closed form, prefactor·(field/E)^power·exp(−field/(3E)), in Float64 from the values of the level's numbers (away from the cutoff)."
adk_closed(l, E) = Float64(value_of(l.prefactor)) * (Float64(value_of(l.field)) / E)^Float64(value_of(l.power)) * exp(-Float64(value_of(l.field)) / (3E))
"The Tong–Lin rate from its closed form, the ADK rate times exp(−suppression·E/field)."
tonglin_closed(l, E) = adk_closed(l, E) * exp(-Float64(value_of(l.suppression)) * E / Float64(value_of(l.field)))
closed_rate(l::ADKLevel, E) = adk_closed(l, E)
closed_rate(l::TongLinLevel, E) = tonglin_closed(l, E)

# --- T-1040 ---------------------------------------------------------------------------------

# The 3D ionization is the 2D ionization where the problem does not vary along the third axis:
# `ion_plane_2d` and the same problem in 3D invariant along z (the plane on x, y) or along y (the
# plane on x, z, a quarter turn about x, the components mapped with the signs of `window_map_3d`):
# a dopant of an ADK and a Tong–Lin level, two atoms per cell along x on 8 of the plane's 12 rows
# (the 3D `rows` on the plane's transverse axis, one atom along the invariant axis), born every 2
# steps; in a box with two cold beams in the uniform field E₀ = (0.2, 0.1, 0), the plane's
# transverse axis periodic or open (the PML), and in the moving window, whose plasma's charge at the
# ramp ionizes a section that the window renews and leaves (64 steps, 25 shifts). Every field and
# current component on every line of the invariant axis is the 2D one after initialize! and after
# every step, scaled as T-1030; the 3D run holds 6 times the 2D run's born electrons at every step
# (one atom per cell along the invariant axis, 6 cells), and their summed weight is 6 times the 2D
# one to roundoff.
function body_T1040(be)
    C = emC("T-1040")
    pml = PML(; cells = 4, kappa_max = 2, alpha_max = 0.1)
    for (label, window, open) in (("box", false, false), ("box open", false, true), ("window", true, false)), T in TYPES,
        mode in (Reference(), Fast()), axis in (3, 2)
        T === Float64 || mode isa Reference || continue
        p2, p3 = ion_plane_2d(T; window, open), ion_inv_3d(T, axis; window, open)
        nums = EMNumerics(; mode, absorber = open ? pml : nothing)
        s2, s3 = EMState(p2, nums; backend = be), EMState(p3, nums; backend = be)
        initialize!(s2, p2)
        initialize!(s3, p3)
        scale = sum(ncells(s2.grid)) * 2
        worst = window_invariant_error(s2, s3, axis) / (base_eps(T) * scale)
        same_count = true
        for k in 1:p2.nsteps
            step!(s2, params(p2), k)
            step!(s3, params(p3), k)
            worst = max(worst, window_invariant_error(s2, s3, axis) / (base_eps(T) * k * scale))
            same_count &= born_3d(s3).n == 6 * born_3d(s2).n
        end
        b2, b3 = born_3d(s2), born_3d(s3)
        w2, w3 = sum(b2.w), sum(b3.w)
        werr = absdiff(w3 / 6, w2) / (base_eps(T) * absmax(w2) * p2.nsteps)
        @info "T-1040" label T mode axis worst werr born = b2.n emitted = maximum(value_of, ionization_state(s2).emitted)
        @test same_count && b2.n > 100
        @test ionization_lost(s2) == ionization_lost(s3) == 0
        @test maximum(value_of, ionization_state(s2).emitted) > 1 / 10
        window && @test window_shifts(s3, p3.nsteps) > 20
        record!("T-1040 $label $(nameof(typeof(mode)))", T, worst, be)
        record!("T-1040 born weight", T, werr, be)
        @test worst <= C
        @test werr <= emC("T-1040", "C_weight")
    end
end

@testset "T-1040 a 3D run with a dopant invariant along y or z is the 2D run, in a box periodic and open and in the moving window [A2, A3, A7, A8, WP-075]" begin
    body_T1040(CPU())
end

# --- T-1041 ---------------------------------------------------------------------------------

# The rates against their closed forms, in a constant field with all three components
# (E = (0.3, 0.2, −0.1), no current, |E| ≈ 0.374), on the 3D box with the dopant on the rows
# y ∈ (1.1, 2.9) and z ∈ (0.4, 2.6) (4 × 4 transverse cells of the 6 × 6):
# - one ADK level (prefactor and density duals), born every 3 steps for 20 steps: the bound
#   population is exp(−W·t) with W = prefactor·(field/E)^power·exp(−field/(3E)) evaluated here from
#   the formula, `emitted` 1 − q¹⁸ and `pending` q¹⁸ − q²⁰ (q = exp(−W·Δt)); birth m put one
#   electron per atom of the weight density·q^{3(m−1)}·(1 − q³) into the run, at rest at its
#   birth step, so that Boris without B gives it u = −E·Δt·(20 − 3m + ½) in every component; the
#   parameters' partials are in the weights and populations only (A3);
# - a chain of an ADK, a Tong–Lin (rate times exp(−suppression·E/field), its closed form) and an
#   ADK level, born every step: the populations are the per-step recurrence of the closed-form
#   rates, an atom passes one level per step, and the released fraction is the conserved count
#   3 − 3f₁ − 2f₂ − f₃, all of it born, equal to the electrons' weight.
function body_T1041(be)
    C = emC("T-1041")
    E0(T) = (T(3) / 10, T(1) / 5, -T(1) / 10)
    Emag = sqrt(0.3^2 + 0.2^2 + 0.1^2)
    rows = ((1.1, 2.9), (0.4, 2.6))
    natoms = 16 * 4 * 4
    worst_W = worst_q = worst_u = worst_w = worst_chain = 0.0
    for T in TYPES
        dens, pref = seeded(T, 1) / 100, seeded(T, 1) / 2
        n, every = 20, 3
        level = ADKLevel(pref, T(3), one(T))
        prob = ion_box_3d(T; E0 = E0(T), levels = (level,), density = dens, every, rows, nsteps = n)
        st = ion_state_3d(prob; current = NoCurrent(), be)
        @test length(ionization_state(st).w) == natoms
        dt = prob.dt
        W = adk_closed(level, Emag)
        worst_W = max(worst_W, abs(Float64(value_of(adk_rate(level, T(Emag)))) - W) / (base_eps(T) * W))
        q(k) = exp(-W * Float64(value_of(dt)) * k)
        tol(x) = C * n * base_eps(T) * abs(x)
        run!(st, params(prob), n)
        s = ionization_state(st)
        err = maximum(x -> abs(Float64(value_of(x)) - q(n)), s.bound[1]) / (n * base_eps(T) * q(n))
        worst_q = max(worst_q, err)
        @test maximum(x -> abs(Float64(value_of(x)) - (1 - q(18))), s.emitted) <= tol(1 - q(18))
        @test maximum(x -> abs(Float64(value_of(x)) - (q(18) - q(20))), s.pending) <= tol(q(18) - q(20))
        @test maximum(absmax, s.bound[1] .+ s.pending .+ s.emitted .- one(T)) <= C * n * base_eps(T)
        el = born_3d(st)
        nb = n ÷ every
        @test el.n == nb * natoms && ionization_lost(st) == 0
        order = sortperm(value_of.(el.u[1]))           # the earliest birth has the most negative u_x
        for m in 1:nb
            idx = order[((m - 1) * natoms + 1):(m * natoms)]
            steps = n - every * m + 0.5
            for d in 1:3
                ud = -Float64(value_of(E0(T)[d])) * Float64(value_of(dt)) * steps
                worst_u = max(worst_u, maximum(i -> abs(Float64(value_of(el.u[d][i])) - ud), idx) / (n * base_eps(T) * abs(ud)))
            end
            wm = Float64(value_of(dens)) * q(every * (m - 1)) * (1 - q(every))
            worst_w = max(worst_w, maximum(i -> abs(Float64(value_of(el.w[i])) - wm), idx) / (n * base_eps(T) * wm))
        end
        @test absdiff(sum(el.w), sum(s.w .* s.emitted)) <= C * n * base_eps(T) * absmax(sum(el.w))
        if T === D3
            fr = ionization_frame(st)
            no_partials(a) = all(x -> all(iszero, partials(x)), Array(a))
            @test all(no_partials, (fr.atoms.offset..., fr.electrons.offset..., fr.electrons.u...))
            @test all(x -> all(!iszero, partials(x)), el.w) && all(x -> all(!iszero, partials(x)), s.bound[1])
        end
        # the chain of three levels, the second a Tong–Lin level
        levels = (ADKLevel(one(T) / 2, T(3) / 2, one(T)), TongLinLevel(seeded(T, 1), T(2), T(3) / 2, T(1)), ADKLevel(one(T), T(3), T(2)))
        cp = ion_box_3d(T; E0 = E0(T), levels, density = dens, rows, nsteps = 12)
        cs = ion_state_3d(cp; current = NoCurrent(), be)
        Wc = map(l -> closed_rate(l, Emag), levels)
        @test all(w -> 0.1 < w * Float64(value_of(cp.dt)) < 2, Wc)     # every level is under way within a few steps
        f, released = [1.0, 0.0, 0.0], 0.0
        for k in 1:cp.nsteps
            dts = Float64(value_of(cp.dt))
            d = [f[j] * -expm1(-Wc[j] * dts) for j in 1:3]
            f = [f[j] * exp(-Wc[j] * dts) + (j == 1 ? 0.0 : d[j - 1]) for j in 1:3]
            released += sum(d)
            step!(cs, params(cp), k)
            s = ionization_state(cs)
            worst_chain = max(worst_chain, maximum(j -> maximum(x -> abs(Float64(value_of(x)) - f[j]), s.bound[j]), 1:3) / (k * base_eps(T)))
            @test all_zero_3d(s.pending)
            @test maximum(x -> abs(Float64(value_of(x)) - released), s.emitted) <= C * k * base_eps(T) * released
            count3 = 3 .- 3 .* s.bound[1] .- 2 .* s.bound[2] .- s.bound[3]
            @test maximum(absmax, s.emitted .- count3) <= C * 8 * base_eps(T)
            k == 1 && @test all_zero_3d(s.bound[3]) && all(x -> value_of(x) > 0, s.bound[2])
            k == 2 && @test all(x -> value_of(x) > 0, s.bound[3])
        end
        s, el = ionization_state(cs), born_3d(cs)
        @test el.n == cp.nsteps * natoms && released > 1
        @test absdiff(sum(el.w), sum(s.w .* s.emitted)) <= C * cp.nsteps * base_eps(T) * absmax(sum(el.w))
    end
    for (name, v) in (("rate", worst_W), ("populations", worst_q), ("momenta", worst_u), ("weights", worst_w), ("chain", worst_chain))
        record!("T-1041 $name", Float64, v, be)
    end
    @info "T-1041 err/eps" worst_W worst_q worst_u worst_w worst_chain
    @test max(worst_W, worst_q, worst_u, worst_w, worst_chain) <= C
end

@testset "T-1041 the 3D dopant's rates are the ADK and Tong–Lin closed forms in a constant field: populations exp(−W·t), a chain's recurrence, births at rest with the released weight [A2, A3, A7, A8, WP-075]" begin
    body_T1041(CPU())
end

# --- T-1042 ---------------------------------------------------------------------------------

# The lattice and the field at the atom's own place, T-784 in 3D. A lattice of 2 × 1 × 3 atoms per
# cell on the rows y ∈ (1.2, 4.8) and z ∈ (0.5, 2.9) of a box of 16 × 8 × 10 cells of
# 1/2 × 3/4 × 2/5 (no two axes' spacings or points per cell can be exchanged unnoticed), in a
# prescribed curl-free field Ex = 0.3 + x/20, Ey = 1/10, Ez = 0.05 + z/4 (each linear along its own
# axis, which the quadratic gather reproduces exactly; no current), born every 2 steps, 8 steps:
# - the atoms are the lattice of the definition in the engine's sorted order (columns fastest,
#   then rows, then layers, the in-cell index fastest within a cell, x before y before z), each of
#   the weight density/6, the transverse cells of the rows only;
# - each atom's bound population is exp(−W·t) of the rate in |E| at its own x and z (off the two
#   columns whose stencil reaches across the periodic seam of x), and the populations differ by
#   more than 0.1 across the lattice;
# - the fourth birth, in the last step, is one electron per atom in the atoms' order behind the
#   three births before it, of the weight (density/6)·q⁶·(1 − q²) of its atom's rate, and it started
#   at its atom on all three axes (its position less the step's displacement u/γ·Δt);
# - the state's kinetic energy is the born electrons' Σ w·(γ − 1); the ions' charge deposited twice
#   into one array is the ions' charge once.
function body_T1042(be)
    C = emC("T-1042")
    for T in TYPES
        dens, pref = seeded(T, 1) / 100, seeded(T, 1) / 2
        n, every, ppc, box, cells = 8, 2, (2, 1, 3), (8, 6, 4), (16, 8, 10)
        a, sx, Ey0, c, sz = T(3) / 10, T(1) / 20, T(1) / 10, T(1) / 20, T(1) / 4
        level = ADKLevel(pref, T(3), one(T))
        prob = ion_box_3d(T; E0 = (a, Ey0, c), levels = (level,), density = dens, ppc, every, rows = ((1.2, 4.8), (0.5, 2.9)), nsteps = n, box,
                          cells, capacity = 16 * 4 * 6 * 6 * 5)
        st = ion_state_3d(prob; current = NoCurrent(), be)
        g, h, dt = guards(st.grid), cellsize(st.grid), prob.dt
        Ex, Ez = Array(st.fields.Ex), Array(st.fields.Ez)          # Ex on (Center, Node, Node), Ez on (Node, Node, Center)
        for i in axes(Ex, 1)
            Ex[i, :, :] .= a + sx * (T(i - g) - one(T) / 2) * h[1]
        end
        for k in axes(Ez, 3)
            Ez[:, :, k] .= c + sz * (T(k - g) - one(T) / 2) * h[3]
        end
        copyto!(st.fields.Ex, Ex)
        copyto!(st.fields.Ez, Ez)
        natoms = 16 * 4 * 6 * 6
        hv = map(x -> Float64(value_of(x)), h)
        pts = [((c1 - 1 + (k1 - 0.5) / 2) * hv[1], (c2 - 0.5) * hv[2], (c3 - 1 + (k3 - 0.5) / 3) * hv[3])
               for c3 in 2:7 for c2 in 3:6 for c1 in 1:16 for k3 in 1:3 for k1 in 1:2]
        s = ionization_state(st)
        @test length(s.x) == natoms == length(pts) && all(s.alive)
        for (d, key) in enumerate((:x, :y, :z))
            @test maximum(i -> abs(Float64(value_of(s[key][i])) - pts[i][d]), 1:natoms) <= 16 * base_eps(T) * box[d]
        end
        @test all(w -> absdiff(w, dens / 6) <= 2 * base_eps(T) * absmax(dens), s.w)
        run!(st, params(prob), n)
        s = ionization_state(st)
        Ef(x, z) = sqrt((0.3 + x / 20)^2 + 0.01 + (0.05 + z / 4)^2)
        W = [adk_closed(level, Ef(p[1], p[3])) for p in pts]
        dts = Float64(value_of(dt))
        inner = [i for i in 1:natoms if 1 < (i - 1) ÷ 6 % 16 + 1 < 16]
        err = maximum(i -> abs(Float64(value_of(s.bound[1][i])) - exp(-W[i] * dts * n)), inner) / (n * base_eps(T))
        record!("T-1042", T, err, be)
        @test err <= C
        @test maximum(value_of, s.bound[1][inner]) - minimum(value_of, s.bound[1][inner]) > 1 / 10
        e = ionization_electrons(st)
        @test Array(e.count)[1] == 4 * natoms && born_3d(st).n == 4 * natoms && ionization_lost(st) == 0
        fourth = (3 * natoms + 1):(4 * natoms)
        w4 = Array(e.w)[fourth]
        d6 = Float64(value_of(dens)) / 6
        werr = maximum(i -> abs(Float64(value_of(w4[i])) - d6 * exp(-W[i] * dts * 6) * -expm1(-W[i] * dts * 2)), inner) / (n * base_eps(T) * d6)
        record!("T-1042 weights", T, werr, be)
        @test werr <= C
        u = map(c -> Float64.(value_of.(Array(c)[fourth])), e.u)
        γ = sqrt.(1 .+ u[1] .^ 2 .+ u[2] .^ 2 .+ u[3] .^ 2)
        start(d) = (Array(e.cell[d])[fourth] .- 1 .+ Float64.(value_of.(Array(e.offset[d])[fourth]))) .* hv[d] .- u[d] ./ γ .* dts
        away(x, L) = abs(mod(x + L / 2, L) - L / 2)
        for d in 1:3
            @test maximum(away.(start(d) .- getindex.(pts, d), box[d])) <= 64 * base_eps(T)
        end
        el = born_3d(st)
        u2 = el.u[1] .^ 2 .+ el.u[2] .^ 2 .+ el.u[3] .^ 2
        Kb = sum(el.w .* u2 ./ (sqrt.(1 .+ u2) .+ 1))
        @test value_of(Kb) > 0 && absdiff(sum(Array(st.K)), Kb) <= 64 * base_eps(T) * absmax(Kb)
        ρi = field(be, T, st.grid)
        ionization_charge!(ρi, st)
        once = bits(ρi)
        ionization_charge!(ρi, st)
        @test bits(ρi) == once && any(!iszero, once)
    end
end

@testset "T-1042 the 3D dopant's lattice of 2 × 1 × 3 per cell on its rows of y and z, the rate of the field at each atom's own place, electrons born at their atom [A2, A3, A8, WP-075]" begin
    body_T1042(CPU())
end

# --- T-1043 ---------------------------------------------------------------------------------

"""
    ion_gauss_3d(st, p, k) -> (; res, bare, scale, escale, ions)

Gauss's law of a 3D state with ionization after step `k`, per interior node: `res`, the residual
`Σ_d D⁻_d E_d − ρ` with ρ the charge of every particle ([`window_charge!`](@ref): the species and the
born electrons) plus the dopant ions' ([`ionization_charge!`](@ref)), less the neutralizing
background (the mean of a periodic box; a window's `window_background!`; none in a box with an
open axis, whose runs here start without charge); `bare`, the same without the ions; `scale`,
`(n₁ + n₂ + n₃)·max|ρ|` of the particles; `escale`, the divergence's own roundoff scale
`(Σ_d 2/Δ_d)·max|E|`; `ions`, the largest ion charge density.
"""
function ion_gauss_3d(st, p, k)
    T = eltype(st.ρ)
    grid = st.grid
    be = get_backend(st.ρ)
    ρ, ρi, ρ0 = field(be, T, grid), field(be, T, grid), field(be, T, grid)
    window_charge!(ρ, st)
    ionization_charge!(ρi, st)
    window = st.aux.window
    window === nothing || window_background!(ρ0, st, p, window_shifts(st, k))
    ρ, ρi, ρ0 = Array(ρ), Array(ρi), Array(ρ0)
    g, n, h = guards(grid), ncells(grid), cellsize(grid)
    I = CartesianIndices(ntuple(d -> (g + 1):(g + n[d]), 3))
    periodic = window === nothing && st.numerics.absorber === nothing
    mean(a) = periodic ? sum(a[I]) / prod(n) : zero(T)
    m, mi = mean(ρ), mean(ρi)
    E = map(Array, (st.fields.Ex, st.fields.Ey, st.fields.Ez))
    e = ntuple(d -> CartesianIndex(ntuple(a -> Int(a == d), 3)), 3)
    div(j) = sum(d -> (E[d][j] - E[d][j - e[d]]) / h[d], 1:3)
    res = [absmax(div(j) - (ρ[j] - m + ρi[j] - mi - ρ0[j])) for j in I]
    bare = [absmax(div(j) - (ρ[j] - m - ρ0[j])) for j in I]
    hv = map(value_of, h)
    escale = sum(d -> 2 / hv[d], 1:3) * maximum(a -> maximum(absmax, a[I]), E)
    return (; res, bare, scale = sum(n) * maximum(absmax, ρ[I]), escale, ions = maximum(absmax, ρi[I]))
end

# Gauss's law with ionization in the 3D box, T-773 and T-785 in 3D. An electron is born on top of
# the immobile ion it leaves, and from then on its current is the charge-conserving one, so
# Σ D⁻E − ρ stays at roundoff when ρ counts the ions' charge:
# - periodic: a thermal plasma (crossing cells and every seam) in Ex = 1/2, which its current then
#   changes, with a dopant of an ADK and a Tong–Lin level at 0.05 n_c (a dual), two atoms per cell
#   along x on the rows y ∈ (1.1, 2.9), born every 2 steps, 20 steps, both modes: the residual with
#   the ions at roundoff after every step (scale as T-773), without them about the ions' charge;
# - open along y and z with an absorbing layer of 3 cells: the uniform field Ey = 1/2 drives the
#   electrons born on the rows about the middle to the lower y face, where they are retired as the
#   plasma's particles are; on the nodes off the layers the residual with the ions stays at
#   roundoff (the scale the charge's plus the divergence's own of the field), fewer electrons are
#   alive than were born, and none of the births is lost.
function body_T1043(be)
    C = emC("T-1043")
    for T in TYPES, mode in (Reference(), Fast())
        levels = (ADKLevel(one(T) / 2, T(3), one(T)), TongLinLevel(one(T) / 2, T(4), T(3) / 2, one(T)))
        prob = ion_box_3d(T; levels, density = seeded(T, 1) / 20, ppc = (2, 1, 1), every = 2, nsteps = 20, plasma = true, rows = ((1.1, 2.9), nothing))
        st = ion_state_3d(prob; mode, be)
        p = params(prob)
        r = ion_gauss_3d(st, p, 0)
        worst = maximum(r.res) / (base_eps(T) * r.scale)
        for k in 1:prob.nsteps
            step!(st, p, k)
            r = ion_gauss_3d(st, p, k)
            worst = max(worst, maximum(r.res) / (base_eps(T) * r.scale * sqrt(k + 1)))
        end
        s = ionization_state(st)
        natoms = length(s.w)
        bare = maximum(r.bare) / r.ions
        @info "T-1043 periodic" T mode worst bare released = extrema(value_of, s.emitted) born = born_3d(st).n ions = r.ions
        record!("T-1043 periodic $(nameof(typeof(mode)))", T, worst, be)
        @test worst <= C
        @test bare > 1 / 4 && r.ions > 1 / 100
        @test ionization_lost(st) == 0 && born_3d(st).n == 10 * natoms
        @test minimum(value_of, s.emitted) > 1 / 2
        @test maximum(abs ∘ value_of, Array(st.fields.Jz)) > 0
        # open along y and z
        layer, n2 = 3, 16
        po = ion_box_3d(T; E0 = (zero(T), one(T) / 2, zero(T)), box = (8, 8, 5), cells = (16, n2, 10), boundaries = (:periodic, :open, :open),
                        rows = ((3.1, 4.9), (1.6, 3.4)), every = 2, nsteps = 40, density = seeded(T, 1) / 20)
        so = ion_state_3d(po; mode, absorber = AbsorbingLayer(; cells = layer, strength = 1, order = 2), be)
        free = ((layer + 2):(n2 - layer), (layer + 2):(10 - layer))
        worst_o = 0.0
        for k in 1:po.nsteps
            step!(so, params(po), k)
            r = ion_gauss_3d(so, params(po), k)
            worst_o = max(worst_o, Float64(maximum(r.res[:, free...]) / (base_eps(T) * (r.scale + r.escale) * sqrt(k + 1))))
        end
        r = ion_gauss_3d(so, params(po), po.nsteps)
        @test maximum(r.bare[:, free...]) > r.ions / 4 > 0
        s, el = ionization_state(so), born_3d(so)
        na = length(s.w)
        emitted = sum(s.w .* s.emitted)
        @info "T-1043 open" T mode worst_o alive = el.n births = 20 * na left = 1 - value_of(sum(el.w)) / value_of(emitted)
        record!("T-1043 open $(nameof(typeof(mode)))", T, worst_o, be)
        @test worst_o <= C
        @test count(x -> value_of(x) > 0, s.emitted) == na == 16 * 4 * 4 && ionization_lost(so) == 0
        @test 0 < el.n < 19 * na
        @test value_of(sum(el.w)) < 0.95 * value_of(emitted)
    end
end

@testset "T-1043 Gauss's law to roundoff with the ions' charge in the 3D box with a dopant being ionized, periodic and open along y and z where born electrons leave, all three types, both modes [A2, A4, A7, A8, WP-075]" begin
    body_T1043(CPU())
end

# --- T-1044 ---------------------------------------------------------------------------------

"""
    ion_lattice_3d(st, k, ppc, (density, start, stop, ramp)) -> (ok, werr)

The atoms' buffer of a 3D window state after step `k` against the definition: `ok` when it is the
full lattice of `ppc` atoms per cell in the engine's sorted order (columns fastest, then the
transverse cells with y before z, the in-cell index fastest within a cell, x before y before z)
with the in-cell offsets `(k_d − ½)/p_d`; `werr` the largest difference of an atom's weight from the
section's profile at its global coordinate over the points per cell, in eps of the density's share.
"""
function ion_lattice_3d(st, k, ppc, prof)
    T = eltype(st.ρ)
    fr = ionization_frame(st)
    a = fr.atoms
    nc = Int.(fr.nc)
    K = prod(ppc)
    dens, start, stop, ramp = prof
    h = cellsize(st.grid)[1]
    sw = window_shifts(st, k)
    s = ionization_state(st)
    kk = [(i % ppc[1], i ÷ ppc[1] % ppc[2], i ÷ (ppc[1] * ppc[2])) for i in 0:(K - 1)]
    δ = ntuple(d -> [(T(x[d]) + one(T) / 2) / ppc[d] for x in kk], 3)
    nt = nc[2] * nc[3]
    ok = all(s.alive) && length(s.w) == nc[1] * nt * K
    ok &= Array(a.cell[1]) == repeat(Int32.(1:nc[1]); inner = K, outer = nt)
    ok &= Array(a.cell[2]) == repeat(Int32.(fr.c0[2] .+ (0:(nc[2] - 1))); inner = nc[1] * K, outer = nc[3])
    ok &= Array(a.cell[3]) == repeat(Int32.(fr.c0[3] .+ (0:(nc[3] - 1))); inner = nc[1] * nc[2] * K)
    ok &= all(d -> Array(a.offset[d]) == repeat(δ[d]; outer = nc[1] * nt), 1:3)
    werr = 0.0
    for c in 1:nc[1], j in 1:K
        x = (T(c + sw - 1) + δ[1][j]) * h
        w = dens * rise_3d(x - start, ramp) * (1 - rise_3d(x - stop, ramp)) / K
        werr = max(werr, maximum(r -> absdiff(s.w[((r - 1) * nc[1] + c - 1) * K + j], w), 1:nt))
    end
    return ok, werr * K / (base_eps(T) * value_of(dens))
end

# The dopant in the 3D window, T-774 in 3D:
# - Gauss's law across shifts with the ions' charge, off nodes 1 and 2 and the antenna's two
#   columns (as T-1032): the round laser of `ion_window_3d` ionizes a doped section on 4 × 4
#   transverse cells about the axis, the window moves on until the section and its electrons
#   have left through the back edge (120 steps, 51 shifts), both modes; while the section is in the
#   window the residual without the ions is about the ions' charge;
# - the lattice through the shifts (vacuum, a level with the cutoff at |E| = 1/8, 2 × 1 × 2 atoms
#   per cell): after any number of shifts the atoms' buffer is the full lattice in sorted order,
#   each atom of the section's weight at its global coordinate (the density and the section's
#   numbers duals), and an atom keeps its populations as it moves back through the buffer: between
#   step 90, when the pulse has left the section, and step 104, every atom of the section still
#   in the window has the bits of its bound, pending and emitted fractions at its new slot, and the
#   fresh atoms at the injection column are in their ground state;
# - a section that lies ahead of the window at the start is loaded by the refill alone, at the
#   window's position: after 90 steps its weights are the profile's.
function body_T1044(be)
    C = emC("T-1044")
    for T in TYPES, mode in (Reference(), Fast())
        prob = ion_window_3d(T; nsteps = 120, density = seeded(T, 1) / 200)
        st = ion_state_3d(prob; mode, be)
        p = params(prob)
        ia = antenna_planes(st)[1].ia
        worst = bare = released = 0.0
        for k in 1:prob.nsteps
            step!(st, p, k)
            k % 4 == 0 || continue
            s_win = window_shifts(st, k)
            r = ion_gauss_3d(st, p, k)
            cols = [i for i in axes(r.res, 1) if !(i in (1, 2, ia - 1 - s_win, ia - s_win))]
            worst = max(worst, Float64(maximum(r.res[cols, :, :]) / (base_eps(T) * r.scale * sqrt(k + 1))))
            r.ions > 1.0e-4 && (bare = max(bare, Float64(maximum(r.bare[cols, :, :]) / r.ions)))
            released = max(released, maximum(value_of, ionization_state(st).emitted))
        end
        @info "T-1044" T mode worst bare released shifts = window_shifts(st, prob.nsteps)
        record!("T-1044 $(nameof(typeof(mode)))", T, worst, be)
        @test worst <= C
        @test 1 / 2 < bare < 2
        @test released > 1 / 4 && ionization_lost(st) == 0 && all(==(0), window_lost(st))
        @test all_zero_3d(ionization_state(st).emitted) && born_3d(st).n == 0        # the section and its electrons have left
    end
    for T in TYPES
        dens, start, stop, ramp = seeded(T, 1) / 200, seeded(T, 28), T(34) + seeded(T, 1) / 2 - T(1) / 2, T(3) + seeded(T, 1) / 4 - T(1) / 4
        ppc = (2, 1, 2)
        K = prod(ppc)
        prob = ion_window_3d(T; nsteps = 104, density = dens, start, stop, ramp, plasma = false, cutoff = 8, ppc, capacity = 65536)
        st = ion_state_3d(prob; be)
        p = params(prob)
        fr = ionization_frame(st)
        nc = Int.(fr.nc)
        nt = nc[2] * nc[3]
        @test nc[1] == injection_column(window_frame(st)) < 72 && nc[2:3] == (4, 4) && Array(fr.atoms.count)[1] == nc[1] * nt * K
        slots(c, r) = ((r - 1) * nc[1] + c - 1) * K .+ (1:K)
        prof = (dens, start, stop, ramp)
        ok, werr = ion_lattice_3d(st, 0, ppc, prof)
        @test ok && werr <= emC("T-1044", "C_weight")
        run!(st, p, 90)
        ok, w1 = ion_lattice_3d(st, 90, ppc, prof)
        @test ok
        s1, sw1 = ionization_state(st), window_shifts(st, 90)
        run!(st, p, 14; first = 91)
        ok, w2 = ion_lattice_3d(st, 104, ppc, prof)
        @test ok
        @test max(w1, w2) <= emC("T-1044", "C_weight")
        s2 = ionization_state(st)
        moved = window_shifts(st, 104) - sw1
        @test moved >= 5 && sw1 > 20
        kept = [(i, j) for r in 1:nt for c in (moved + 1):nc[1] for (i, j) in zip(slots(c, r), slots(c - moved, r)) if value_of(s1.w[i]) > 0]
        @test length(kept) > 40 * K
        @test all(((i, j),) -> same_bits([s1.emitted[i], s1.bound[1][i], s1.pending[i]], [s2.emitted[j], s2.bound[1][j], s2.pending[j]]), kept)
        @test count(((i, j),) -> value_of(s2.emitted[j]) > 1 / 10, kept) > 20 * K        # ionized atoms among them
        fresh = [i for r in 1:nt for c in (nc[1] - moved + 1):nc[1] for i in slots(c, r)]
        @test all(i -> absmax(s2.emitted[i]) == 0 && s2.bound[1][i] == one(T), fresh)
        if T === D3
            @test all(a -> all(x -> all(iszero, partials(x)), Array(a)), fr.atoms.offset)
            @test any(x -> any(!iszero, partials(x)), s2.w)
        end
        # a section ahead of the window: only the refill loads it
        ppc, prof = (1, 2, 1), (dens, start + 28, stop + 27, ramp)   # beyond the injection column's 54.2
        pa = ion_window_3d(T; nsteps = 90, density = dens, start = prof[2], stop = prof[3], ramp, plasma = false, cutoff = 8, ppc)
        sa = ion_state_3d(pa; be)
        @test all_zero_3d(ionization_state(sa).w)
        run!(sa, params(pa), pa.nsteps)
        ok, w3 = ion_lattice_3d(sa, pa.nsteps, ppc, prof)
        @test ok && w3 <= emC("T-1044", "C_weight")
        @test count(w -> value_of(w) > value_of(dens) / 8, ionization_state(sa).w) > 50 && window_shifts(sa, pa.nsteps) > 20
        record!("T-1044 weights", T, max(werr, w1, w2, w3), be)
    end
end

@testset "T-1044 the dopant in the 3D window: Gauss's law with the ions across shifts, the refill at the global column with the section's weights, populations that move with their atoms [A2, A3, A4, A8, WP-075]" begin
    body_T1044(CPU())
end

# --- T-1045 ---------------------------------------------------------------------------------

# Reference mode is bitwise with ionization in 3D: the runs of `ion3d_digests` (the window with
# births, shifts and leaving electrons, and the open box with a plasma) hash to the same bits across
# reruns and launch shapes, and on the CPU at 1 and 4 threads (the worker); the two deposition modes
# agree to roundoff in the fields and in the atoms' released fractions (Float64, the window run).
function body_T1045(be)
    ref = ion3d_digests(; be)
    @test ref == ion3d_digests(; be)
    @test length(ref) == 2 * length(TYPES) && allunique(last.(ref))
    for wgs in (be isa CPU ? (32, 256) : (64, 256))
        @test ion3d_digests(; workgroupsize = wgs, be) == ref
    end
    prob = ion_window_3d(Float64; nsteps = 100)
    a, b = ion_state_3d(prob; mode = Reference(), be), ion_state_3d(prob; mode = Fast(), be)
    run!(a, params(prob), prob.nsteps)
    run!(b, params(prob), prob.nsteps)
    comps = (:Ex, :Ey, :Ez, :Bx, :By, :Bz)
    m = maximum(c -> maximum(abs, Array(getfield(a.fields, c))), comps)
    d = maximum(c -> maximum(abs, Array(getfield(a.fields, c)) .- Array(getfield(b.fields, c))), comps)
    sa, sb = ionization_state(a), ionization_state(b)
    de = maximum(abs, sa.emitted .- sb.emitted)
    @info "T-1045 Reference against Fast" fields = d / m emitted = de born = (born_3d(a).n, born_3d(b).n)
    record!("T-1045", Float64, max(d / m, de) / eps(Float64), be)
    @test max(d / m, de) <= emC("T-1045") * eps(Float64)
    @test born_3d(a).n == born_3d(b).n > 1000
    return ref
end

@testset "T-1045 the 3D step with ionization is bitwise across reruns, launch shapes and 1 and 4 threads in Reference mode, and the modes agree [A4, WP-075]" begin
    ref = body_T1045(CPU())
    for nt in (1, 4)
        fields = split.(ion3d_worker_lines(nt, "EM3DI "))
        got = Dict((String(f[3]), String(f[4])) => parse(UInt64, f[5]) for f in fields)
        @test all(f -> parse(Int, f[2]) == nt, fields)
        @test length(got) == length(ref)
        for (label, T, d) in ref
            @test get(got, (label, type_label(T)), nothing) == d
        end
    end
end

# --- T-1046 ---------------------------------------------------------------------------------

"The parameters of T-1046: the laser's a₀, the dopant's density, the level's prefactor and the section's start (c/ω₀)."
const ION3_X0 = (1.0, 0.005, 0.2, 28.0)
const ION3_NAMES = ("a0", "density", "prefactor", "start")

"""
The born charge (the electrons' summed weight) and their kinetic energy Σ w·(γ − 1) after 90 steps
of `ion_window_3d_small` at `x = (a0, density, prefactor, start)`, when no electron has left the window.
"""
function ion3d_objectives(x, mode = Reference(); be = CPU(), gather = nothing)
    T = promote_type(map(typeof, x)...)
    prob = ion_window_3d_small(T; nsteps = 90, a0 = x[1], density = x[2], prefactor = x[3], start = x[4])
    st = ion_state_3d(prob; mode, be, gather)
    run!(st, params(prob), prob.nsteps)
    el = born_3d(st)
    ionization_lost(st) == 0 || error("a birth found no slot")
    γ = sqrt.(1 .+ el.u[1] .^ 2 .+ el.u[2] .^ 2 .+ el.u[3] .^ 2)
    return [sum(el.w), sum(el.w .* (γ .- 1))]
end

# The gradient through the 3D ionization (A7): ForwardDiff of both objectives in a₀ (through the
# rates in the field at each atom) and the prefactor (through the rates), Float64 Reference, one
# Jacobian, against the finite-difference floor; then a `Dual{Float32,3}` seeded in a₀, the density
# (through the atoms' weights, A3) and the prefactor, against the Float64 Jacobian of the three.
const T1046_AD = Ref{Any}(nothing)
function body_T1046(be)
    x0 = collect(ION3_X0)
    J0 = ion3d_objectives(ION3_X0; be)
    @test J0[1] > 0.05 && J0[2] > 1.0e-4
    ad = ForwardDiff.jacobian(x -> ion3d_objectives((x[1], x[2], x[3], x0[4]); be), x0[1:3])
    runs = Dict{Tuple{Int,Float64},Vector{Float64}}()            # both objectives of a run, for both scans
    for (o, name) in enumerate(("charge", "energy")), i in (1, 3)
        f(xi, mode, seed) = get!(() -> ion3d_objectives(ntuple(j -> j == i ? xi : x0[j], 4), mode; be), runs, (i, xi))[o]
        hs = x0[i] .* 2.0 .^ (-6:-3:-18)
        r = fd_scan(f, x0[i], hs; seed = 1, ad = ad[o, i])
        ratio = r.rel[r.istar] / r.floor_rel
        @info "T-1046 $name, $(ION3_NAMES[i])" ad = ad[o, i] fd = r.fd[r.istar] h = hs[r.istar] rel = r.rel[r.istar] floor = r.floor_rel ratio
        get!(MEASURED, "T-1046 rel/floor, $name", Dict{String,Float64}())[ION3_NAMES[i]] = ratio
        @test isfinite(ad[o, i]) && ad[o, i] != 0
        @test r.rel[r.istar] <= emC("T-1046") * r.floor_rel
        @test r.floor_rel <= emC("T-1046", "floor_max")
    end
    @test ad[1, 2] * x0[2] ≈ J0[1] rtol = 0.05            # the born charge is linear in the density up to the electrons' own fields
    xd = (mkdual(x0[1], 1, 0, 0), mkdual(x0[2], 0, 1, 0), mkdual(x0[3], 0, 0, 1), mkdual(x0[4], 0, 0, 0))
    J = ion3d_objectives(xd; be)
    for (o, name) in enumerate(("charge", "energy"))
        pd = collect(Float64.(Tuple(partials(J[o]))))
        rel = maximum(abs.(pd .- ad[o, 1:3]) ./ abs.(ad[o, 1:3]))
        primal = abs(value(J[o]) - J0[o]) / abs(J0[o])
        @info "T-1046 Dual{Float32,3} $name" partials = pd ref = ad[o, 1:3] rel primal
        get!(MEASURED, "T-1046 Dual{Float32,3} rel", Dict{String,Float64}())[name] = rel
        get!(MEASURED, "T-1046 Dual{Float32,3} primal rel", Dict{String,Float64}())[name] = primal
        @test J[o] isa D3 && all(isfinite, pd) && all(!iszero, pd)
        @test rel <= emC("T-1046", "C_dual") && primal <= emC("T-1046", "C_dual")
    end
end

@testset "T-1046 ForwardDiff of the born charge and energy through a small 3D ionization-injection run against the finite-difference floor, and Dual{Float32,3} against the Float64 gradient [A0, A3, A7, A8, WP-075]" begin
    body_T1046(CPU())
end

# --- T-1047 ---------------------------------------------------------------------------------

"Bytes allocated by four steps with births and shifts of `ion_window_3d` with `ppc` atoms per cell and `capacity` electron slots, after 66 steps."
function ion3d_step_bytes(; ppc, capacity)
    prob = ion_window_3d_small(Float64; ppc, capacity, nsteps = 70)
    st = ion_state_3d(prob)
    p = params(prob)
    run!(st, p, 66)
    return @allocated run!(st, p, 4; first = 67)
end

# The evolved state and the allocation (§0.4, WP-056). The atoms (their cells, offsets, weights,
# status, count and extras), the electrons and the lost-birth counter are in `evolved` once each,
# so a snapshot of the 3D window run with ionization taken before the first birth, at a birth step
# and after shifts with electrons leaving, restored into a fresh state, continues with the
# uninterrupted run's bits; and the bytes four steps with births and shifts allocate do not grow
# with the number of atoms or the electrons' capacity (one thread).
@testset "T-1047 the 3D atoms, their electrons and the lost-birth counter are evolved state that a restart continues bitwise, and a step's allocation does not grow with the lattice or the buffer [A4, A7, WP-075]" begin
    for T in TYPES
        prob = ion_window_3d_small(T; nsteps = 110, density = seeded(T, 1) / 200, capacity = 4096)
        p = params(prob)
        full = ion_state_3d(prob)
        fr = ionization_frame(full)
        ev = evolved(full)
        mine = (fr.atoms.cell..., fr.atoms.offset..., fr.atoms.w, fr.atoms.status, fr.atoms.count, values(fr.atoms.extras)...,
                fr.electrons.cell..., fr.electrons.offset..., fr.electrons.u..., fr.electrons.w, fr.electrons.status, fr.electrons.count, fr.lost)
        @test all(a -> count(x -> x === a, ev) == 1, mine)
        @test !any(a -> any(x -> x === a, ev), (fr.flags, values(fr.electrons.extras)...))
        ks = (0, 56, 70, 97)
        snaps = EMSnapshot[snapshot(full, 0)]
        done = 0
        for k in ks[2:end]
            run!(full, p, k - done; first = done + 1)
            done = k
            push!(snaps, snapshot(full, k))
        end
        run!(full, p, prob.nsteps - done; first = done + 1)
        @test born_3d(full).n > 0
        want = map(bits, evolved(full))
        for (k, snap) in zip(ks, snaps)
            fresh = EMState(prob, EMNumerics(); backend = CPU())
            restore!(fresh, snap)
            run!(fresh, p, prob.nsteps - k; first = k + 1)
            @test map(bits, evolved(fresh)) == want
        end
    end
    small = ion3d_step_bytes(; ppc = (1, 1, 1), capacity = 8192)
    more_atoms = ion3d_step_bytes(; ppc = (2, 1, 2), capacity = 4 * 8192)
    larger_buffer = ion3d_step_bytes(; ppc = (1, 1, 1), capacity = 4 * 8192)
    @info "T-1047 bytes per four steps" small more_atoms larger_buffer threads = Threads.nthreads()
    if Threads.nthreads() == 1
        bound = small + 4 * 2 * SORT_COUNTS_BYTES
        @test_alloc more_atoms <= bound
        @test_alloc larger_buffer <= bound
    end
end

# --- T-1048 ---------------------------------------------------------------------------------

# What is refused at N = 3 and what a 3D state holds: a lattice of two entries or a single range of
# rows (the 2D forms), a state asked to run a problem with another lattice, rows, cadence,
# capacity or number of levels, or parameters whose ionization does not fit; the 3D state's
# atoms, electrons and kinetic-energy chunks, `ionization_state` with z, and parameters with other
# rates or another density or section, which are the state's to run (A3).
@testset "T-1048 input checks of the 3D ionization and the parts of its state [A2, A10, WP-075]" begin
    T = Float64
    lv = ADKLevel(0.5, 3.0, 1.0)
    @test_throws ArgumentError ion_box_3d(T; ppc = (1, 1))
    @test_throws ArgumentError ion_box_3d(T; rows = (1.1, 2.9))
    @test ion_box_3d(T; rows = ((1.1, 2.9), nothing)).ionization.rows === ((1.1, 2.9), nothing)
    @test_throws ArgumentError EMState(ion_box_3d(T; rows = ((1.1, 2.9), (5.0, 9.0))), EMNumerics())   # no centre of a z cell in (5, 9) on a box of 3
    prob = ion_box_3d(T; ppc = (2, 1, 1), rows = ((1.1, 2.9), nothing), every = 2)
    e3 = SpeciesDescription{T}(:e, -1.0, 1.0, 0.0, 0.0, (1, 0, 0), (1, 1, 1), zero3(T), zero3(T), 1)
    @test EMProblem("x", prob.length, prob.cells, prob.dt, 4, (e3,), 1, (1, 0, 0);
                    ionization = FieldIonization(; density = 0.01, levels = lv, capacity = 10)).ionization.ppc === (1, 1, 1)
    st = ion_state_3d(prob; current = NoCurrent())
    fr = ionization_frame(st)
    @test fr.c0 == Int32.((1, 3, 1)) && fr.nc == Int32.((16, 4, 6)) && fr.ppc == Int32.((2, 1, 1))
    @test capacity(fr.atoms) == 16 * 4 * 6 * 2 && capacity(fr.electrons) == prob.ionization.capacity
    @test length(st.K) == cld(capacity(st.species[1]), ParticleMeshElectromagnetic.KINETIC_CHUNK) +
                          cld(capacity(fr.electrons), ParticleMeshElectromagnetic.KINETIC_CHUNK)
    s = ionization_state(st)
    @test keys(s)[1:3] == (:x, :y, :z) && length(s.z) == capacity(fr.atoms)
    cap = prob.ionization.capacity
    for other in (ion_box_3d(T; ppc = (1, 2, 1), rows = ((1.1, 2.9), nothing), every = 2, capacity = cap),
                  ion_box_3d(T; ppc = (2, 1, 1), rows = ((1.1, 2.9), (0.4, 2.6)), every = 2, capacity = cap),
                  ion_box_3d(T; ppc = (2, 1, 1), rows = ((1.1, 2.9), nothing), every = 3, capacity = cap),
                  ion_box_3d(T; ppc = (2, 1, 1), rows = ((1.1, 2.9), nothing), every = 2, capacity = 77),
                  ion_box_3d(T; ppc = (2, 1, 1), rows = ((1.1, 2.9), nothing), every = 2, capacity = cap, levels = (lv, lv)))
        @test_throws ArgumentError initialize!(st, other)
    end
    plain = EMProblem(prob.name, prob.length, prob.cells, prob.dt, prob.nsteps, prob.species, prob.seed, prob.kmode; E0 = prob.E0)
    @test_throws ArgumentError initialize!(st, plain)
    @test initialize!(st, ion_box_3d(T; ppc = (2, 1, 1), rows = ((1.1, 2.9), nothing), every = 2, density = 0.5,
                                     levels = (ADKLevel(7.0, 1.0, 2.0),), section = DopantSection(; start = 1, stop = 2, ramp = 1))) === st
    p = params(prob)
    @test_throws ArgumentError step!(st, EMParams(p; ionization = nothing), 1)
    @test_throws ArgumentError step!(st, EMParams(p; ionization = IonizationParams{Float64}(0.01, (lv, lv))), 1)
    sp = EMState(plain, EMNumerics(; current = NoCurrent()))
    initialize!(sp, plain)
    @test_throws ArgumentError step!(sp, EMParams(params(plain); ionization = p.ionization), 1)
    for f in (ionization_electrons, ionization_state, ionization_lost)
        @test_throws ArgumentError f(sp)
    end
    @test_throws ArgumentError ionization_charge!(similar(sp.ρ), sp)
    @test step!(st, EMParams(p; ionization = IonizationParams{Float64}(0.5, (ADKLevel(7.0, 1.0, 2.0),))), 1) === st
    # the filter and the resort stay refused at N = 3 with a dopant too
    @test_throws ArgumentError EMState(prob, EMNumerics(; filter = BinomialFilter(1)))
    @test_throws ArgumentError EMState(prob, EMNumerics(; resort = true))
end
