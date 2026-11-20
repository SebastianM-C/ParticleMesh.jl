# WP-076: the compensated gather of the 3D step, T-1050 to T-1056 (T-1059 is their rerun on CUDA, in
# gpu.jl). The CompensatedGather of WP-067 is one method over N: the particles (the push, the
# ionization's rates and births, the initial kick) read a copy of E and B sharpened along axis 1,
# the laser's and the window's axis, by E + β(2E_i − E_{i−1} − E_{i+1}), every transverse line of
# the 3D arrays alike; the mesh solve, the current and the diagnostics keep the grid's fields. The
# helpers of the transfer are gather_cases.jl's, and the problems with a dopant and WP-075's helpers
# for them (`born_3d`, `ion_gauss_3d`, `ion3d_objectives`, the ionization worker) em3d_cases.jl's.

using ParticleMeshElectromagnetic: ionization_frame, state_view
haskey(ENV, "PM_NO_ION3D_WORKER") || foreach(start_ion3d_worker, (1, 4))   # beside this file's in-process runs

# --- T-1050 ---------------------------------------------------------------------------------

# The transfer in 3D, from the definitions, T-960 at N = 3. On a 3D state with each of E and B a
# sampled mode along x of wavenumber 1 or 2 (its own phase, each partial of a dual its own),
# `sharpen_gather!`'s copy is the mode times 1 + 4β sin²(kΔx/2) at every entry, guards included,
# and its guards are the periodic images of its interior bit for bit; an oblique mode with the
# same k_x and any k_y, k_z is sharpened by the same factor (the stencil acts along x only, so the
# correction of the shape's loss is the laser axis'); a constant is copied bit for bit. Through the
# module's 3D gather (`fields_at` on `particle_fields`) at 8 offsets per cell along x of every cell
# (fixed offsets across), the projection of each of the six components onto the mode is the
# stencil's transfer times the shape's mean over those offsets (`gather_lattice_mean`), to
# roundoff: one at k = 1 for both orders but for the aliases of the offsets' lattice, and
# `gather_transfer` at k = 2. Without compensation the particles read the grid's fields.
function body_T1050(be)
    C = emC("T-1050")
    worst_s, worst_g, worst_o = 0.0, 0.0, 0.0
    for T in TYPES, shape in (Linear, Quadratic), k in (1, 2), comp in (true, false)
        cg = comp ? CompensatedGather() : nothing
        st = gather_state(T, 3, shape, cg; be)
        grid = st.grid
        h = Float64(value_of(cellsize(grid)[1]))
        p = order(shape)
        for (j, name) in enumerate(GATHER_NAMES)
            mode_field!(st.fields[name], grid, k, j / 3, T)
        end
        sharpen_gather!(st)
        pf = particle_fields(st)
        comp || @test pf === st.fields && st.sharpened === nothing
        x = k * h / 2
        Ts = comp ? 1 + 4 * big(sharpening_beta(cg, shape, h)) * sin(big(x))^2 : big(1)
        for name in GATHER_NAMES
            a, b = Array(st.fields[name]), Array(pf[name])
            err = maximum(I -> maximum(abs.(BigFloat.(components(b[I])) .- Ts .* BigFloat.(components(a[I])))), CartesianIndices(a))
            worst_s = max(worst_s, Float64(err / base_eps(T)))
            @test err <= C * base_eps(T)
            img = copy(pf[name])
            fill_guards!(img, grid)
            @test bits(img) == bits(pf[name])
        end
        M = 8
        hf = map(Array, pf)
        g = Int32(guards(grid))
        n1 = ncells(grid)[1]
        mean = Ts * gather_lattice_mean(p, x, M)
        for (j, name) in enumerate(GATHER_NAMES)
            nc = length(components(zero(T)))
            Cs, Ss = zeros(BigFloat, nc), zeros(BigFloat, nc)
            for c in 1:n1, m in 0:(M - 1)
                E, B = fields_at(hf, (Int32(c), Int32(3), Int32(4)), (T(m) / M, T(37) / 100, T(71) / 100), shape, g)
                v = (E..., B...)[j]
                ξ = k * (c - 1 + m / M) * big(h)
                Cs .+= BigFloat.(components(v)) .* cos(ξ)
                Ss .+= BigFloat.(components(v)) .* sin(ξ)
            end
            amp = sqrt.(Cs .^ 2 .+ Ss .^ 2) .* 2 ./ (n1 * M)
            ref = T <: Dual ? (1, 1, big(1) / 2, 1) : (1,)
            err = maximum(abs.(amp .- mean .* collect(ref)))
            worst_g = max(worst_g, Float64(err / base_eps(T)))
            @test err <= C * base_eps(T)
        end
        comp && k == 1 && @test abs(mean - 1) < 2.0e-3
        comp && @test Float64(Ts) * Float64(gather_lattice_mean(p, x, 4096)) ≈ gather_transfer(cg, shape, h, k) rtol = 1.0e-6
        # an oblique mode: the same factor whatever its wavenumbers across x
        if comp
            for (j, name) in enumerate(GATHER_NAMES)
                oblique_field!(st.fields[name], grid, (k, 2π / value_of(st.grid.dx[2]) / 6, -2π / value_of(st.grid.dx[3]) / 3), j / 5, T)
            end
            sharpen_gather!(st)
            for name in GATHER_NAMES
                a, b = Array(st.fields[name]), Array(particle_fields(st)[name])
                err = maximum(I -> maximum(abs.(BigFloat.(components(b[I])) .- Ts .* BigFloat.(components(a[I])))), CartesianIndices(a))
                worst_o = max(worst_o, Float64(err / base_eps(T)))
                @test err <= C * base_eps(T)
            end
        end
        for name in GATHER_NAMES
            fill!(st.fields[name], one(T) / 3)
        end
        sharpen_gather!(st)
        @test all(name -> bits(particle_fields(st)[name]) == bits(st.fields[name]), GATHER_NAMES)
    end
    record!("T-1050 stencil", Float64, worst_s, be)
    record!("T-1050 gather", Float64, worst_g, be)
    record!("T-1050 oblique", Float64, worst_o, be)
    @info "T-1050 err/eps" stencil = worst_s gather = worst_g oblique = worst_o
end

@testset "T-1050 the 3D compensated gather's transfer: the copy of a sampled mode times 1 + 4β sin²(k_xΔx/2) whatever k_y and k_z, guards the interior's images, the particles' gather of every component one at the laser's wavenumber [A2, A7, A8, WP-076]" begin
    @test_throws ArgumentError gather_state(Float64, 3, Quadratic, CompensatedGather(; k = 13))     # 12 cells per wavelength 2π/k₀
    @test gather_state(Float64, 3, Quadratic, CompensatedGather()).sharpened isa NamedTuple{(:Ex, :Ey, :Ez, :Bx, :By, :Bz)}
    body_T1050(CPU())
end

# --- T-1051 ---------------------------------------------------------------------------------

# A 3D run with the compensated gather that is invariant along y or z is the 2D run with it:
# T-1040's pairs (the beams in a field with a dopant, periodic, and the window's plasma with a
# doped section) with `gather = CompensatedGather()` on both, Δx = 1/2, after every step, scaled as
# T-1030, with 6 times the 2D run's born electrons; and the compensation is active (the momenta and
# the born electrons differ from the plain runs').
function body_T1051(be)
    C = emC("T-1051")
    cg = CompensatedGather()
    for (label, window) in (("box", false), ("window", true)), T in TYPES, axis in (3, 2)
        p2, p3 = ion_plane_2d(T; window), ion_inv_3d(T, axis; window)
        nums = EMNumerics(; gather = cg)
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
        @info "T-1051" label T axis worst born = born_3d(s2).n
        @test same_count && born_3d(s2).n > 100
        record!("T-1051 $label", T, worst, be)
        @test worst <= C
        if axis == 3
            plain = EMState(p3, EMNumerics(); backend = be)
            initialize!(plain, p3)
            run!(plain, params(p3), p3.nsteps)
            @test born_3d(plain).u[1] != born_3d(s3).u[1] && Array(plain.species[1].u[1]) != Array(s3.species[1].u[1])
        end
    end
end

@testset "T-1051 a 3D run with the compensated gather invariant along y or z is the 2D run with it, in a box and in the moving window [A2, A7, A8, WP-076]" begin
    body_T1051(CPU())
end

# --- T-1052 ---------------------------------------------------------------------------------

# The current and the mesh solve keep the grid's fields, so Gauss's law is untouched in 3D: T-1043's
# periodic run (a thermal plasma in Ex = 1/2 with a dopant of an ADK and a Tong–Lin level) and the
# dopant in the 3D window of T-1044 (the round laser ionizing the section, 120 steps, off nodes 1, 2
# and the antenna's columns) with the compensated gather, all three types, both modes, hold
# Σ D⁻E − ρ (the ions' charge included) at roundoff after every step with T-1043's and T-1044's
# bounds. In the window the compensated particles read more of the laser's field, so more charge is
# born than without compensation. In Reference mode the box run is bitwise across launch shapes.
function body_T1052(be)
    cg = CompensatedGather()
    for T in TYPES, mode in (Reference(), Fast())
        levels = (ADKLevel(one(T) / 2, T(3), one(T)), TongLinLevel(one(T) / 2, T(4), T(3) / 2, one(T)))
        prob = ion_box_3d(T; levels, density = seeded(T, 1) / 20, ppc = (2, 1, 1), every = 2, nsteps = 20, plasma = true, rows = ((1.1, 2.9), nothing))
        st = ion_state_3d(prob; mode, be, gather = cg)
        p = params(prob)
        r = ion_gauss_3d(st, p, 0)
        worst = maximum(r.res) / (base_eps(T) * r.scale)
        for k in 1:prob.nsteps
            step!(st, p, k)
            r = ion_gauss_3d(st, p, k)
            worst = max(worst, maximum(r.res) / (base_eps(T) * r.scale * sqrt(k + 1)))
        end
        record!("T-1052 box $(nameof(typeof(mode)))", T, worst, be)
        @test worst <= emC("T-1052")
        @test ionization_lost(st) == 0 && born_3d(st).n == 10 * length(ionization_state(st).w)
        if mode isa Reference
            again = ion_state_3d(prob; mode, be, gather = cg, workgroupsize = 64)
            run!(again, p, prob.nsteps)
            @test ion_digest_3d(again) == ion_digest_3d(st)
        end
        wp = ion_window_3d(T; nsteps = 120)
        ws = ion_state_3d(wp; mode, be, gather = cg)
        pw = params(wp)
        ia = antenna_planes(ws)[1].ia
        worst_w, most = 0.0, 0.0
        for k in 1:wp.nsteps
            step!(ws, pw, k)
            k % 4 == 0 || continue
            s_win = window_shifts(ws, k)
            r = ion_gauss_3d(ws, pw, k)
            cols = [i for i in axes(r.res, 1) if !(i in (1, 2, ia - 1 - s_win, ia - s_win))]
            worst_w = max(worst_w, Float64(maximum(r.res[cols, :, :]) / (base_eps(T) * r.scale * sqrt(k + 1))))
            k == 100 && (most = Float64(value_of(sum(born_3d(ws).w))))
        end
        @info "T-1052" T mode worst worst_w
        record!("T-1052 window $(nameof(typeof(mode)))", T, worst_w, be)
        @test worst_w <= emC("T-1052")
        if T === Float64 && mode isa Reference
            plain = ion_state_3d(wp; mode, be)
            run!(plain, pw, 100)
            ratio = most / sum(born_3d(plain).w)
            MEASURED["T-1052 born charge after 100 steps, compensated/plain"] = Dict("Float64" => ratio)
            @test ratio > 1.02
        end
    end
end

@testset "T-1052 with the 3D compensated gather Gauss's law holds to roundoff with the ions' charge in the box and across window shifts, all three types, both modes, bitwise across launch shapes [A2, A4, A7, A8, WP-076]" begin
    body_T1052(CPU())
end

# --- T-1053 ---------------------------------------------------------------------------------

"The bits of every particle array of a 3D state: its species, the atoms with their populations and the born electrons."
function particle_bits_3d(st)
    sb(s) = (map(bits, s.cell)..., map(bits, s.offset)..., map(bits, s.u)..., bits(s.w), bits(s.status), bits(s.count),
             map(bits, values(s.extras))...)
    fr = ionization_frame(st)
    return (map(sb, st.species)..., (fr === nothing ? () : (sb(fr.atoms), sb(fr.electrons), bits(fr.lost)))...)
end

"A plain 3D state holding the snapshot `snap` of `prob`."
snap_state_3d(prob, snap, be) = restore!(EMState(prob, EMNumerics(); backend = be), snap)

# What reads the copy, T-962 in 3D. The compensated step is the plain step with the particles'
# fields replaced by the sharpened copy: T-1052's box run with the compensated gather to step 5,
# restored into a plain state whose E and B are then the compensated state's copy, and both take
# step 6 (a birth step): every particle array (the push, the atoms' populations, the births'
# positions, weights and kicked momenta) has the same bits, and the fields differ. The same for
# `initialize!`'s half-step kick. The step writes the copy itself before anything reads it, at
# every step (from NaN before steps 5 and 6, the copy after each is the stencil's of the fields the
# step started from), and the observer reads the grid's fields. A compensated run restarted at step
# 3, and a view of the state over copies of its arrays, continue with the uninterrupted run's bits.
# In the 3D window the copy's x-guards are zero as the grid's are, and its interior is the stencil's.
function body_T1053(be)
    cg = CompensatedGather()
    for T in TYPES
        levels = (ADKLevel(one(T) / 2, T(3), one(T)), TongLinLevel(one(T) / 2, T(4), T(3) / 2, one(T)))
        prob = ion_box_3d(T; levels, density = seeded(T, 1) / 20, ppc = (2, 1, 1), every = 2, nsteps = 8, plasma = true, rows = ((1.1, 2.9), nothing))
        p = params(prob)
        st = ion_state_3d(prob; be, gather = cg)
        β = sharpening_beta(cg, st.numerics.shape, st.grid.dx[1])
        copy_bits() = map(name -> bits(st.sharpened[name]), GATHER_NAMES)
        run!(st, p, 4)
        want = map(name -> (r = similar(st.fields[name]); sharpen_x!(r, st.fields[name], β, st.grid); bits(r)), GATHER_NAMES)
        foreach(a -> fill!(a, T(NaN)), values(st.sharpened))
        step!(st, p, 5)
        @test copy_bits() == want                  # the copy of the fields step 5 started from
        snap = snapshot(st, 5)
        plain = snap_state_3d(prob, snap, be)
        diag = snap_state_3d(prob, snap, be)
        for name in GATHER_NAMES
            sharpen_x!(plain.fields[name], st.fields[name], β, st.grid)
        end
        want = map(name -> bits(plain.fields[name]), GATHER_NAMES)
        foreach(a -> fill!(a, T(NaN)), values(st.sharpened))
        step!(st, p, 6)
        @test copy_bits() == want                  # and of those step 6 started from
        step!(plain, p, 6)
        step!(diag, p, 6)
        @test particle_bits_3d(plain) == particle_bits_3d(st)
        @test bits(plain.fields.Ey) != bits(st.fields.Ey)
        @test born_3d(st).n > born_3d(snap_state_3d(prob, snap, be)).n
        @test [bits(Array(h)[6:6]) for h in values(st.recorder.hist)] == [bits(Array(h)[6:6]) for h in values(diag.recorder.hist)]
        ic = EMState(prob, EMNumerics(; gather = cg); backend = be)
        initialize!(ic, prob)
        ip = EMState(prob, EMNumerics(); backend = be)
        initialize!(ip, prob; momenta_at = :t_minus_half)
        copyf = map(similar, (; Ex = ip.fields.Ex, Ey = ip.fields.Ey, Ez = ip.fields.Ez, Bx = ip.fields.Bx, By = ip.fields.By, Bz = ip.fields.Bz))
        for name in GATHER_NAMES
            sharpen_x!(copyf[name], ip.fields[name], β, ip.grid)
        end
        pf = merge(ip.fields, copyf)
        for s in ip.species
            ParticleMeshCore.launcher(kick_kernel!, get_backend(s), nothing)(s, pf, ip.numerics.pusher, ip.numerics.shape, Int32(guards(ip.grid)),
                                                                           -s.q / s.m * ip.dt / 4; ndrange = capacity(s))
        end
        @test map(bits, ic.species[1].u) == map(bits, ip.species[1].u)
        it0 = EMState(prob, EMNumerics(); backend = be)
        initialize!(it0, prob)
        @test map(bits, ic.species[1].u) != map(bits, it0.species[1].u)
        full = ion_state_3d(prob; be, gather = cg)
        run!(full, p, 3)
        s3 = snapshot(full, 3)
        run!(full, p, prob.nsteps - 3; first = 4)
        fresh = EMState(prob, EMNumerics(; gather = cg); backend = be)
        restore!(fresh, s3)
        run!(fresh, p, prob.nsteps - 3; first = 4)
        @test map(bits, evolved(fresh)) == map(bits, evolved(full))
        ran = ion_state_3d(prob; be, gather = cg)
        run!(ran, p, 3)
        sv = state_view(ran, map(copy, evolved(ran)))
        @test sv.sharpened === ran.sharpened
        run!(sv, p, prob.nsteps - 3; first = 4)
        @test map(bits, evolved(sv)) == map(bits, evolved(full))
        wp = ion_window_3d(T; nsteps = 66)
        ws = ion_state_3d(wp; be, gather = cg)
        run!(ws, params(wp), wp.nsteps)
        @test window_shifts(ws, wp.nsteps) > 0
        sharpen_gather!(ws)
        g, n1 = guards(ws.grid), ncells(ws.grid)[1]
        βw = sharpening_beta(cg, ws.numerics.shape, ws.grid.dx[1])
        for name in GATHER_NAMES
            c = Array(ws.sharpened[name])
            ref = similar(ws.fields[name])
            sharpen_x!(ref, ws.fields[name], βw, ws.grid)
            r = Array(ref)
            @test all(iszero, c[1:g, :, :]) && all(iszero, c[(end - g + 1):end, :, :])
            @test bits(c[(g + 1):(g + n1), :, :]) == bits(r[(g + 1):(g + n1), :, :])
        end
        @test maximum(absmax, Array(ws.sharpened.Ey)) > 1 / 10
    end
end

@testset "T-1053 what reads the 3D compensated copy: the push, the ionization's rates and births and the initial kick, bit for bit as the plain step in the sharpened fields; the observer the grid's; a restart and a view continue bitwise; the window's edges [A2, A4, A8, WP-076]" begin
    body_T1053(CPU())
end

# --- T-1054 ---------------------------------------------------------------------------------

# Reference mode is bitwise with the compensated gather in 3D: the runs of `ion3d_digests` with it
# hash to the same bits across reruns and launch shapes, and on the CPU at 1 and 4 threads (the
# worker, `EM3DG`), and they differ from the plain runs' digests (the option acts in both runs).
function body_T1054(be)
    ref = ion3d_digests(; be, gather = CompensatedGather())
    @test ref == ion3d_digests(; be, gather = CompensatedGather())
    @test allunique(last.(ref))
    for wgs in (be isa CPU ? (32, 256) : (64, 256))
        @test ion3d_digests(; workgroupsize = wgs, be, gather = CompensatedGather()) == ref
    end
    return ref
end

@testset "T-1054 the 3D step with the compensated gather is bitwise across reruns, launch shapes and 1 and 4 threads in Reference mode [A4, WP-076]" begin
    ref = body_T1054(CPU())
    plain = Dict((String(f[3]), String(f[4])) => parse(UInt64, f[5]) for f in split.(ion3d_worker_lines(1, "EM3DI ")))
    for nt in (1, 4)
        fields = split.(ion3d_worker_lines(nt, "EM3DG "))
        got = Dict((String(f[3]), String(f[4])) => parse(UInt64, f[5]) for f in fields)
        @test all(f -> parse(Int, f[2]) == nt, fields)
        @test length(got) == length(ref)
        for (label, T, d) in ref
            @test get(got, (label, type_label(T)), nothing) == d
            @test get(plain, (label, type_label(T)), nothing) != d
        end
    end
end

# --- T-1055 ---------------------------------------------------------------------------------

# The gradient through the 3D compensated gather (A7), T-963 in 3D: T-1046's run with the
# compensated gather, ForwardDiff of the born charge and energy in a₀ (through the sharpened
# field the rates read) against the finite-difference floor, and the Dual{Float32,3} partials in a₀,
# the density and the prefactor against the Float64 Jacobian; the compensated particles read more of
# the laser's field, so more charge is born than without.
function body_T1055(be)
    cg = CompensatedGather()
    x0 = collect(ION3_X0)
    J0 = ion3d_objectives(ION3_X0; be, gather = cg)
    Jp = ion3d_objectives(ION3_X0; be)
    @info "T-1055 objectives" compensated = J0 plain = Jp rise = J0 ./ Jp .- 1
    MEASURED["T-1055 born charge, compensated/plain"] = Dict("Float64" => J0[1] / Jp[1])
    @test J0[1] > 1.02 * Jp[1]
    ad = ForwardDiff.jacobian(x -> ion3d_objectives((x[1], x[2], x[3], x0[4]); be, gather = cg), x0[1:3])
    runs = Dict{Float64,Vector{Float64}}()
    for (o, name) in enumerate(("charge", "energy"))
        f(xi, mode, seed) = get!(() -> ion3d_objectives((xi, x0[2], x0[3], x0[4]), mode; be, gather = cg), runs, xi)[o]
        hs = x0[1] .* 2.0 .^ (-6:-3:-18)
        r = fd_scan(f, x0[1], hs; seed = 1, ad = ad[o, 1])
        ratio = r.rel[r.istar] / r.floor_rel
        @info "T-1055 $name, a0" ad = ad[o, 1] fd = r.fd[r.istar] rel = r.rel[r.istar] floor = r.floor_rel ratio
        get!(MEASURED, "T-1055 rel/floor", Dict{String,Float64}())[name] = ratio
        @test isfinite(ad[o, 1]) && ad[o, 1] != 0
        @test r.rel[r.istar] <= emC("T-1055") * r.floor_rel
        @test r.floor_rel <= emC("T-1055", "floor_max")
    end
    xd = (mkdual(x0[1], 1, 0, 0), mkdual(x0[2], 0, 1, 0), mkdual(x0[3], 0, 0, 1), mkdual(x0[4], 0, 0, 0))
    J = ion3d_objectives(xd; be, gather = cg)
    for (o, name) in enumerate(("charge", "energy"))
        pd = collect(Float64.(Tuple(partials(J[o]))))
        rel = maximum(abs.(pd .- ad[o, 1:3]) ./ abs.(ad[o, 1:3]))
        primal = abs(value(J[o]) - J0[o]) / abs(J0[o])
        @info "T-1055 Dual{Float32,3} $name" partials = pd ref = ad[o, 1:3] rel primal
        get!(MEASURED, "T-1055 Dual{Float32,3} rel", Dict{String,Float64}())[name] = rel
        get!(MEASURED, "T-1055 Dual{Float32,3} primal rel", Dict{String,Float64}())[name] = primal
        @test J[o] isa D3 && all(isfinite, pd) && all(!iszero, pd)
        @test rel <= emC("T-1055", "C_dual") && primal <= emC("T-1055", "C_dual")
    end
end

@testset "T-1055 ForwardDiff through the 3D compensated gather against the finite-difference floor, Dual{Float32,3} against the Float64 gradient, and more charge born [A0, A3, A7, A8, WP-076]" begin
    body_T1055(CPU())
end

# --- T-1056 ---------------------------------------------------------------------------------

"Bytes allocated by four steps of `ion_window_3d_small` with the compensated gather (births and shifts), after 66 steps, on `cells`."
function gather3d_step_bytes(; cells)
    prob = ion_window_3d(Float64; nsteps = 70, cells, rows = ((3.2, 9.4), (3.2, 9.4)))
    st = ion_state_3d(prob; gather = CompensatedGather())
    p = params(prob)
    run!(st, p, 66)
    return @allocated run!(st, p, 4; first = 67)
end

# The checks and the parts of the 3D state with the compensated gather: a wavenumber the grid does
# not resolve is refused, the copy is six arrays of the fields' size (none without the gather), the
# copy is scratch (not in `evolved`), and the bytes four steps allocate do not grow with the grid.
@testset "T-1056 the 3D state's compensated gather: input checks, the copy's arrays, and a step's allocation that does not grow with the grid [A7, A10, WP-076]" begin
    prob = ion_window_3d_small(Float64; nsteps = 10)
    @test_throws ArgumentError EMState(prob, EMNumerics(; gather = CompensatedGather(; k = 5)))      # kΔx = 5π/4 > π
    st = EMState(prob, EMNumerics(; gather = CompensatedGather()))
    @test keys(st.sharpened) == GATHER_NAMES && all(name -> size(st.sharpened[name]) == size(st.fields[name]), GATHER_NAMES)
    @test !any(a -> any(x -> x === a, evolved(st)), values(st.sharpened))
    @test EMState(prob, EMNumerics()).sharpened === nothing
    small = gather3d_step_bytes(; cells = (72, 8, 8))
    larger = gather3d_step_bytes(; cells = (72, 12, 12))
    @info "T-1056 bytes per four steps" small larger threads = Threads.nthreads()
    Threads.nthreads() == 1 && @test_alloc larger <= small + 4 * 2 * SORT_COUNTS_BYTES
end
