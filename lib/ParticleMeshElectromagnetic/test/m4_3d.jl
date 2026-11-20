# WP-079: the M4 solver in the 3D step, T-1094 to T-1097 (the 2D step is m4_step.jl; T-1098 and
# T-1099 are the reruns on CUDA, in gpu.jl). As there, each test is the Yee test of the same
# property with `solver = M4()`, against M4's theory (`m4_kx`) where the Yee test used Yee's. After
# em3d_step.jl (`invariant_error_3d`), em3d_open.jl (`alive_3d`, `open_residuals_3d`, `free_rows`)
# and em3d_antenna.jl (`angular_error_3d`, `incident_dtft_3d`, `forward_window_3d`), and
# m4_step.jl's `m4_divergence` and `m4_div_scale`, which are defined again here when the group
# runs without it.

using ForwardDiff: ForwardDiff
using ParticleMeshElectromagnetic: faraday_stencil, forward_difference

if !isdefined(@__MODULE__, :m4_divergence)
    function m4_divergence(st::EMState{N}, ranges) where {N}
        stc = faraday_stencil(M4(), st.grid, st.dt)
        B = map(Array, (st.fields.Bx, st.fields.By, st.fields.Bz))
        g, h = guards(st.grid), cellsize(st.grid)
        I = CartesianIndices(map(r -> g .+ r, ranges))
        return maximum(j -> absmax(sum(d -> forward_difference(stc, B[d], j, Val(d), Val(N)) / h[d], 1:N)), I)
    end
    function m4_div_scale(st::EMState{N}) where {N}
        g, n, h = guards(st.grid), ncells(st.grid), cellsize(st.grid)
        I = ntuple(d -> (g + 1):(g + n[d]), N)
        return sum(d -> 2 / Float64(value_of(h[d])), 1:N) * maximum(c -> Float64(maximum(absmax, Array(getfield(st.fields, c))[I...])), (:Bx, :By, :Bz))
    end
end

"`conserved_energy_3d` of M4: the half steps of B by M4's difference."
function conserved_energy_m4_3d(f, grid, dt; cols)
    g, n, h = guards(grid), ncells(grid), cellsize(grid)
    st = faraday_stencil(M4(), grid, dt)
    fp, fm = map(copy, f), map(copy, f)
    m4_b_half!(fp, grid, dt / 2, st)
    m4_b_half!(fm, grid, -dt / 2, st)
    I = (g .+ cols, (g + 1):(g + n[2]), (g + 1):(g + n[3]))
    W = sum(c -> sum(abs2, Array(c)[I...]), (f.Ex, f.Ey, f.Ez))
    W += sum(k -> sum(Array(getfield(fp, k))[I...] .* Array(getfield(fm, k))[I...]), (:Bx, :By, :Bz))
    return W * prod(h) / 2
end

# --- T-1094 ---------------------------------------------------------------------------------

# T-1022 with M4: the launched 3D beam is M4's angular-spectrum propagation of the incident field in
# two transverse wavenumbers (`m4_kx`), to the antenna's order k_⊥⁴, in each polarization and on a
# cross-section whose y and z differ, and the component the incident field does not have stays at
# that order (a wrong cross term of the B corrections would launch it at order k_⊥²). A plane wave
# polarized at π/3 (both components) is launched exactly at ω₀, to roundoff, in the A8 matrix; the
# Yee antenna in the same M4 run misses by orders of magnitude. Yee's propagation misses the M4
# beam by several times the antenna's order (at 12 and 6 cells per wavelength).
function body_T1094(be)
    C, Cx = emC("T-1022"), emC("T-1022", "C_cross")
    Cp, Cn = emC("T-1085", "C_plane"), emC("T-1085", "C_naive")
    w0 = 4
    if be isa CPU
        for (θ, main, cross, box) in ((0.0, :Ey, :Ez, (; length = (140, 30, 30))), (π / 2, :Ez, :Ey, (; length = (140, 30, 30))),
                                      (0.0, :Ey, :Ez, (; length = (140, 30, 26), nct = (6, 5), center = (16.5, 11.0))))
            prob = antenna3d_problem(Float64; waist = w0, duration = 8, position = 6, focus = 16, polarization = θ, tend = 130, box...)
            r = antenna3d_run(prob; probes = (6, 12, 20, 30), solver = M4())
            e = angular_error_3d(r; component = main, kx = m4_kx)
            x = angular_error_3d(r; component = cross, against = main, kx = m4_kx)
            yee = angular_error_3d(r; component = main)
            worst, worst_x = maximum(p -> p.err, e) * w0^4, maximum(p -> p.err, x) * w0^4
            @info "T-1094" θ box worst worst_x yee = maximum(p -> p.err, yee) * w0^4
            tag = haskey(box, :nct) ? " y≠z" : ""
            record!("T-1094 θ=$(round(θ; digits = 2))$tag", Float64, worst)
            record!("T-1094 cross θ=$(round(θ; digits = 2))$tag", Float64, worst_x)
            @test worst <= C
            @test worst_x <= Cx
            @test maximum(p -> p.err, yee) * w0^4 > 3 * worst
        end
    end
    for T in TYPES
        prob = antenna3d_problem(T; length = (110, 6.3, 6.3), waist = 1e8, duration = 5, position = 8, focus = 14, polarization = 1 // 3 * T(π),
                                 tend = 90)
        r = antenna3d_run(prob; probes = (8, 20, 30), solver = M4(), be)
        err = maximum(c -> maximum(x -> x.err, angular_error_3d(r; component = c, kx = m4_kx)), (:Ey, :Ez)) / (base_eps(T) * sqrt(prob.nsteps))
        record!("T-1094 plane", T, err, be)
        @test err <= Cp
        if T === Float64
            rn = antenna3d_run(prob; probes = (8, 20, 30), solver = M4(), antenna_solver = Yee(), be)
            naive = maximum(c -> maximum(x -> x.err, angular_error_3d(rn; component = c, kx = m4_kx)), (:Ey, :Ez)) /
                    (base_eps(T) * sqrt(prob.nsteps))
            @info "T-1094 plane wave" err naive
            @test naive > Cn
        end
    end
end

@testset "T-1094 the launched 3D M4 beam is M4's angular-spectrum propagation of the incident field to the antenna's order k_⊥⁴, and its ω₀ plane wave is exact [A2, A7, A8, WP-079]" begin
    body_T1094(CPU())
end

# --- T-1095 ---------------------------------------------------------------------------------

# T-1021 with M4: the 3D antenna launches one way, the energy behind the plane within T-1021's
# fraction, at θ = 0 and 3/5, in the A8 matrix, and the fields are linear in a0 (a0·∂E/∂a0 = E);
# and its B corrections are the M4 half step's TF/SF split to roundoff (T-1086's operator test,
# here at θ = 3/5 on y ≠ z cells with the axis off the cells' centres), Yee's far from it.
@testset "T-1095 the 3D M4 antenna launches one way, is linear in a0, and its B corrections are the M4 split to roundoff [A2, A3, A8, WP-079]" begin
    C, Cl = emC("T-1021"), emC("T-1021", "C_linear")
    for T in TYPES
        prob = antenna3d_problem(T; length = (20, 12, 10), nct = (6, 5), waist = 3, duration = 3, position = 6, polarization = 3 // 5, tend = 1,
                                 center = (6.5, 4.0))
        st = EMState(prob, EMNumerics(; solver = M4()))
        t = value_of(prob.lasers[1].delay)
        r = tfsf_operator_error(st, prob, t)
        worst = max(r.scattered, r.total) / base_eps(T)
        record!("T-1095 operator", T, worst)
        @test worst <= emC("T-1086", "C_operator")
        inc = incident_column_error(st, prob, t) / base_eps(T)
        record!("T-1095 incident columns", T, inc)
        @test inc <= emC("T-1086", "C_incident")
        T === Float64 && @test tfsf_operator_error(st, prob, t; source_solver = Yee()).scattered > 1e-2
    end
    for T in TYPES, θ in (0, 3 // 5)
        D = T === D3
        a0 = D ? mkdual(1, 1, 0, 0) : one(T)
        focus = D ? mkdual(14, 0, 1, 0) : T(14)
        chirp = D ? mkdual(1 // 100, 0, 0, 1) : T(1 // 100)
        prob = antenna3d_problem(T; length = (100, 24, 24), position = 6, focus, chirp, a0, polarization = θ, tend = 60)
        r = antenna3d_run(prob; solver = M4())
        fwd, back = forward_window_3d(r, 58)
        f = map(Array, r.st.fields)
        frac = Float64(value_of(plain_energy_3d(f, r.st.grid; cols = back) / conserved_energy_m4_3d(f, r.st.grid, prob.dt; cols = fwd)))
        @info "T-1095" T θ frac
        record!("T-1095 θ=$θ", T, frac)
        @test frac <= C
        if D
            worst = 0.0
            for c in (f.Ex, f.Ey, f.Ez, f.Bx, f.By, f.Bz)
                m = maximum(x -> abs(value_of(x)), c)
                d = maximum(x -> abs(ForwardDiff.partials(x)[1] - value_of(x)), c)
                worst = max(worst, m == 0 ? (d == 0 ? 0.0 : Inf) : d / (eps(Float32) * m * sqrt(prob.nsteps)))
            end
            record!("T-1095 linear", T, worst)
            @test worst <= Cl
            @test any(x -> !iszero(ForwardDiff.partials(x)[2]), f.Ey) && any(x -> !iszero(ForwardDiff.partials(x)[3]), f.Ey)
        end
    end
end

# --- T-1096 ---------------------------------------------------------------------------------

# The 3D M4 step where the problem does not vary along one axis is the 2D M4 step (T-953 and
# T-1010 with M4): on an invariant axis every second difference across it is zero, so its β terms
# add exactly zero, and the coefficients of the other axes are those of the plane (the same Δt and
# cell sizes). Periodic, along each of the three axes, and open with each absorber, the open axis
# y or z, all three types, both modes.
function body_T1096(be)
    C, Co = emC("T-953"), emC("T-1010")
    for T in TYPES, mode in (Reference(), Fast()), axis in (3, 1, 2)
        amplitude = seeded(T, 1 / 20)
        p2, p3 = beams_plane_2d(T; amplitude), beams_3d(T, axis; amplitude)
        s2 = EMState(p2, EMNumerics(; mode, solver = M4()); backend = be)
        s3 = EMState(p3, EMNumerics(; mode, solver = M4()); backend = be)
        initialize!(s2, p2)
        initialize!(s3, p3)
        scale = sum(ncells(s2.grid)) * sqrt(prod(p2.species[1].ppc))
        worst = invariant_error_3d(s2, s3, axis) / (base_eps(T) * scale)
        for k in 1:p2.nsteps
            step!(s2, params(p2), k)
            step!(s3, params(p3), k)
            worst = max(worst, invariant_error_3d(s2, s3, axis) / (base_eps(T) * k * scale))
        end
        record!("T-1096 axis $axis $(nameof(typeof(mode)))", T, worst, be)
        @test worst <= C
    end
    for T in TYPES, mode in (Reference(), Fast()), axis in (3, 1), (label, ab) in OPEN3_ABSORBERS
        amplitude = seeded(T, 1 / 20)
        p2, p3 = beams_plane_2d(T; amplitude, nsteps = 30, OPEN3_PLANE...), beams_3d(T, axis; amplitude, nsteps = 30, OPEN3_PLANE...)
        s2 = EMState(p2, EMNumerics(; mode, absorber = ab, solver = M4()); backend = be)
        s3 = EMState(p3, EMNumerics(; mode, absorber = ab, solver = M4()); backend = be)
        initialize!(s2, p2)
        initialize!(s3, p3)
        n0 = alive_3d(s2)
        scale = sum(ncells(s2.grid)) * sqrt(prod(p2.species[1].ppc))
        worst = invariant_error_3d(s2, s3, axis) / (base_eps(T) * scale)
        for k in 1:p2.nsteps
            step!(s2, params(p2), k)
            step!(s3, params(p3), k)
            worst = max(worst, invariant_error_3d(s2, s3, axis) / (base_eps(T) * k * scale))
        end
        @test alive_3d(s2) < n0
        record!("T-1096 open $label axis $axis $(nameof(typeof(mode)))", T, worst, be)
        @test worst <= Co
    end
end

@testset "T-1096 a 3D M4 run invariant along one axis is the 2D M4 run, periodic and open with each absorber [A2, A7, A8, WP-079]" begin
    body_T1096(CPU())
end

# --- T-1097 ---------------------------------------------------------------------------------

# T-1011 with M4: with particles leaving through the open faces of y and z, Gauss's law holds to
# roundoff on the nodes off every layer and M4's divergence on the centres whose stencil reads no
# layer row (D̃_y reaches the rows j − 1 to j + 2, the β terms one row across), each absorber, all three types, both modes. And the 3D M4
# step with the antenna, the PML on y and z and a drifting plasma is bitwise across reruns and
# launch shapes in Reference mode, per element type.
function m4_3d_digests(; workgroupsize = nothing, be = CPU())
    out = Tuple{DataType,UInt64}[]
    for T in TYPES
        tenth = one(T) / 10
        sp = (SpeciesDescription{T}(:electron, -one(T), one(T), tenth, zero(T), (1, 0, 0), (1, 2, 2), (3tenth, tenth, -tenth),
                                    (tenth / 2, tenth / 2, tenth / 2), 1),)
        prob = antenna3d_problem(T; ncx = 8, nct = 6, length = (16, 12, 12), waist = 3, duration = 2, position = 4, focus = 8,
                                 polarization = 3 // 5, a0 = 1 // 2, tend = 15, species = sp, boundaries = (:periodic, :open, :open))
        st = EMState(prob, EMNumerics(; workgroupsize, current = Esirkepov(), absorber = PML(; cells = 4, kappa_max = 2, alpha_max = 0.1),
                                      solver = M4()); backend = be)
        initialize!(st, prob)
        run!(st, params(prob), prob.nsteps)
        h = state_digest_3d(st)
        for a in evolved(st)
            h = hash(bits(a), h)
        end
        push!(out, (T, h))
    end
    return out
end

function body_T1097(be)
    C, CB = emC("T-1011"), emC("T-1089", "C_divB")
    for T in TYPES, mode in (Reference(), Fast()), (label, ab) in OPEN3_ABSORBERS
        prob = open_plasma_3d(T; amplitude = seeded(T, 1 / 10))
        st = open_state_3d(prob, ab; mode, be, solver = M4())
        ρbar0 = open_residuals_3d(st, zero(T)).ρbar
        r = open_residuals_3d(st, ρbar0)
        worst = r.free / (base_eps(T) * r.scale)
        worstB = 0.0
        n = ncells(st.grid)
        c = ab.cells
        free = (1:n[1], (c + 2):(n[2] - c - 2), (c + 2):(n[3] - c - 2))
        n0 = alive_3d(st)
        p = params(prob)
        for k in 1:prob.nsteps
            step!(st, p, k)
            r = open_residuals_3d(st, ρbar0)
            worst = max(worst, r.free / (base_eps(T) * r.scale * sqrt(k + 1)))
            sB = m4_div_scale(st)
            sB > 0 && (worstB = max(worstB, Float64(m4_divergence(st, free)) / (base_eps(T) * sB * sqrt(k))))
        end
        @info "T-1097" label T mode worst worstB left = n0 - alive_3d(st)
        @test n0 - alive_3d(st) > n0 ÷ 10
        record!("T-1097 $label $(nameof(typeof(mode)))", T, worst, be)
        record!("T-1097 divB $label $(nameof(typeof(mode)))", T, worstB, be)
        @test worst <= C
        @test worstB <= CB
    end
    ref = m4_3d_digests(; be)
    @test allunique(last.(ref))
    @test m4_3d_digests(; be) == ref
    for wgs in (be isa CPU ? (32, 256) : (64, 256))
        @test m4_3d_digests(; workgroupsize = wgs, be) == ref
    end
end

@testset "T-1097 Gauss's law and M4's ∇·B off the 3D layers with particles leaving, and the 3D M4 step is bitwise across reruns and launch shapes [A2, A4, A7, A8, WP-079]" begin
    body_T1097(CPU())
end
