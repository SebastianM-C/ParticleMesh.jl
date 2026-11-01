# The charge-conserving flux deposition of WP-009: T-110 to T-117 (computations in flux_cases.jl).

const MODES = (Reference(), Fast())
const INPROCESS_FLUX = flux_digests()
worker_flux(nt) = [chopprefix(l, "FLUX ") for l in worker_lines(nt) if startswith(l, "FLUX ")]

function body_T110(be)
    C = tolC("T-110")
    for T in TYPES, shape in SHAPES, stagger in STAGGERS, mode in MODES, sort_between in (true, false)
        # Without a sort between the move and the deposit only Fast with atomics is possible.
        !sort_between && effective_mode(mode, T) isa Reference && continue
        c = folded(continuity_case(T, mode, shape, stagger; sort_between, be); be)
        r = record!("T-110", T, continuity_residual(c), be)
        @test r <= C
    end
end

@testset "T-110 discrete continuity of deposit_flux! [A2, A7, A8, WP-009]" begin
    body_T110(CPU())
    # The displacements do cross cell faces and the periodic seam.
    s = moved_species(Float64, PeriodicGrid(10, 0.5, 3), 400; seed = 3)
    crossed = count(i -> s.cell[1][i] != s.extras.old_cell[i], 1:400)
    seam = count(i -> abs(s.cell[1][i] - s.extras.old_cell[i]) > 1, 1:400)
    @test crossed > 100 && seam > 10
    # Too few guards and missing extras are argument errors, checked before any launch.
    grid = PeriodicGrid(10, 0.5, 2)
    s = moved_species(Float64, grid, 50; seed = 1)
    ws = Workspace(s, grid)
    J = field(backend, Float64, grid)
    @test_throws ArgumentError deposit_flux!(Fast(), J, copy(J), copy(J), s, Quadratic, Node(), grid, ws;
                                             dt = 0.25, quantity_y = ChargeUy(), quantity_z = ChargeUz())
    bare = test_species(Float64, grid, 50)
    @test_throws ArgumentError deposit_flux!(Fast(), J, copy(J), copy(J), bare, Quadratic, Center(), grid, Workspace(bare, grid);
                                             dt = 0.25, quantity_y = ChargeUy(), quantity_z = ChargeUz())
    # A grid of two cells is too small for the stencils, whatever the guards.
    tiny = PeriodicGrid(2, 0.5, flux_guard_width(Quadratic, Node()))
    s = moved_species(Float64, tiny, 20; seed = 1)
    J = field(backend, Float64, tiny)
    @test_throws ArgumentError deposit_flux!(Fast(), J, copy(J), copy(J), s, Quadratic, Node(), tiny, Workspace(s, tiny);
                                             dt = 0.25, quantity_y = ChargeUy(), quantity_z = ChargeUz())
end

@testset "T-111 the continuity residual does not grow over a drift [A7, WP-009]" begin
    # A drift at constant velocity (|v|·Δt < Δx), one deposit per step; the residual of each
    # step against the previous step's ρ, worst over 20 and over 200 steps.
    C = tolC("T-111")
    for T in TYPES, shape in SHAPES, mode in MODES
        stagger = shape === Linear ? Node() : Center()
        grid = PeriodicGrid(12, basetype(T)(1 // 2), flux_guard_width(shape, stagger))
        n = Int32(ncells(grid))
        dt = T(1 // 4)
        worst = Dict{Int,Float64}()
        for N in (20, 200)
            s = moved_species(T, grid, 300; seed = 5)
            ws = Workspace(s, grid)
            v = [seeded(T, 0.9 * sin(1.3i)) for i in 1:capacity(s)]    # in cells per step
            sort_by_cell!(s, grid, ws)
            ρold = deposit!(mode, field(backend, T, grid), s, shape, stagger, grid, ws)
            fold_guards!(ρold, grid)
            w = 0.0
            for _ in 1:N
                store_old!(s)
                for i in 1:capacity(s)
                    s.status[i] == ParticleMeshCore.ALIVE || continue
                    s.cell[1][i], s.offset[1][i] = ParticleMeshCore.advance(s.cell[1][i], s.offset[1][i], v[i], n)
                end
                sort_by_cell!(s, grid, ws)
                v .= v[ws.perm]                                       # the velocities follow the sort
                ρnew = deposit!(mode, field(backend, T, grid), s, shape, stagger, grid, ws)
                Jx, Jy, Jz = field(backend, T, grid), field(backend, T, grid), field(backend, T, grid)
                deposit_flux!(mode, Jx, Jy, Jz, s, shape, stagger, grid, ws; dt, quantity_y = ChargeUy(), quantity_z = ChargeUz())
                c = folded((; ρold, ρnew = ρnew, Jx, Jy, Jz, grid, s, dt))
                w = max(w, continuity_residual(c))
                ρold = c.ρnew
            end
            worst[N] = w
        end
        record!("T-111", T, worst[200])
        @test worst[20] <= C
        @test worst[200] <= C
    end
end

function body_T112(be)
    # Σ_j Jy_j = Σ q·w·u_y (densities per node, the convention of deposit!'s total, T-134), the
    # same for Jz, and Σ_j Jx_j = Σ q·w·Δ·Δx/Δt (the first moment of the shape). At zero
    # displacement Jy is deposit! of q·w·u_y.
    C = tolC("T-112")
    for T in TYPES, shape in SHAPES, stagger in STAGGERS, mode in MODES
        ε = base_eps(T)
        c = folded(continuity_case(T, mode, shape, stagger; be); be)
        (; s, grid, dt) = c
        alive = [i for i in 1:capacity(s) if s.status[i] == ParticleMeshCore.ALIVE]
        np = length(alive)
        for (J, q) in ((c.Jy, ChargeUy()), (c.Jz, ChargeUz()))
            expect = sum(q(s, i) for i in alive)
            scale = np * maximum(i -> absmax(q(s, i)), alive)
            @test record!("T-112", T, absdiff(sum(J), expect) / (ε * scale), be) <= C
        end
        n = ncells(grid)
        disp(i) = begin
            e = mod(s.extras.old_cell[i] - s.cell[1][i] + 1, n) - 1        # old cell relative to new
            s.offset[1][i] - (e + s.extras.old_offset[i])                  # displacement in cells
        end
        sx = cellsize(grid) / dt
        expect = sum(s.q * s.w[i] * disp(i) * sx for i in alive)
        scale = np * maximum(i -> absmax(s.q * s.w[i] * sx), alive)
        @test record!("T-112", T, absdiff(sum(c.Jx), expect) / (ε * scale), be) <= C
    end
    # Zero displacement: Jy is the deposition of q·w·u_y, to roundoff.
    for T in TYPES, shape in SHAPES, stagger in STAGGERS
        grid = PeriodicGrid(10, basetype(T)(1 // 2), flux_guard_width(shape, stagger))
        s = moved_species(T, grid, 400; seed = 9, maxstep = 0)
        ws = Workspace(s, grid)
        on(be, sort_by_cell!, s, grid, ws)
        Jx, Jy, Jz = field(backend, T, grid), field(backend, T, grid), field(backend, T, grid)
        on(be, deposit_flux!, Reference(), Jx, Jy, Jz, s, shape, stagger, grid, ws; dt = T(1 // 4), quantity_y = ChargeUy(),
           quantity_z = ChargeUz())
        ρy = on(be, deposit!, Reference(), field(backend, T, grid), s, shape, stagger, grid, ws; quantity = ChargeUy())
        ratio = record!("T-112 zero displacement", T, maximum(absdiff.(Jy, ρy)) / (base_eps(T) * maximum(absmax, ρy)), be)
        @test ratio <= (be isa CPU ? 2 : tolC("T-190 T-112 zero displacement"))
    end
end

@testset "T-112 transverse currents [A7, WP-009]" begin
    body_T112(CPU())
end

"The per-face Reference–Fast agreement of T-113 on `be` (T-194 on CUDA)."
function body_T113_faces(be)
    C = tolC("T-113")
    for T in (Float32, Float64), shape in SHAPES, stagger in STAGGERS
        cR = continuity_case(T, Reference(), shape, stagger; np = 4000, be)
        cF = continuity_case(T, Fast(), shape, stagger; np = 4000, be)
        A = flux_abs(cR, shape, stagger)
        worst = 0.0
        for (d, (JR, JF)) in enumerate(((cR.Jx, cF.Jx), (cR.Jy, cF.Jy), (cR.Jz, cF.Jz))), a in eachindex(JR)
            A[d][a] == 0 && (@test JR[a] == 0 && JF[a] == 0; continue)
            worst = max(worst, abs(JR[a] - JF[a]) / (eps(T) * A[d][a]))
        end
        record!("T-113", T, worst, be)
        @test worst <= C
    end
end

@testset "T-113 Reference and Fast agree per face; Reference is bitwise [A4, A7, WP-009]" begin
    body_T113_faces(CPU())
    # Bitwise: reruns, the default and two explicit launch shapes, and 1 and 4 threads.
    for (case, digests) in by_case(INPROCESS_FLUX)
        @test length(digests) == length(WG_SIZES)
        @test allequal(digests)
    end
    @test INPROCESS_FLUX == flux_digests()
    for nt in (1, 4)
        @test worker_flux(nt) == INPROCESS_FLUX
    end
end

function body_T114(be)
    # The flux is linear in the weights: the partials of Jx, Jy, Jz from Dual weights are the
    # fluxes of the weights' partials, computed independently in Float64 at the same positions.
    C = tolC("T-114")
    for shape in SHAPES, stagger in STAGGERS
        grid = PeriodicGrid(10, 0.5f0, flux_guard_width(shape, stagger))
        s32 = moved_species(Float32, grid, 400; seed = 4)
        sD = Species(backend, D3, capacity(s32); q = -1, m = 1, extras = old_extras(D3))
        for i in 1:capacity(s32)
            sD.cell[1][i], sD.extras.old_cell[i] = s32.cell[1][i], s32.extras.old_cell[i]
            sD.offset[1][i], sD.extras.old_offset[i] = D3(s32.offset[1][i]), D3(s32.extras.old_offset[i])
            sD.u[2][i], sD.u[3][i] = D3(s32.u[2][i]), D3(s32.u[3][i])
            sD.w[i] = mkdual(s32.w[i], cos(0.3i), s32.w[i]^2, 0)
            sD.status[i] = s32.status[i]
        end
        fill!(sD.count, alive_count(s32))
        flux(s, g, dt) = begin
            ws = Workspace(s, g)
            on(be, sort_by_cell!, s, g, ws)
            J = ntuple(_ -> field(backend, eltype(s), g), 3)
            on(be, deposit_flux!, Reference(), J..., s, shape, stagger, g, ws; dt, quantity_y = ChargeUy(), quantity_z = ChargeUz())
            J
        end
        JD = flux(sD, grid, D3(1 // 4))
        g64 = PeriodicGrid(10, 0.5, flux_guard_width(shape, stagger))
        for dir in 1:3
            s64 = Species(backend, Float64, capacity(sD); q = -1, m = 1, extras = old_extras(Float64))
            for i in 1:capacity(sD)
                s64.cell[1][i], s64.extras.old_cell[i] = sD.cell[1][i], sD.extras.old_cell[i]
                s64.offset[1][i], s64.extras.old_offset[i] = value(sD.offset[1][i]), value(sD.extras.old_offset[i])
                s64.u[2][i], s64.u[3][i] = value(sD.u[2][i]), value(sD.u[3][i])
                s64.w[i] = partials(sD.w[i])[dir]
                s64.status[i] = sD.status[i]
            end
            fill!(s64.count, alive_count(sD))
            J64 = flux(s64, g64, 0.25)
            for (JDd, J64d) in zip(JD, J64)
                ∂ = [partials(x)[dir] for x in JDd]
                if dir == 3
                    @test all(iszero, ∂)
                else
                    r = record!("T-114", D3, maximum(abs.(∂ .- J64d)) / (eps(Float32) * maximum(abs, J64d)), be)
                    @test r <= C
                end
            end
        end
    end
end

@testset "T-114 Dual weights give the exact partials of the flux [A3, A8, WP-009]" begin
    body_T114(CPU())
end

function body_T115(be)
    for T in TYPES, shape in SHAPES, stagger in STAGGERS, mode in MODES
        grid = PeriodicGrid(10, basetype(T)(1 // 2), flux_guard_width(shape, stagger))
        s = moved_species(T, grid, 400; seed = 2, maxstep = 0)
        ws = Workspace(s, grid)
        effective_mode(mode, T) isa Reference && on(be, sort_by_cell!, s, grid, ws)
        Jx, Jy, Jz = field(backend, T, grid), field(backend, T, grid), field(backend, T, grid)
        on(be, deposit_flux!, mode, Jx, Jy, Jz, s, shape, stagger, grid, ws; dt = T(1 // 4), quantity_y = ChargeUy(),
           quantity_z = ChargeUz())
        @test all(iszero, Jx)
        @test any(!iszero, Jy)
    end
end

@testset "T-115 a particle at rest deposits exactly zero Jx [A7, WP-009]" begin
    body_T115(CPU())
end

@testset "T-116 the flux is smooth through a cell crossing [A2, A7, WP-009]" begin
    # A smooth functional of Jx against the displacement, ForwardDiff against fd_scan (Float64,
    # Reference, one seed), with end points on both sides of a face as close as 1e-7, so the
    # larger steps of the scan straddle the crossing. Order 2 is asserted (its weights are C¹);
    # order 1 is recorded only.
    hs = 10.0 .^ (-1:-0.5:-9)
    Δ0 = 0.3
    for shape in SHAPES, stagger in STAGGERS
        f(x, mode, seed) = flux_functional(x, mode, seed; shape, stagger)
        ad = ForwardDiff.derivative(x -> f(x, Reference(), 7), Δ0)
        r = fd_scan(f, Δ0, hs; seed = 7, ad)
        label = "order $(order(shape)), $(nameof(typeof(stagger)))"
        @info "T-116 $label" ad best_h = hs[r.istar] r.floor_rel rel_at_best = r.rel[r.istar] ratio = r.rel[r.istar] / r.floor_rel
        MEASURED["T-116 rel/floor"] = merge(get(MEASURED, "T-116 rel/floor", Dict{String,Float64}()), Dict(label => r.rel[r.istar] / r.floor_rel))
        MEASURED["T-116 floor_rel"] = merge(get(MEASURED, "T-116 floor_rel", Dict{String,Float64}()), Dict(label => r.floor_rel))
        @test isfinite(ad) && ad != 0
        if shape === Quadratic
            @test r.rel[r.istar] <= tolC("T-116") * r.floor_rel
            @test r.floor_rel <= tolC("T-116", "floor_max")
        end
    end
end

@testset "T-117 allocation of deposit_flux! does not grow with particle count or grid size [A7, WP-009]" begin
    for T in (Float32, Float64), mode in MODES
        bytes = Dict{Tuple{Int,Int},Int}()
        for (n, np) in ((16, 500), (16, 8000), (64, 500))
            grid = PeriodicGrid(n, T(1 // 2), flux_guard_width(Quadratic, Center()))
            s = moved_species(T, grid, np; seed = 3)
            ws = Workspace(s, grid)
            sort_by_cell!(s, grid, ws)
            J = ntuple(_ -> field(backend, T, grid), 3)
            run() = deposit_flux!(mode, J..., s, Quadratic, Center(), grid, ws; dt = T(1 // 4), quantity_y = ChargeUy(),
                                  quantity_z = ChargeUz())
            run(); run()
            bytes[(n, np)] = @allocated run()
        end
        @info "T-117 bytes per deposit_flux!" T mode bytes
        @test bytes[(16, 8000)] <= bytes[(16, 500)]
        @test bytes[(64, 500)] <= bytes[(16, 500)]
    end
end
