# WP-013, the engine on CUDA: T-190 to T-194 and T-198; WP-042, T-876 and T-946; WP-041, T-891 and T-892; WP-077, T-1066. Included last by runtests.jl, after the
# files that define the shared bodies; the testsets run only when the GPU switch says so
# (gpu_switch.jl), otherwise the suite prints one line.

"Run the shared bodies `(id, body)` on `be`, one nested testset per CPU test id."
function rerun_on(be, bodies)
    for (id, body) in bodies
        @testset "$id on $(backend_label(be))" begin
            body(be)
        end
    end
end

"Launch shapes of the CUDA bitwise checks: the backend default and two explicit sizes."
const GPU_WG_SIZES = (nothing, 64, 256)

# Device-callable draws of T-193, one per counter value.
struct PhiloxAt end
(::PhiloxAt)(c::UInt32) = philox4x32((c, c ⊻ 0x9e3779b9, UInt32(7), c * 0x01000193), (0x243f6a88, c))
struct UniformAt{T} end
(::UniformAt{T})(x::UInt32) where {T} = uniform_open(T, x)
struct BoxMullerAt{T} end
(::BoxMullerAt{T})(x::NTuple{2,UInt32}) where {T} = box_muller(T, x[1], x[2])

# Device-callables of T-191: the 2D advance and the tensor-product weights.
struct Advance2{T}
    Δ::NTuple{2,T}
    n::NTuple{2,Int32}
    steps::Int
end
function (a::Advance2)(start)
    c, δ = start
    for _ in 1:a.steps
        c, δ = advance(c, δ, a.Δ, a.n)
    end
    return (c, δ)
end
struct TensorAt{S,G}
    st::G
end
TensorAt(shape, st) = TensorAt{shape,typeof(st)}(st)
(t::TensorAt{S})(δ) where {S} = ParticleMeshCore.tensor_weights(S, t.st, δ)

"Distance in units in the last place of `b` (at least one ulp for zero)."
ulps(a, b) = abs(Float64(a) - Float64(b)) / Float64(eps(max(abs(b), floatmin(typeof(b)))))

if GPU_BACKEND === nothing
    println(gpu_skip_line(GPU_SWITCH))
else
    CUDACore.allowscalar(false)
    const GPU = GPU_BACKEND
    println("GPU tests (WP-013) on ", CUDACore.name(CUDACore.device()), ": ", GPU_SWITCH.reason)

    @testset "T-190 the engine's N = 1 identities on CUDA [A7, A8, WP-013]" begin
        rerun_on(GPU, (("T-130", body_T130), ("T-131", body_T131), ("T-132", body_T132), ("T-133", body_T133),
                       ("T-134", body_T134), ("T-140", body_T140), ("T-172", body_T172), ("T-173", body_T173),
                       ("T-174", body_T174), ("T-175", body_T175), ("T-110", body_T110), ("T-112", body_T112),
                       ("T-114", body_T114), ("T-115", body_T115)))
    end

    @testset "T-191 the engine's N = 2 identities on CUDA [A7, A8, WP-013]" begin
        n = Int32.(N2)
        for T in TYPES
            # T-261's advance, on the device against the host, bitwise, through both seams and the corner.
            starts = [((Int32(c1), Int32(c2)), (seeded(T, a), seeded(T, b)))
                      for (c1, c2, a, b) in ((16, 12, 15 / 16, 15 / 16), (1, 1, 1 / 16, 0.0), (16, 5, 0.5, 0.375), (7, 12, 0.0, 0.8125),
                                             (3, 4, 0.3, 0.7))]
            for Δ in ((seeded(T, 1 // 8), seeded(T, 1 // 8)), (seeded(T, -0.19), seeded(T, 0.33)))
                f = Advance2(Δ, n, 300)
                dev, host = devmap(GPU, f, starts), map(f, starts)
                @test all(zip(dev, host)) do ((dc, dδ), (hc, hδ))
                    dc == hc && all(d -> components(dδ[d]) == components(hδ[d]), 1:2)
                end
            end
            # T-262's tensor-product weights on the device against the host: the stencil start
            # bitwise, the weights within a measured bound (the device fuses multiply-adds).
            offs = [(seeded(T, a), seeded(T, b)) for a in (0.0, 0.25, 0.7, (k / 13 for k in 0:12)...) for b in (0.0, 0.5, 0.9)]
            for shape in SHAPES, st in STAGGER_PAIRS
                dev, host = devmap(GPU, TensorAt(shape, st), offs), map(TensorAt(shape, st), offs)
                @test all(zip(dev, host)) do ((df, _), (hf, _))
                    df === hf
                end
                worst = maximum(zip(dev, host)) do ((_, dW), (_, hW))
                    maximum(absdiff.(collect(dW), collect(hW))) / (base_eps(T) * maximum(absmax, collect(hW)))
                end
                record!("T-191 tensor weights", T, worst, GPU)
                @test worst <= tolC("T-191 tensor weights")
            end
        end
        rerun_on(GPU, (("T-263", body_T263), ("T-264", body_T264), ("T-265", be -> body_T265(be; nodes = false)),
                       ("T-267", body_T267), ("T-269", body_T269), ("T-530", body_T530), ("T-531", body_T531)))
    end

    @testset "T-876 the engine's N = 3 fill and fold, load, sort, gather and deposition on CUDA [A7, A8, WP-042]" begin
        rerun_on(GPU, (("T-871", body_T871), ("T-872", body_T872), ("T-873", body_T873)))
    end

    @testset "T-946 the 3D flux on CUDA: continuity, the sums, Fast against Reference, Reference bitwise across reruns, launch shapes and planes, and Reference against the sums in the stated order to roundoff [A4, A7, A8, WP-042]" begin
        rerun_on(GPU, (("T-940", body_T940), ("T-941", body_T941), ("T-944", body_T944)))
        runs = [flux3d_digests(GPU; wgs = GPU_WG_SIZES) for _ in 1:2]
        @test runs[1] == runs[2]
        for (case, digests) in by_case(runs[1])
            @test length(digests) == length(GPU_WG_SIZES)
            @test allequal(digests)
        end
        for T in TYPES
            runs = [continuity_case_3d(T, Reference(), Quadratic, NODES3; bins = NodeBins(Quadratic, NODES3), planes, be = GPU)
                    for planes in (nothing, 1, 5)]
            @test all(r -> same_bits(r.Jx, runs[1].Jx) && same_bits(r.Jy, runs[1].Jy) && same_bits(r.Jz, runs[1].Jz), runs)
            # the sums in the stated order, by hand on the host: T-945 holds the CPU's Reference to
            # them bit for bit, but the device's arithmetic is not the host's (it does not give the
            # host's bits on an RTX 5090, probably from fused multiply-adds, see tolerances.toml),
            # so here they agree per node within T-944's roundoff bound, guards folded
            for (shape, st, bins) in ((Quadratic, NODES3, NodeBins(Quadratic, NODES3)), (Quadratic, (Node(), Node(), Center()), CellBins()))
                c = continuity_case_3d(T, Reference(), shape, st; bins, be = GPU)
                H = reference_by_hand_3d(c, shape, st, bins)
                A = flux_abs_3d(c.s, shape, st, c.grid, c.dt, bins)
                foreach(a -> fold_guards!(a, c.grid), (c.Jx, c.Jy, c.Jz, H..., A...))
                worst, zeros_ok, bitwise = 0.0, true, true
                for (d, J) in enumerate((c.Jx, c.Jy, c.Jz)), a in eachindex(J)
                    bitwise &= same_bits(J[a:a], H[d][a:a])
                    A[d][a] == 0 && (zeros_ok &= J[a] == 0 && H[d][a] == 0; continue)
                    worst = max(worst, absmax(J[a] - H[d][a]) / (eps(basetype(T)) * A[d][a]))
                end
                @info "T-946 the device's Reference against the host's sums in the stated order" T bins bitwise worst
                @test zeros_ok
                @test record!("T-946 order", T, worst, GPU) <= tolC("T-944")
            end
        end
    end

    @testset "T-192 Reference mode of the engine on CUDA is bitwise across reruns and launch shapes [A4, WP-013]" begin
        # N = 1: the scan, the sort with extras, both depositions (q·w and q·w·u_y) and the flux,
        # per element type, at the default and two explicit workgroup sizes; three reruns.
        # N = 2 (T-266's set): the sort with extras, the deposition of a species that crossed both
        # seams and the corner, the gather, fill, fold and the load.
        # WP-033: the momentum loads of T-536 (Maxwell–Jüttner at N = 1 and 2, Gaussian in momentum).
        runs = [vcat(thread_digests(GPU; wgs = GPU_WG_SIZES), flux_digests(GPU; wgs = GPU_WG_SIZES),
                     engine2d_digests(GPU; wgs = GPU_WG_SIZES), momenta_digests(GPU; wgs = GPU_WG_SIZES)) for _ in 1:3]
        @test runs[1] == runs[2] == runs[3]
        @test all(l -> endswith(l, "|true"), filter(startswith("sortmatch|"), runs[1]))
        for (case, digests) in by_case(filter(l -> !startswith(l, "sortmatch"), runs[1]))
            @test length(digests) == length(GPU_WG_SIZES)
            @test allequal(digests)
        end
        # gather!, fill_guards! and fold_guards! at every launch shape.
        for T in TYPES, shape in SHAPES, stagger in STAGGERS
            grid = PeriodicGrid(12, basetype(T)(1 // 2), guard_width(shape))
            outs = map(GPU_WG_SIZES) do wg
                s = test_species(T, grid, 300; seed = 5)
                f = test_field(T, grid, j -> seeded(T, sin(0.7j) + 0.3))
                out = fill!(similar(s.w), zero(T))       # the free slots are not written
                on(GPU, gather!, out, f, s, shape, stagger, grid; workgroupsize = wg)
                ff = on(GPU, fill_guards!, copy(f), grid; workgroupsize = wg)
                fo = on(GPU, fold_guards!, copy(f), grid; workgroupsize = wg)
                fnv_digest(out, ff, fo)
            end
            @test allequal(outs)
        end
    end

    @testset "T-193 Philox and loading on CUDA against the CPU [A4, WP-013]" begin
        ctrs = UInt32[0, 1, 2, 0xffffffff, 0x243f6a88, (UInt32(k) * 0x9e3779b9 for k in 1:4091)...]
        @test devmap(GPU, PhiloxAt(), ctrs) == map(PhiloxAt(), ctrs)
        for T in (Float32, Float64)
            @test same_bits(devmap(GPU, UniformAt{T}(), ctrs), map(UniformAt{T}(), ctrs))
            pairs = [(c, c ⊻ 0xdeadbeef) for c in ctrs]
            dev, host = devmap(GPU, BoxMullerAt{T}(), pairs), map(BoxMullerAt{T}(), pairs)
            worst = maximum(max(ulps(d[1], h[1]), ulps(d[2], h[2])) for (d, h) in zip(dev, host))
            record!("T-193 box_muller ulp", T, worst, GPU)
            @test worst <= tolC("T-193 box_muller")
        end
        # A quiet-start load: cells, offsets and statuses bitwise against the CPU. The weights are
        # density(x)/ppc, and the density's `cos` is CUDA's on the device and Base's on the host,
        # so they agree within a measured number of eps (first CUDA run: last-bit differences in
        # Float32 and Float64), as do the Box–Muller velocities (`log`, `cos`, `sin`).
        for T in TYPES
            grid = PeriodicGrid(16, basetype(T)(1 // 8), 2)
            p = (seeded(T, 1), seeded(T, 1 // 20), T(2π / length(grid)))
            vth = (T(1 // 20), T(1 // 20), T(1 // 20))
            load(be) = on(be, load_quiet!, Species(backend, T, 16 * 32; q = -1, m = 1), grid, CosineDensity(), p, 32;
                          vth, seed = 4, stream = 1)
            h, d = load(CPU()), load(GPU)
            @test h.cell[1] == d.cell[1] && h.status == d.status && h.count == d.count
            @test same_bits(h.offset[1], d.offset[1])
            wdiff = maximum(i -> absdiff(h.w[i], d.w[i]) / (base_eps(T) * absmax(h.w[i])), 1:(16 * 32))
            record!("T-193 load w", T, wdiff, GPU)
            @test wdiff <= tolC("T-193 load", "C_w")
            uworst = maximum(ulps(value_of(d.u[k][i]), value_of(h.u[k][i])) for k in 1:3, i in 1:(16 * 32))
            record!("T-193 load u ulp", T, uworst, GPU)
            @test uworst <= tolC("T-193 load", "C_u")
        end
    end

    @testset "T-198 engine allocation per step on CUDA does not grow with particle count or grid size [A7, WP-013]" begin
        # A proxy step on device-resident data (sort, Reference and Fast deposit, fold, fill,
        # gather, and the flux at N = 1), after a warm-up: host bytes (`@allocated`, the launches'
        # bookkeeping) and device bytes (`CUDACore.@allocated`) at two particle counts per grid size
        # and two grid sizes. The sort's host bytes are measured on their own: AcceleratedKernels'
        # GPU merge sort launches one merge-pass kernel per doubling of the run length
        # (`merge_sort.jl`, AcceleratedKernels 0.5), so its host bookkeeping grows as log₂ of the
        # capacity while its device scratch is the workspace's; it is held to about 1 KiB per
        # doubling. The rest of the step is held constant at 64 and 512 particles per cell (at 8
        # per cell the second CUDA run measured 592 host bytes fewer, a step not yet explained).
        function proxy(T, N, ncell, ppc)
            if N == 1
                grid = PeriodicGrid(ncell, basetype(T)(1 // 2), flux_guard_width(Quadratic, Center()))
                s = todev(GPU, moved_species(T, grid, ncell * ppc; seed = 3))
            else
                grid = PeriodicGrid((ncell, ncell), dx2(T), guard_width(Quadratic))
                s = todev(GPU, moved_species_2d(T, grid, ncell^2 * ppc))
            end
            ws = Workspace(s, grid)
            ρ = todev(GPU, field(backend, T, grid))
            J = ntuple(_ -> todev(GPU, field(backend, T, grid)), 3)
            out = similar(s.w)
            st = N == 1 ? Center() : (Center(), Center())
            sort() = (sort_by_cell!(s, grid, ws); KernelAbstractions.synchronize(GPU))
            function step()
                deposit!(Reference(), ρ, s, Quadratic, st, grid, ws)
                effective_mode(Fast(), T) isa Fast && deposit!(Fast(), ρ, s, Quadratic, st, grid, ws)
                fold_guards!(ρ, grid)
                fill_guards!(ρ, grid)
                gather!(out, ρ, s, Quadratic, st, grid)
                N == 1 && deposit_flux!(Reference(), J..., s, Quadratic, st, grid, ws; dt = T(1 // 4), quantity_y = ChargeUy(),
                                        quantity_z = ChargeUz())
                KernelAbstractions.synchronize(GPU)
            end
            sort(); step(); sort(); step()
            # the minimum of three samples: a single host sample once came out 288 bytes high
            # without a pattern (the third CUDA run)
            host = minimum(_ -> @allocated(step()), 1:3)
            dev = maximum(_ -> CUDACore.@allocated(step()), 1:3)
            sort_host = minimum(_ -> @allocated(sort()), 1:3)
            sort_dev = maximum(_ -> CUDACore.@allocated(sort()), 1:3)
            return (; host, dev, sort_host, sort_dev)
        end
        for T in TYPES, N in (1, 2)
            sizes = N == 1 ? (32, 128) : (8, 16)
            rows = [(ncell, ppc, proxy(T, N, ncell, ppc)) for ncell in sizes for ppc in (64, 512)]
            for (ncell, ppc, b) in rows
                println("T-198 bytes on CUDA: ", type_label(T), " N=$N cells=$ncell ppc=$ppc: step host=$(b.host) dev=$(b.dev), ",
                        "sort host=$(b.sort_host) dev=$(b.sort_dev)")
            end
            ref = rows[1][3]
            refcap = sizes[1]^N * 64
            for (ncell, ppc, b) in rows
                @test b.dev <= ref.dev && b.sort_dev <= ref.sort_dev
                @test b.host <= ref.host
                doublings = log2(ncell^N * ppc / refcap)
                @test b.sort_host <= ref.sort_host + 1024 * (doublings + 1)
            end
            record!("T-198 sort host bytes", T, maximum(r -> r[3].sort_host, rows), GPU)
            record!("T-198 device bytes", T, maximum(r -> r[3].dev, rows), GPU)
            record!("T-198 host bytes", T, maximum(r -> r[3].host, rows), GPU)
        end
    end

    @testset "T-891 on CUDA from 2^20 slots the workspace sorts its keys by radix sort, with the merge sort's bits [A4, WP-041]" begin
        # The keys are unique, so the radix sort (key_sort_alg's choice here) and the merge sort
        # (AcceleratedKernels' Auto below 2^23 keys) must give the same permutation; one sort of
        # a species whose slots are not in cell order, every array compared.
        AK = ParticleMeshCore.AK
        n = (128, 96)
        cap = 1 << 20
        for T in (Float32, Float64)
            grid = PeriodicGrid(n, (basetype(T)(1 // 2), basetype(T)(1 // 2)), 1)
            host = resort_species(T, n, cap, 3 * cap ÷ 4; seed = 7)
            a, b = todev(GPU, host), todev(GPU, host)
            wa = Workspace(a, grid)
            wb = Workspace(b, grid; key_sort = AK.MergeSort())
            @test wa.ak.plan.alg isa AK.RadixSort && wb.ak.plan.alg isa AK.MergeSort
            sort_by_cell!(a, grid, wa); sort_by_cell!(b, grid, wb)
            KernelAbstractions.synchronize(GPU)
            @test all(same_bits(Array(x), Array(y)) for (x, y) in zip(species_arrays(a), species_arrays(b)))
            @test Array(a.count) == Array(b.count)
            @test Array(wa.perm) == Array(wb.perm) && Array(wa.seg_start) == Array(wb.seg_start) && Array(wa.seg_stop) == Array(wb.seg_stop)
            @test Array(wa.perm) != collect(1:cap)
        end
    end

    @testset "T-892 resort_by_cell! on CUDA gives sort_by_cell!'s bits at N = 2 and 3 through moves, frees and appended particles [A4, WP-041]" begin
        # T-850's run with device species: the moves, frees and appends of resort.jl's helpers on
        # host copies (moved to the host and back around them), the sorts on the device, compared
        # bit for bit every round; the resort must have sorted from the order, not in full.
        # `species_arrays` includes `count` (engine_cases.jl), so the host copies append behind the device's count.
        sync!(dst, src) = foreach(copyto!, species_arrays(dst), species_arrays(src))
        for (T, n) in ((Float32, (40, 36)), (Float32, (24, 20, 18)), (Float64, (24, 20, 18)))
            N = length(n)
            cap = 12 * prod(n)
            append = 23
            grid = PeriodicGrid(n, ntuple(_ -> 1 / 2, N), 1)
            ha = resort_species(T, n, cap, 8 * prod(n); seed = 3)
            hb = deepcopy(ha)
            a, b = todev(GPU, ha), todev(GPU, hb)
            wa, wb = Workspace(a, grid), Workspace(b, grid; append)
            sort_by_cell!(a, grid, wa); resort_by_cell!(b, grid, wb)
            same = Bool[]
            for k in 1:6
                for (h, d) in ((ha, a), (hb, b))
                    sync!(h, d)
                    rng = ResortLCG(3 + 100k)
                    resort_move!(h, n, rng)
                    iseven(k) && resort_append!(h, n, append, rng; tag0 = 1000k)
                    sync!(d, h)
                end
                # the resort's precondition: no alive particle past the last sort's count plus the tail
                tail_end = Int(Array(wb.resort.sorted_count)[1]) + append
                @test !any(Array(b.status)[(tail_end + 1):end] .== ParticleMeshCore.ALIVE)
                sort_by_cell!(a, grid, wa); resort_by_cell!(b, grid, wb)
                KernelAbstractions.synchronize(GPU)
                push!(same, all(same_bits(Array(x), Array(y)) for (x, y) in zip(species_arrays(a), species_arrays(b))) &&
                            Array(wa.perm) == Array(wb.perm) && Array(wa.seg_start) == Array(wb.seg_start) &&
                            Array(wa.seg_stop) == Array(wb.seg_stop))
            end
            @test all(same)
            @test wb.resort.resorts[] == 6
        end
    end

    @testset "T-194 Fast and Reference on CUDA agree per node and per face [A4, A7, WP-013]" begin
        rerun_on(GPU, (("T-135", body_T135), ("T-113", body_T113_faces), ("T-265 nodes", body_T265_nodes)))
    end

    @testset "T-1066 the pullbacks on CUDA against the CPU: flux at N = 1 and 2, deposit and gather at N = 1 to 3, the sort bitwise [A7, A8, WP-077]" begin
        # The same kernels on both backends: each entry within C·eps(T) of the CPU's, scaled by the
        # largest entry of its array (a device may contract to FMAs; the gather's adjoint field is
        # added by atomics in another order).
        C = tolC("T-1066")
        close(a, b, T) = record!("T-1066", T, maximum(abs.(Array(a) .- b)) / (eps(T) * max(maximum(abs, b), floatmin(T))), GPU) <= C
        for T in (Float32, Float64)
            for shape in SHAPES, st in (Node(), Center())
                grid = flux_grid_1d(T, shape, st)
                s = moved_species(T, grid, 300; seed = 9, dead = (40, 41, 250))
                J̄ = ntuple(k -> T.(lcg_field(field(backend, T, grid), 20 + k)), 3)
                x̄ = zero_like(s); y = zero_like(s)
                flux_pullback!(x̄, J̄..., s, shape, st, grid; dt = T(1 // 4), quantity_y = ChargeUy(), quantity_z = ChargeUz())
                on(GPU, flux_pullback!, y, J̄..., s, shape, st, grid; dt = T(1 // 4), quantity_y = ChargeUy(), quantity_z = ChargeUz())
                @test all(close(b, a, T) for ((_, a), (_, b)) in zip(float_arrays(x̄), float_arrays(y)))
            end
            for shape in SHAPES, st in STAGGER_PAIRS
                grid = grid2(T, flux_g2(shape, st))
                s = flux_species_2d(T, grid, 300; seed = 9)
                J̄ = ntuple(k -> T.(lcg_field(field(backend, T, grid), 30 + k)), 3)
                x̄ = zero_like(s); y = zero_like(s)
                flux_pullback!(x̄, J̄..., s, shape, st, grid; dt = T(1 // 4), quantity_z = QZ2)
                on(GPU, flux_pullback!, y, J̄..., s, shape, st, grid; dt = T(1 // 4), quantity_z = QZ2)
                @test all(close(b, a, T) for ((_, a), (_, b)) in zip(float_arrays(x̄), float_arrays(y)))
            end
            for (N, s, grid, staggers, quantities) in deposit_cases(T), shape in SHAPES, st in staggers
                ρ̄ = T.(lcg_field(field(backend, T, grid), 11))
                x̄ = zero_like(s); y = zero_like(s)
                deposit_pullback!(x̄, ρ̄, s, shape, st, grid; quantity = last(quantities))
                on(GPU, deposit_pullback!, y, ρ̄, s, shape, st, grid; quantity = last(quantities))
                @test all(close(b, a, T) for ((_, a), (_, b)) in zip(float_arrays(x̄), float_arrays(y)))
                a = T.(lcg_field(field(backend, T, grid), 13))
                ō = T.(lcg_values(capacity(s), 19))
                ā, x̄, ō1 = zero(a), zero_like(s), copy(ō)
                āg, y, ō2 = zero(a), zero_like(s), copy(ō)
                gather_pullback!(ā, x̄, ō1, a, s, shape, st, grid)
                on(GPU, gather_pullback!, āg, y, ō2, a, s, shape, st, grid)
                @test close(āg, ā, T) && ō1 == ō2
                @test all(close(b, a, T) for ((_, a), (_, b)) in zip(float_arrays(x̄), float_arrays(y)))
            end
            s = flux_species_2d(T, grid2(T, 2), 300; seed = 5)
            perm = Int32.(sortperm(lcg_values(capacity(s), 3)))
            ȳ = zero_like(s)
            foreach(((k, (_, a)),) -> (a .= T.(lcg_values(length(a), 40 + k))), enumerate(float_arrays(ȳ)))
            d = todev(GPU, ȳ)
            sort_pullback!(ȳ, perm)
            sort_pullback!(d, todev(GPU, perm))
            KernelAbstractions.synchronize(GPU)
            @test all(same_bits(Array(b), a) for ((_, a), (_, b)) in zip(float_arrays(ȳ), float_arrays(d)))
        end
    end
end
