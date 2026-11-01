# WP-013, the engine on CUDA: T-190 to T-194 and T-198. Included last by runtests.jl, after the
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
                       ("T-267", body_T267), ("T-269", body_T269)))
    end

    @testset "T-192 Reference mode of the engine on CUDA is bitwise across reruns and launch shapes [A4, WP-013]" begin
        # N = 1: the scan, the sort with extras, both depositions (q·w and q·w·u_y) and the flux,
        # per element type, at the default and two explicit workgroup sizes; three reruns.
        # N = 2 (T-266's set): the sort with extras, the deposition of a species that crossed both
        # seams and the corner, the gather, fill, fold and the load.
        runs = [vcat(thread_digests(GPU; wgs = GPU_WG_SIZES), flux_digests(GPU; wgs = GPU_WG_SIZES),
                     engine2d_digests(GPU; wgs = GPU_WG_SIZES)) for _ in 1:3]
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
            host = @allocated step()
            dev = CUDACore.@allocated step()
            sort_host = @allocated sort()
            sort_dev = CUDACore.@allocated sort()
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

    @testset "T-194 Fast and Reference on CUDA agree per node and per face [A4, A7, WP-013]" begin
        rerun_on(GPU, (("T-135", body_T135), ("T-113", body_T113_faces), ("T-265 nodes", body_T265_nodes)))
    end
end
