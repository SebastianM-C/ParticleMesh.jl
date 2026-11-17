# #297: a fixed workgroupsize that a kernel cannot take, T-500 and T-501. The largest work-group
# a compiled kernel accepts is per kernel; PoCL lowers it for kernels with a large private stack.
# KernelAbstractions computes it inside the launch, after compiling the kernel for its arguments,
# and has no public query for it, so the engine cannot clamp to it: it refuses the launch with an
# ArgumentError that names the size and the kernel (`launcher`, src/launch.jl).

import KernelAbstractions.KernelInterface as KI

"""
    flux2d_or_refusal(T, shape, st, wg) -> (; J, err)

The unfolded 2D Reference flux (sort and deposition) of the 400-particle case of T-300 with the
fixed `workgroupsize = wg`, or the exception its launch threw (`J = nothing` then).
"""
function flux2d_or_refusal(::Type{T}, shape, st, wg) where {T}
    grid = grid2(T, flux_g2(shape, st))
    s = flux_species_2d(T, grid, 400)
    try
        Jx, Jy, Jz, _ = flux2d(Reference(), s, shape, st, grid; workgroupsize = wg)
        return (; J = (Jx, Jy, Jz), err = nothing)
    catch err
        return (; J = nothing, err)
    end
end

"Whether `err` is the engine's refusal of `workgroupsize = wg` for the kernel `kernel`."
engine_refusal(err, wg, kernel) =
    err isa ArgumentError && occursin("workgroupsize = $wg does not fit the launch of $kernel on", err.msg) &&
    occursin("workgroupsize = nothing", err.msg)

@testset "T-500 a fixed workgroupsize either gives the Reference bits of a small one or is refused by name, for the Dual 2D quadratic flux [A4, A8]" begin
    st = (Node(), Node())   # the stagger of the electromagnetic 2D step's current
    wmax = KI.max_work_group_size(backend)
    # #297: Dual{Float32,3}, Quadratic, 2D at 512 and 1024 (PoCL reported 983 work-items as the
    # flux patch kernel's maximum at this stagger in #297; the maximum moves with the kernel's code
    # and with the machine: since WP-061's change of it PoCL reported 3269 on one CPU and ran 4096
    # on CI's runners), and at the backend's maximum, which the sort kernels before the flux patch
    # take. A size past the backend's limit is T-501's.
    ref = flux2d_or_refusal(D3, Quadratic, st, 32)
    @test ref.err === nothing
    @test same_bits.(flux2d_or_refusal(D3, Quadratic, st, nothing).J, ref.J) == (true, true, true)
    outcomes = Dict{Any,Symbol}()
    for wg in (512, 1024, wmax)
        r = flux2d_or_refusal(D3, Quadratic, st, wg)
        if r.err === nothing
            outcomes[wg] = :ran
            @test same_bits.(r.J, ref.J) == (true, true, true)
        else
            outcomes[wg] = :refused
            @test engine_refusal(r.err, wg, :flux_patch_2d_kernel!)
        end
    end
    # The per-kernel refusal of #297 needs a flux patch whose own maximum is below the backend's.
    # PoCL derives the maximum from the kernel's private memory, so where Dual{Float32,3} runs at
    # the backend's limit, Dual{Float64,8}, with 4.5 times the bytes per value, is the kernel that
    # is refused there (PoCL reported 727 work-items for it where it reported 3269 for the former).
    heavy = Dual{Nothing,Float64,8}
    if outcomes[wmax] === :ran
        r = flux2d_or_refusal(heavy, Quadratic, st, wmax)
        outcomes[heavy] = r.err === nothing ? :ran : :refused
        @test r.err !== nothing && engine_refusal(r.err, wmax, :flux_patch_2d_kernel!)
    end
    println("T-500 Dual{Float32,3} Quadratic 2D flux on $(nameof(typeof(backend))): ",
            join(("workgroupsize $wg $(outcomes[wg])" for wg in (512, 1024, wmax)), ", "),
            haskey(outcomes, heavy) ? "; Dual{Float64,8} at workgroupsize $wmax $(outcomes[heavy])" : "")
    @test :refused in values(outcomes)
    # Float32 and Float64 at 1024: the bits of 32.
    for T in (Float32, Float64)
        a, b = flux2d_or_refusal(T, Quadratic, st, 32), flux2d_or_refusal(T, Quadratic, st, 1024)
        @test a.err === nothing && b.err === nothing
        @test same_bits.(a.J, b.J) == (true, true, true)
    end
end

@testset "T-501 the launcher passes nothing through, refuses an oversized launch by name before it writes, and rewraps only ArgumentErrors about the work-group [A4, A9]" begin
    k = ParticleMeshCore.fill_guards_kernel!
    @test ParticleMeshCore.launcher(k, backend, nothing) isa KernelAbstractions.Kernel
    @test ParticleMeshCore.launcher(k, backend, 7) isa ParticleMeshCore.SizedLaunch
    grid = PeriodicGrid(12, 0.5, 2)
    a = field(backend, Float64, grid)
    a .= 1:length(a)
    before = copy(a)
    wmax = KI.max_work_group_size(backend)
    err = try
        fill_guards!(a, grid; workgroupsize = 2wmax)
        nothing
    catch e
        e
    end
    @test engine_refusal(err, 2wmax, :fill_guards_kernel!)
    # KernelAbstractions' own message is kept. Which of its two checks refuses 2wmax depends on
    # the backend's per-dimension limit: POCL builds report either 4096 per dimension ("exceeds
    # the maximum of ... per dimension") or more ("has more than ... work-items, the maximum
    # for this kernel"), and the CI pre-release job resolves a newer pocl_standalone_jll.
    @test occursin(r"exceeds the maximum|the maximum for this kernel", err.msg)
    @test a == before                                # refused before the kernel ran
    b = copy(before)
    fill_guards!(a, grid; workgroupsize = 3)
    fill_guards!(b, grid)
    @test a == b && a != before
    # Only an ArgumentError about the work-group is rewrapped; any other exception, an
    # ArgumentError about something else included (#325), passes unchanged.
    stub(e) = ParticleMeshCore.SizedLaunch((args...; ndrange) -> throw(e), k, backend, 7)
    @test_throws DomainError(1.0) stub(DomainError(1.0))(a; ndrange = 4)
    unrelated = ArgumentError("Don't know how to convert arguments for Kernel")
    @test_throws unrelated stub(unrelated)(a; ndrange = 4)
    msg = "Work-group size (9,) has more than 8 work-items, the maximum for this kernel"
    r = try stub(ArgumentError(msg))(a; ndrange = 4) catch e e end
    @test engine_refusal(r, 7, :fill_guards_kernel!) && occursin(": $msg.", r.msg)
end

@testset "T-502 the tuned workgroupsize runs the Dual{Float64,2} 2D quadratic flux over 1024 cells with the bits of a small one [A4, A8]" begin
    # #375: KernelAbstractions 0.10.4 and 0.10.5 tuned the CPU work-group without regard to the
    # kernel's private memory, and the flux patch kernel's three NTuple{36,T} accumulators (2592 B
    # per work-item at Dual{Float64,2}) times a work-group of 1024 overflowed the thread's stack:
    # the process died with signal 11. 0.10.6 caps the tuned size; a regression kills this process.
    D2 = Dual{Nothing,Float64,2}
    st = (Node(), Node())
    grid = grid2(D2, flux_g2(Quadratic, st); n = (32, 32))
    tuned = flux2d(Reference(), flux_species_2d(D2, grid, 400), Quadratic, st, grid)
    small = flux2d(Reference(), flux_species_2d(D2, grid, 400), Quadratic, st, grid; workgroupsize = 32)
    @test same_bits.(tuned[1:3], small[1:3]) == (true, true, true)
    @test any(!iszero, tuned[1])
end

"A backend that is not the CPU's, to build (never launch) a kernel on."
struct OtherBackend <: KernelAbstractions.Backend end

@testset "T-801 the flux patch launches in work-groups of one on the CPU backend, a fixed workgroupsize is kept, another backend chooses [A7, WP-061]" begin
    # #442: with Dual elements the CPU backend's tuned work-group costs 4 to 11 times the
    # work-group of one for this kernel (reports/wp-061-flux-deposition-cost.md).
    k = ParticleMeshCore.flux_patch_2d_kernel!
    il = ParticleMeshCore.item_launcher
    wg(kernel) = KernelAbstractions.workgroupsize(kernel)
    @test wg(il(k, backend, nothing)) === KernelAbstractions.StaticSize{(1,)}
    sized = il(k, backend, 64)
    @test sized isa ParticleMeshCore.SizedLaunch && sized.workgroupsize == 64 && wg(sized.instance) === KernelAbstractions.StaticSize{(64,)}
    @test wg(il(k, OtherBackend(), nothing)) === KernelAbstractions.DynamicSize
    @test wg(ParticleMeshCore.launcher(k, backend, nothing)) === KernelAbstractions.DynamicSize
    # The Reference deposit launches its patch kernel through item_launcher and its reduction through launcher.
    lowered = only(code_lowered(ParticleMeshCore._deposit_flux_2d!, Tuple{Reference,Vararg{Any,11}}))
    refname(x) = x isa Core.SSAValue ? refname(lowered.code[x.id]) : x isa GlobalRef ? x.name : x isa Symbol ? x : nothing
    launches = [(refname(e.args[1]), refname(e.args[2])) for e in lowered.code if e isa Expr && e.head === :call && length(e.args) >= 2]
    @test (:item_launcher, :flux_patch_2d_kernel!) in launches
    @test (:launcher, :flux_reduce_2d_kernel!) in launches
    # The default launch of the Reference flux gives the bits of a fixed size, for the Dual of #442 too.
    D2 = Dual{Nothing,Float64,2}
    st = (Node(), Node())
    for T in (TYPES..., D2)
        grid = grid2(T, flux_g2(Quadratic, st))
        tuned = flux2d(Reference(), flux_species_2d(T, grid, 400), Quadratic, st, grid)
        fixed = flux2d(Reference(), flux_species_2d(T, grid, 400), Quadratic, st, grid; workgroupsize = 7)
        @test same_bits.(tuned[1:3], fixed[1:3]) == (true, true, true)
        @test any(!iszero, tuned[1])
    end
end

@testset "T-803 the flux patch is stored one column per cell on the CPU backend and cell-major on another, and the Reference flux has the same bits either way [A4, A7, WP-061]" begin
    SM, CM = ParticleMeshCore.StencilMajor(), ParticleMeshCore.CellMajor()
    @test ParticleMeshCore.flux_patch_layout(backend) === SM
    @test ParticleMeshCore.flux_patch_layout(OtherBackend()) === CM
    D2 = Dual{Nothing,Float64,2}
    for T in (TYPES..., D2), shape in SHAPES, st in ((Node(), Node()), (Center(), Node()))
        grid = grid2(T, flux_g2(shape, st))
        default = flux2d(Reference(), flux_species_2d(T, grid, 400), shape, st, grid)
        @test default[4].flux_layout === SM
        a = flux2d(Reference(), flux_species_2d(T, grid, 400), shape, st, grid; flux_layout = SM)
        b = flux2d(Reference(), flux_species_2d(T, grid, 400), shape, st, grid; flux_layout = CM)
        @test same_bits.(default[1:3], a[1:3]) == (true, true, true)
        @test same_bits.(a[1:3], b[1:3]) == (true, true, true)
        @test any(!iszero, a[1])
    end
    # A workspace of one layout whose patch array has the other's size is refused, either way.
    T = Float64
    st = (Node(), Node())
    grid = grid2(T, flux_g2(Quadratic, st))
    s = flux_species_2d(T, grid, 400)
    good = Workspace(s, grid; flux = (Quadratic, st), flux_layout = SM)
    other = Workspace(s, grid; flux = (Quadratic, st), flux_layout = CM)
    # The constructor's own default, which the module's states get, is the backend's layout.
    plain = Workspace(s, grid; flux = (Quadratic, st))
    @test plain.flux_layout === SM && size(plain.flux_patch) == size(good.flux_patch) != size(other.flux_patch)
    sort_by_cell!(s, grid, good)
    Jx, Jy, Jz = fields3(T, grid)
    for (layout, patch) in ((SM, other.flux_patch), (CM, good.flux_patch))
        bad = Workspace(good.key, good.perm, good.ak, good.tmp_i32, good.tmp_T, good.tmp_u8, good.tmp_extras, good.seg_start, good.seg_stop,
                        patch, good.flux, layout, good.resort, good.bins)
        @test_throws "values per cell in the layout" deposit_flux!(Reference(), Jx, Jy, Jz, s, Quadratic, st, grid, bad; dt = T(1 // 4), quantity_z = QZ2)
    end
end
