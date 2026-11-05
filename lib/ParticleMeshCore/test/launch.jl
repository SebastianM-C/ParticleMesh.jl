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
    # flux patch kernel's maximum at this stagger, both on the machine of #297 and where this test
    # was written, and 1164 at (Node(), Center())), and at the backend's maximum, which the sort
    # kernels before the flux patch take. A size past the backend's limit is T-501's.
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
    println("T-500 Dual{Float32,3} Quadratic 2D flux on $(nameof(typeof(backend))): ",
            join(("workgroupsize $wg $(outcomes[wg])" for wg in (512, 1024, wmax)), ", "))
    # At the backend's limit the Dual flux patch is over its own, smaller, maximum: the per-kernel
    # refusal of #297, which the 1024 case meets where the kernel's maximum is below 1024.
    @test outcomes[wmax] === :refused
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
    @test occursin("exceeds the maximum", err.msg)   # KernelAbstractions' own message is kept
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
