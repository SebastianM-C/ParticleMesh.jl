# Allocation of the N = 1 engine operations: T-144 (A7). The module's whole step is measured by
# its own T-220; here every engine entry point a step composes is measured on its own, as T-268
# does at N = 2.

@testset "T-144 allocation of the N = 1 engine operations does not grow with particle count or grid size [A7, WP-010]" begin
    # Bytes per call after warm-up at two particle counts and two grid sizes, with and without a
    # tuple of module extras (WP-011), which the sort permutes. The CPU backend allocates a
    # constant amount per kernel launch (task bookkeeping); what must not happen is growth with
    # the problem size.
    for T in (Float32, Float64), extras in (NamedTuple(), (old_cell = Int32, old_offset = T))
        bytes = Dict{Tuple{String,Int,Int},Int}()
        for (n, ppc) in ((32, 16), (32, 256), (128, 16))
            grid = PeriodicGrid(n, T(1 // 4), guard_width(Quadratic))
            s = Species(backend, T, n * ppc; q = -1, m = 1, extras)
            p = (one(T), T(1 // 20), 2 * T(π) / (n * T(1 // 4)))
            vth = (T(1 // 20), T(1 // 20), T(1 // 20))
            load_quiet!(s, grid, N1Density(), p, ppc; vth, seed = 1, stream = 1)
            ws = Workspace(s, grid)
            ρ = field(backend, T, grid)
            E = field(backend, T, grid)
            out = similar(s.w)
            ops = (sort = () -> sort_by_cell!(s, grid, ws),
                   reference = () -> deposit!(Reference(), ρ, s, Quadratic, Center(), grid, ws),
                   fast = () -> deposit!(Fast(), ρ, s, Quadratic, Center(), grid, ws),
                   gather = () -> gather!(out, ρ, s, Quadratic, Center(), grid),
                   fill = () -> fill_guards!(ρ, grid), fold = () -> fold_guards!(ρ, grid),
                   cumsum = () -> cumsum_periodic!(E, ρ, grid),
                   load = () -> load_quiet!(s, grid, N1Density(), p, ppc; vth, seed = 1, stream = 1))
            for (name, op) in pairs(ops)
                op(); op()
                bytes[(string(name), n, ppc)] = @allocated op()
            end
        end
        @info "T-144 bytes per call" T extras = keys(extras) bytes nthreads = Threads.nthreads()
        for name in ("sort", "reference", "fast", "gather", "fill", "fold", "cumsum", "load")
            small = bytes[(name, 32, 16)]
            # With more than one thread AcceleratedKernels' threaded CPU sample sort allocates
            # about 8 bytes per particle despite the workspace (#82), so the sort's checks are
            # broken there (questions/wp-011-t144-threads.md); an unexpected pass after an
            # upstream fix fails the test, which is the signal to drop this.
            sort_allocates = Threads.nthreads() > 1 && name == "sort"
            # Julia 1.14 allocates per particle in some of these cases, and which ones differs
            # between machines (#58), so the particle-count check is skipped there (recorded as
            # broken); the @info above still logs the bytes. Remove the skip when #58 is closed.
            if VERSION >= v"1.14-"
                @test_skip bytes[(name, 32, 256)] <= small
            else
                @test bytes[(name, 32, 256)] <= small broken = sort_allocates
            end
            @test bytes[(name, 128, 16)] <= small broken = sort_allocates
        end
    end
end
