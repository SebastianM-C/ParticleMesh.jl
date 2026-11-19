# WP-069: the module's transposed pairs, by dot-product tests on random data (T-985 to T-987): the
# Yee step with its current against the solver's own kernels with negated time arguments, the
# particles' read of E and B (`fields_at`, plain and compensated) against the engine's deposit on
# the field layout, and the self-adjointness of the two periodic stencils on that path (the
# compensated gather's sharpening and the current filter). These are the pieces an adjoint of the
# step would run backwards; the engine's own pair is T-980. Grids, random fields and `rand_T` are
# yee_nd_cases.jl's.

"⟨a, b⟩ over the interior of the components `keys` of two field tuples."
dot_interior(a, b, keys, grid) = sum(k -> sum(interior_view(getfield(a, k), grid) .* interior_view(getfield(b, k), grid)), keys)
"√(Σ max|component|²) over the interior of the components `keys` (the value and every partial for duals)."
norm_interior(a, keys, grid) = sqrt(sum(k -> sum(x -> absmax(x)^2, interior_view(getfield(a, k), grid)), keys))

"""
    yee_transposed!(f, grid, dt, steps, norms) -> J̄

The transpose of `steps` periodic solves (`solve_fields!(Yee(), …)` with one current J held over
all of them), applied to the fields `f` (their J zero): per step `yee_e!(−Δt/2)`, then
`J̄ −= Δt·Ē`, then `yee_b_half!(−Δt)` and `yee_e!(−Δt/2)`. Returns J̄ as the fields' J arrays of a
fresh tuple (interior only), and pushes ‖(Ē, B̄)‖ before each transposed step onto `norms`. The
forward half step `B −= h∇×E` transposes to `E −= h∇×B` and the full step `E += Δt(∇×B − J)` to
`B += Δt∇×E` and `J̄ −= Δt·Ē`, since on a periodic grid `(D⁺)ᵀ = −D⁻` makes the two discrete curls
each other's transpose (yee.jl).
"""
function yee_transposed!(f, grid, dt, steps, norms)
    T = eltype(f.Ex)
    Jbar = em_fields(CPU(), T, grid)
    for _ in 1:steps
        push!(norms, norm_interior(f, EB_KEYS, grid))
        yee_e!(f, grid, -dt / 2)
        for (j, e) in zip(J_KEYS, E_KEYS)
            interior_view(getfield(Jbar, j), grid) .-= dt .* interior_view(getfield(f, e), grid)
        end
        yee_b_half!(f, grid, -dt)
        yee_e!(f, grid, -dt / 2)
    end
    return Jbar
end

"‖(E, B, Δt·J)‖ over the interior: the size of a forward step's input."
norm_step_input(f, dt, grid) = sqrt(norm_interior(f, EB_KEYS, grid)^2 + (absmax(dt) * norm_interior(f, J_KEYS, grid))^2)

"""
    yee_pair(T, N, steps; seed, transposed = yee_transposed!) -> (lhs, rhs, scale)

`lhs = ⟨M x, y⟩` for `steps` solves `M` of random fields and current `x` at 0.9 of the Courant
limit, `rhs = ⟨x, Mᵀ y⟩` with `Mᵀ` from `transposed(f, grid, dt, steps, norms)` (which returns J̄)
for random fields `y` without current. The scale is step-aware, `Σₖ ‖(E, B, Δt·J)‖ before forward
step k · ‖(Ē, B̄)‖ before the transposed step of k`, since the rounding of each step reaches the
products through the adjoint state of that step: one bound for any number of steps, Δt and size
of the current (the physics breaker's runs to 1000 steps, near the Courant limit and with
Δt·J dominant stayed within a factor 12 of the 25-step cases of this test).
"""
function yee_pair(::Type{T}, N, steps; seed, transposed = yee_transposed!) where {T}
    grid = nd_grid(Val(N), basetype(T))
    dt = basetype(T)(9 // 10) * courant_limit(Yee(), grid)
    x = random_fields(T, grid; seed)
    y = random_fields(T, grid; seed = seed + 1, current = false)
    Mx = map(copy, x)
    forward = Float64[]
    for _ in 1:steps
        push!(forward, norm_step_input(Mx, dt, grid))
        solve_fields!(Yee(), Mx, grid, dt)
    end
    Mty = map(copy, y)
    adjoint = Float64[]
    Jbar = transposed(Mty, grid, dt, steps, adjoint)
    lhs = dot_interior(Mx, y, EB_KEYS, grid)
    rhs = dot_interior(x, Mty, EB_KEYS, grid) + dot_interior(x, Jbar, J_KEYS, grid)
    scale = sum(k -> forward[k] * adjoint[steps + 1 - k], 1:steps)
    return lhs, rhs, scale
end

"The naive guess for the transpose, the forward solve itself (J zero): the control of T-985."
function forward_again!(f, grid, dt, steps, norms)
    for _ in 1:steps
        push!(norms, norm_interior(f, EB_KEYS, grid))
        solve_fields!(Yee(), f, grid, dt)
    end
    return em_fields(CPU(), eltype(f.Ex), grid)
end

@testset "T-985 the Yee step with its current transposes to its own kernels with negated time arguments, N = 1, 2, 3 [A2, A7, A8, WP-069]" begin
    C = emC("T-985")
    for N in (1, 2, 3), T in TYPES, steps in (1, 25)
        ε = base_eps(T)
        lhs, rhs, scale = yee_pair(T, N, steps; seed = 9850 + N)
        r = record!("T-985", T, absdiff(lhs, rhs) / (ε * scale), CPU())
        @test r <= C
        # control: the step itself is not its transpose
        lhs, rhs, scale = yee_pair(T, N, steps; seed = 9850 + N, transposed = forward_again!)
        @test absdiff(lhs, rhs) > 100 * C * ε * scale
    end
end

"""
    random_particles(T, grid, np; next) -> Species

`np` alive particles of charge 1 at random cells and offsets in [0, 1) (offsets with random
partials for duals), the weights left for the caller; `next()` draws in [−1/2, 1/2) (`rand_T`).
"""
function random_particles(::Type{T}, grid::PeriodicGrid{N}, np; next) where {T,N}
    s = ParticleMeshCore.Species(backend, T, np; q = 1, m = 1, N)
    n = ncells(grid)
    for i in 1:np, d in 1:N
        s.cell[d][i] = Int32(mod1(round(Int, (value_of(next()) + 1) * 1000 * n[d]), n[d]))
        s.offset[d][i] = next() + one(T) / 2
        s.status[i] = ParticleMeshCore.ALIVE
    end
    fill!(s.count, Int32(np))
    return s
end

"The copy the particles read with `gather` (`nothing`: the fields themselves): `sharpen_x!` of every E and B component."
read_fields(::Nothing, f, shape, grid) = f
function read_fields(c::CompensatedGather, f, shape, grid)
    β = sharpening_beta(c, shape, grid.dx[1])
    return particle_fields(map(k -> sharpen_x!(similar(getfield(f, k)), getfield(f, k), β, grid), NamedTuple{EB_KEYS}(EB_KEYS)), f)
end
"The transpose of `read_fields` on a folded deposit `ρ` (interior values; the stencils are self-adjoint, T-987)."
read_transposed(::Nothing, ρ, shape, grid) = ρ
read_transposed(c::CompensatedGather, ρ, shape, grid) =
    sharpen_x!(similar(ρ), fill_guards!(ρ, grid), sharpening_beta(c, shape, grid.dx[1]), grid)

"""
    gather_pair(T, N, shape, gather; seed, layout = field_layout(Val(N))) -> (lhs, rhs, scale)

`lhs = Σᵢ Σ_c pᵢ_c·F_c(xᵢ)` with `F_c` the six components that `fields_at` reads for particle `i`
(from the copy of `gather`) and `p` random per particle and component; `rhs = Σ_c ⟨x_c, Gᵀ
fold(deposit!(p_c at layout[c]))⟩` with `Gᵀ` the transpose of the copy; `scale = Σᵢ,c|pᵢ_c|·max|x|`.
"""
function gather_pair(::Type{T}, N, shape, gather; seed, layout = field_layout(Val(N)), np = 300) where {T}
    grid = nd_grid(Val(N), basetype(T))
    g = Int32(guards(grid))
    rng = Xoshiro(seed)
    next = () -> rand_T(rng, T)
    s = random_particles(T, grid, np; next)
    x = random_fields(T, grid; seed, current = false)
    probes = NamedTuple{EB_KEYS}(map(_ -> [next() for _ in 1:np], EB_KEYS))
    fr = read_fields(gather, x, shape, grid)
    lhs = sum(1:np) do i
        E, B = ParticleMeshElectromagnetic.fields_at(fr, ntuple(d -> s.cell[d][i], N), ntuple(d -> s.offset[d][i], N), shape, g)
        sum(c -> probes[E_KEYS[c]][i] * E[c] + probes[B_KEYS[c]][i] * B[c], 1:3)
    end
    rhs = sum(EB_KEYS) do k
        sk = deepcopy(s)
        sk.w .= probes[k]
        ρ = deposit!(Fast(), field(backend, T, grid), sk, shape, getfield(layout, k), grid, Workspace(sk, grid))
        d = read_transposed(gather, fold_guards!(ρ, grid), shape, grid)
        sum(interior_view(d, grid) .* interior_view(getfield(x, k), grid))
    end
    scale = sum(k -> sum(absmax, probes[k]), EB_KEYS) * maximum(k -> maximum(absmax, interior_view(getfield(x, k), grid)), EB_KEYS)
    return lhs, rhs, scale
end

"The layout with the staggers of E and B exchanged: the wrong partner of the control of T-986."
swapped_layout(L) = merge(L, (Ex = L.Bx, Ey = L.By, Ez = L.Bz, Bx = L.Ex, By = L.Ey, Bz = L.Ez))

@testset "T-986 the particles' read of E and B, plain and compensated, is the transpose of deposit! on the field layout, N = 1, 2, 3 [A2, A7, A8, WP-069, WP-067]" begin
    C = emC("T-986")
    for N in (1, 2, 3), T in TYPES, shape in SHAPES, gather in (nothing, CompensatedGather())
        ε = base_eps(T)
        lhs, rhs, scale = gather_pair(T, N, shape, gather; seed = 9860 + N)
        r = record!("T-986", T, absdiff(lhs, rhs) / (ε * scale), CPU())
        @test r <= C
        # control: the deposit with the staggers of E and B exchanged is not the read's transpose
        lhs, rhs, scale = gather_pair(T, N, shape, gather; seed = 9860 + N, layout = swapped_layout(field_layout(Val(N))))
        @test absdiff(lhs, rhs) > 100 * C * ε * scale
    end
end

@testset "T-987 the sharpening of the compensated gather and the binomial current filter are self-adjoint on the periodic interior and fill their guards with its images, N = 1, 2, 3 [A2, A7, A8, WP-069, WP-029, WP-067]" begin
    C = emC("T-987")
    for N in (1, 2, 3), T in TYPES
        ε = base_eps(T)
        grid = nd_grid(Val(N), basetype(T))
        f = random_fields(T, grid; seed = 9870 + N, current = false)
        x, y = f.Ex, f.By                     # two random fields of the grid (guards filled)
        β = sharpening_beta(CompensatedGather(), Quadratic, grid.dx[1])
        sharpen(a) = sharpen_x!(similar(a), a, β, grid)
        filtered(a, passes) = binomial!(copy(a), similar(a), grid, passes)
        # each with its largest gain: 1 + 4β for the sharpening (the shortest wave), 1 for the filter
        ops = ((sharpen, 1 + 4β), (a -> filtered(a, 1), 1), (a -> filtered(a, 2), 1))
        for (op, gain) in ops
            lhs = sum(interior_view(op(x), grid) .* interior_view(y, grid))
            rhs = sum(interior_view(x, grid) .* interior_view(op(y), grid))
            scale = gain * sqrt(sum(v -> absmax(v)^2, interior_view(x, grid))) * sqrt(sum(v -> absmax(v)^2, interior_view(y, grid)))
            r = record!("T-987", T, absdiff(lhs, rhs) / (ε * scale), CPU())
            @test r <= C
            # the result's guards are the periodic images of its interior, bit for bit (the
            # interior alone is what the dot product sees)
            out = op(x)
            @test same_bits(out, fill_guards!(copy(out), grid))
        end
    end
end
