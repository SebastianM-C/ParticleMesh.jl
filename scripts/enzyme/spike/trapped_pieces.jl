# WP-078: which piece of the trapped-beam case's step Enzyme's reverse mode handles. One piece per
# process (an Enzyme failure can be a segfault): the state after a primal run to `tmid`, then
# the reverse pass of that piece of `step!` (in its order) with Duplicated(state), Active(params).
#
#     PM_BACKEND=cuda PM_ENZYME_PATCH=scripts/enzyme/active_reduce.jl JULIA_LOAD_PATH=@:<repo>:@stdlib \
#       julia +1.13 --project=lib/ParticleMeshElectromagnetic/test scripts/enzyme/spike/trapped_pieces.jl <piece> [tmid] [ppl] [ny]
#
# pieces: sharpen, ionize, push, retire_x, current, retire_open, solve, window, summand, step
include(joinpath(@__DIR__, "..", "common.jl"))
include(joinpath(REPO, "scripts", "optim", "trapped2d_case.jl"))
const PEM = ParticleMeshElectromagnetic
using ParticleMeshElectromagnetic: CompensatedGather, ionization_electrons
using ParticleMeshCore: cellsize, guards
using Printf

piece = ARGS[1]
tmid = parse(Float64, get(ARGS, 2, "180"))
ppl = parse(Int, get(ARGS, 3, "32"))
ny = parse(Int, get(ARGS, 4, "2"))
F = Float32
θ = [TCASE.x0[i] for i in TCASE.free]
st, p, prob = start_of(trapped_problem(trapped_point(F.(θ))...; energy = F(budget(1.9)), ppl, ny, float = F);
                       ppl, ny, backend = backend(), gather = CompensatedGather(; k = 1))
kmid = round(Int, tmid / Float64(st.dt))
run!(st, p, kmid; observer = ())
sync()
println("piece $piece at step $(kmid + 1) (t = $(kmid * Float64(st.dt))), ppl $ppl ny $ny; born alive ", alive_count(ionization_electrons(st)))
flush(stdout)


# variants of the ionization kernel (ionization.jl), for the check: V = 1 newborns at rest (no
# push_momentum), V = 2 no `emitted` update
@eval PEM @kernel function ionize_variant_kernel!(a, e, flags, @Const(count), f, levels::NTuple{L,Any}, pusher, shape, g::Int32, dt, qm_h, birth::Bool,
                                cap::Int32, ::Val{V}) where {L,V}
    i = Int32(@index(Global, Linear))
    deferred = Int32(0)
    @inbounds if a.status[i] == ALIVE
        T = eltype(a.w)
        c = particle_cell(a, i, Val(2))
        δ = particle_offset(a, i, Val(2))
        E, B = fields_at(f, c, δ, shape, g)
        E2 = E[1] * E[1] + E[2] * E[2] + E[3] * E[3]
        b = bound_populations(a.extras, Val(L))
        old = ntuple(j -> (@inbounds b[j][i]), Val(L))
        wdt = map(l -> ionization_rate_sq(l, E2) * dt, levels)
        out = ntuple(j -> old[j] * -expm1(-wdt[j]), Val(L))
        ntuple(j -> (@inbounds b[j][i] = old[j] * exp(-wdt[j]) + (j == 1 ? zero(T) : out[max(j - 1, 1)]); nothing), Val(L))
        pending = a.extras.pending[i] + sum(out)
        w = a.w[i] * pending
        if birth && w > zero(T)
            slot = count[1] + i
            if slot <= cap
                u = V == 1 ? (zero(T), zero(T), zero(T)) : push_momentum(pusher, (zero(T), zero(T), zero(T)), E, B, qm_h)
                e.cell[1][slot] = c[1]
                e.cell[2][slot] = c[2]
                e.offset[1][slot] = δ[1]
                e.offset[2][slot] = δ[2]
                e.u[1][slot] = u[1]
                e.u[2][slot] = u[2]
                e.u[3][slot] = u[3]
                e.w[slot] = w
                e.status[slot] = ALIVE
                V == 2 || (a.extras.emitted[i] += pending)
                pending = zero(T)
            else
                deferred = Int32(1)
            end
        end
        a.extras.pending[i] = pending
    end
    @inbounds flags[i] = deferred
end

function piece!(st, p, k, ::Val{P}) where {P}
    (; numerics, grid, fields, dt) = st
    wgs = numerics.workgroupsize
    n = Int32.(grid.n)
    g = Int32(guards(grid))
    dt_dx = map(h -> dt / h, grid.dx)
    window = PEM.window_frame(st)
    if P === :sharpen
        PEM.sharpen_gather!(st)
    elseif P === :ionize
        PEM.ionize!(PEM.ionization_frame(st), st, p, k)
    elseif P === :push
        pf = PEM.particle_fields(st)
        for s in (PEM.deposited_species(st)..., PEM.witness_species(st)...)
            PEM.launcher(PEM.push_kernel!, get_backend(s), wgs)(s, pf, numerics.pusher, PEM.read_shape(numerics), g, s.q / s.m * dt / 2, dt_dx, n;
                                                                ndrange = PEM.capacity(s))
        end
    elseif P === :retire_x
        PEM.retire_x!(window, st)
    elseif P === :current
        PEM.current!(numerics.current, st)
    elseif P === :retire_open
        PEM.retire_open!(PEM.open_absorbers(st), st)
    elseif P === :solve
        PEM.solve_fields!(numerics.solver, fields, grid, dt, st.modifiers, p, k; workgroupsize = wgs)
    elseif P === :window
        PEM.advance_window!(window, st, p, k)
    elseif P === :step
        step!(st, p, k; observer = ())
    elseif P in (:resort_e, :resort_b, :dep_e, :dep_b)
        i = P in (:resort_e, :dep_e) ? 1 : 2
        s, ws = PEM.deposited_species(st)[i], PEM.deposited_workspaces(st)[i]
        if P in (:resort_e, :resort_b)
            PEM.resort_by_cell!(s, grid, ws; workgroupsize = wgs)
        else
            PEM.deposit_flux!(PEM.deposit_mode(st, s), fields.Jx, fields.Jy, fields.Jz, s, numerics.shape, PEM.rho_stagger(Val(2)), grid, ws;
                              dt = st.dt, PEM.transverse_currents(Val(2))..., workgroupsize = wgs)
        end
    elseif P === :edges
        for J in (fields.Jx, fields.Jy, fields.Jz)
            PEM.zero_x_edges!(window, J; workgroupsize = wgs)
            PEM.fold_guards!(J, grid; workgroupsize = wgs)
        end
    elseif P === :ionize_k || P === :ionize_nobirth
        fr = PEM.ionization_frame(st)
        a, e = fr.atoms, fr.electrons
        PEM.launcher(PEM.ionize_kernel!, get_backend(a), wgs)(a, e, fr.flags, e.count, PEM.particle_fields(st), p.ionization.levels, numerics.pusher,
                                                              PEM.read_shape(numerics), Int32(guards(grid)), dt, -e.q / e.m * dt / 4, P === :ionize_k,
                                                              Int32(PEM.capacity(e)); ndrange = PEM.capacity(a))
    elseif P === :ionize_v1 || P === :ionize_v2
        fr = PEM.ionization_frame(st)
        a, e = fr.atoms, fr.electrons
        PEM.launcher(PEM.ionize_variant_kernel!, get_backend(a), wgs)(a, e, fr.flags, e.count, PEM.particle_fields(st), p.ionization.levels, numerics.pusher,
                                                              PEM.read_shape(numerics), Int32(guards(grid)), dt, -e.q / e.m * dt / 4, true,
                                                              Int32(PEM.capacity(e)), Val(P === :ionize_v1 ? 1 : 2); ndrange = PEM.capacity(a))
    elseif P === :w_fields
        for a in (fields.Ex, fields.Ey, fields.Ez, fields.Bx, fields.By, fields.Bz)
            PEM.launcher(PEM.shift_columns_kernel!, get_backend(a), wgs)(a, st.ρ, Int32(g + 1), Int32(g + n[1]), a === fields.Ex; ndrange = size(a))
        end
    elseif P === :w_absorbers
        PEM.shift_absorbers!(st.modifiers, wgs)
    elseif P === :w_species
        w = st.aux.window
        s, ws, setup, ps = st.species[1], st.ws[1], st.setup[1], p.species[1]
        PEM.launcher(PEM.shift_particles_kernel!, get_backend(s), wgs)(s; ndrange = PEM.capacity(s))
        PEM.sort_after_shift!(st, s, ws)
    elseif P === :w_load
        w = st.aux.window
        s, setup, ps = st.species[1], st.setup[1], p.species[1]
        PEM.load_window_columns!(s, grid, PEM.window_load(setup, ps, p), w.lost[1], PEM.injection_column(w), 1, 100;
                                 fast = w.fast[1], flags = w.flags[1], workgroupsize = wgs)
    elseif P === :w_ionization
        PEM.shift_ionization!(PEM.ionization_frame(st), st, p, 100)
    elseif P === :filter
        PEM.filter_current!(numerics.filter, fields, grid, st.ρ; workgroupsize = wgs, edges = window)
    end
    return nothing
end
@kernel function sq_kernel!(out, @Const(a))
    i = @index(Global, Linear)
    @inbounds out[i] = a[i] * a[i]
end
# the function differentiated: the piece, then Σ Ex² + Σ w² of the plasma (a summand per element, seeded with ones)
function f!(o1, o2, st, p, k, v)
    piece!(st, p, k, v)
    sq_kernel!(get_backend(o1))(o1, st.fields.Ey; ndrange = length(o1))
    sq_kernel!(get_backend(o2))(o2, st.species[1].u[1]; ndrange = length(o2))
    return nothing
end
if get(ARGS, 5, "") != "check"
    o1 = similar(st.fields.Ey); fill!(o1, 0); d1 = similar(o1); fill!(d1, 1)
    o2 = similar(st.species[1].u[1]); fill!(o2, 0); d2 = similar(o2); fill!(d2, 1)
    dst = Enzyme.make_zero(st)
    t = @elapsed res = Enzyme.autodiff(set_runtime_activity(Reverse), f!, Const, Duplicated(o1, d1), Duplicated(o2, d2), Duplicated(st, dst), Active(p),
                                       Const(kmid + 1), Const(Val(Symbol(piece))))
    sync()
    dp = res[1][4]
    println("piece $piece: reverse OK in $(round(t; digits = 1)) s; |dst| over arrays ",
            sum(a -> Float64(sum(abs, a)), values(leaves(dst))), "; dp lasers ", dp.lasers)
    println("kernel arguments with dropped adjoints: ", sort!(collect(ACTIVE_SEEN)))
else
    # The dot-product test of the piece (its Jacobian J at the state): random r on every
    # floating-point array of the state after the piece (the shadow on entry), random v on every
    # array before it (a Dual state's tangent); ⟨Jᵀr, v⟩ from the reverse pass against ⟨r, Jv⟩
    # from the Dual run, both minus ⟨r, v⟩ (the arrays the piece leaves alone), per array.
    using Random: Xoshiro
    env_scratch = get(ENV, "PM_CHECK_SCRATCH", "1") == "1"
    only_v = Regex(get(ENV, "PM_CHECK_V", ""))      # v only on the arrays whose path matches
    struct PTag end
    std, pd, _ = start_of(trapped_problem(trapped_point([ForwardDiff.Dual{PTag}(F(x), zero(F)) for x in θ])...; energy = F(budget(1.9)), ppl, ny, float = F);
                          ppl, ny, backend = backend(), gather = CompensatedGather(; k = 1))
    run!(std, pd, kmid; observer = ())
    allarr!(out, path, x::AbstractArray) = (any(y -> y === x, values(out)) || (out[path] = x); out)
    allarr!(out, path, x::Union{Tuple,NamedTuple}) = (foreach(k -> allarr!(out, "$path.$k", x[k]), keys(x)); out)
    function allarr!(out, path, x)
        (isstructtype(typeof(x)) && !(x isa Module) && !(x isa Type)) || return out
        foreach(f -> isdefined(x, f) && allarr!(out, "$path.$f", getfield(x, f)), fieldnames(typeof(x)))
        return out
    end
    # the workspaces' arrays and the step's scratch fields: written before they are read in a step,
    # so neither a tangent on them nor an adjoint of them is part of the step's derivative
    scratch(k) = env_scratch && (occursin(".ws.", k) || occursin("_ws.", k) || occursin("flux_patch", k) || occursin(".scratch", k) || k == "st.ρ")
    A, Ad = allarr!(Dict{String,Any}(), "st", st), allarr!(Dict{String,Any}(), "st", std)
    rng = Xoshiro(11)
    L = leaves(st)
    V, R = Dict{String,Any}(), Dict{String,Any}()
    for k in intersect(keys(A), keys(Ad))
        size(A[k]) == size(Ad[k]) || continue
        if eltype(Ad[k]) <: ForwardDiff.Dual && haskey(L, k)
            v = similar(A[k]); copyto!(v, randn(rng, F, size(v))); V[k] = v
            (scratch(k) || !occursin(only_v, k)) && fill!(v, 0)
            Ad[k] .= ForwardDiff.Dual{PTag}.(A[k], v)
        elseif eltype(Ad[k]) == eltype(A[k])
            copyto!(Ad[k], A[k])
        end
    end
    st0 = deepcopy(st)
    dst = Enzyme.make_zero(st)
    Ls = leaves(dst)
    for k in keys(V)
        r = similar(V[k]); copyto!(r, randn(rng, F, size(r))); R[k] = r
        scratch(k) && fill!(r, 0)
        copyto!(Ls[k], r)
    end
    t = @elapsed Enzyme.autodiff(set_runtime_activity(Reverse), piece!, Const, Duplicated(st, dst), Active(p), Const(kmid + 1), Const(Val(Symbol(piece))))
    piece!(std, pd, kmid + 1, Val(Symbol(piece)))
    sync()
    Ld, Ls = leaves(std), leaves(dst)
    rows = []
    for k in sort!(collect(keys(V)))
        rev = Float64(sum((Ls[k] .- R[k]) .* V[k]))
        fwd = Float64(sum(R[k] .* (ForwardDiff.partials.(Ld[k], 1) .- V[k])))
        scale = Float64(sum(abs.(R[k] .* V[k])))
        (rev != 0 || fwd != 0) && push!(rows, (k, rev, fwd, abs(rev - fwd) / max(abs(fwd), abs(rev), 1e-30), scale))
    end
    rev, fwd = sum(r[2] for r in rows; init = 0.0), sum(r[3] for r in rows; init = 0.0)
    # the born electrons' arrays split at the count before the piece: alive slots and the rest
    if PEM.ionization_frame(st) !== nothing
        c0 = Int(Array(PEM.ionization_frame(st0).electrons.count)[1])
        for k in sort!(collect(keys(V)))
            startswith(k, "st.aux.ionization.electrons.data") || continue
            for (name, rg) in (("≤count", 1:c0), (">count", (c0 + 1):length(V[k])))
                isempty(rg) && continue
                rv = Float64(sum((Array(Ls[k])[rg] .- Array(R[k])[rg]) .* Array(V[k])[rg]))
                fw = Float64(sum(Array(R[k])[rg] .* (ForwardDiff.partials.(Array(Ld[k])[rg], 1) .- Array(V[k])[rg])))
                @printf("    region %-6s %-45s rev %+.6e fwd %+.6e\n", name, k, rv, fw)
            end
        end
        @printf("    count before %d, after (reverse state) %d, after (dual) %d\n", c0, Int(Array(PEM.ionization_frame(st).electrons.count)[1]), Int(Array(PEM.ionization_frame(std).electrons.count)[1]))
    end
    # central differences of the piece on Float32 states, x ± εv (PM_CHECK_FD = ε)
    if haskey(ENV, "PM_CHECK_FD")
        ε = parse(Float32, ENV["PM_CHECK_FD"])
        sp, sm = deepcopy(st0), deepcopy(st0)
        Lp, Lm = leaves(sp), leaves(sm)
        for k in keys(V)
            Lp[k] .+= ε .* V[k]; Lm[k] .-= ε .* V[k]
        end
        piece!(sp, p, kmid + 1, Val(Symbol(piece))); piece!(sm, p, kmid + 1, Val(Symbol(piece))); sync()
        Lp, Lm = leaves(sp), leaves(sm)
        fd = sum(Float64(sum(Float64.(R[k]) .* ((Float64.(Lp[k]) .- Float64.(Lm[k])) ./ (2 * Float64(ε)) .- Float64.(V[k])))) for k in keys(V))
        @printf("piece %s central differences (ε = %g): ⟨r,(F(x+εv)−F(x−εv))/2ε⟩−⟨r,v⟩ = %+.6e\n", piece, ε, fd)
        for k in sort!(collect(keys(V)))
            startswith(k, "st.aux.ionization") || continue
            d = Float64(sum(Float64.(R[k]) .* ((Float64.(Lp[k]) .- Float64.(Lm[k])) ./ (2 * Float64(ε)) .- Float64.(V[k]))))
            d != 0 && @printf("    fd %-50s %+.6e\n", k, d)
        end
    end
    @printf("piece %s check: reverse %.3f s; total ⟨Jᵀr,v⟩−⟨r,v⟩ = %+.6e, ⟨r,Jv⟩−⟨r,v⟩ = %+.6e, relative %.2e\n", piece, t, rev, fwd, abs(rev - fwd) / max(abs(fwd), 1e-30))
    for r in sort(rows; by = r -> -abs(r[2] - r[3]))[1:min(end, 12)]
        @printf("    %-50s rev %+.6e fwd %+.6e rel %.2e (|r·v| %.2e)\n", r...)
    end
end

