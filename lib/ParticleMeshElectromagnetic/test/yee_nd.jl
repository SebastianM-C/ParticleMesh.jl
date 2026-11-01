# T-320 to T-329: the Yee solver in N dimensions held to the exact discrete theory of its own
# scheme (WP-022), at N = 1 and N = 2, on the non-square grids of `nd_grid` (Δx ≠ Δy).
#
# The one-step map. With C⁺ the discrete curl built from forward differences (E's staggers to
# B's) and C⁻ the one built from backward differences (B's staggers to E's), solve_fields! is
#     Bⁿ⁺¹ᐟ² = Bⁿ − (Δt/2)·C⁺Eⁿ,   Eⁿ⁺¹ = Eⁿ + Δt·(C⁻Bⁿ⁺¹ᐟ² − J),   Bⁿ⁺¹ = Bⁿ⁺¹ᐟ² − (Δt/2)·C⁺Eⁿ⁺¹,
# so the half-step values obey the leapfrog Bⁿ⁺³ᐟ² = Bⁿ⁺¹ᐟ² − Δt·C⁺Eⁿ⁺¹ and the stored
# integer-time B is the mean of its two neighbouring half steps, Bⁿ = (Bⁿ⁻¹ᐟ² + Bⁿ⁺¹ᐟ²)/2. On a
# periodic grid (D⁺_d)ᵀ = −D⁻_d, hence (C⁺)ᵀ = C⁻ (each curl term pairs a difference with the
# opposite sign), and D⁺ along different axes commute, hence div⁺·C⁺ = 0 and div⁻·C⁻ = 0.
# On a mode e^{ik·x} sampled at each component's own stagger, both D⁺_d and D⁻_d act as i·κ_d with
# κ_d = 2 sin(k_dΔ_d/2)/Δ_d (the half-cell shift of the result is absorbed by the stagger), so the
# scheme is Maxwell's equations with k replaced by κ, advanced by a leapfrog.

const PARITY_T = Dict{String,Any}()   # bitwise observations recorded (not asserted), printed below

# ---------------------------------------------------------------------------------------------
# Host-side reference arithmetic in BigFloat, on the interiors (periodic shifts, no guards).

nparts(::Type{<:AbstractFloat}) = 0
nparts(::Type{Dual{Nothing,V,P}}) where {V,P} = P
part(x::AbstractFloat, k) = x
part(x::Dual, k) = k == 0 ? value(x) : partials(x)[k]

"Part `k` (0 the value, `k ≥ 1` a partial) of the interior of `a`, as a BigFloat array."
bigpart(a, grid, k) = map(x -> big(Float64(part(x, k))), collect(interior_view(a, grid)))
"Part `k` of every field of `f` as BigFloat interiors, a NamedTuple with the keys of `f`."
bigfields(f, grid, k) = map(a -> bigpart(a, grid, k), f)
bigΔ(grid) = map(d -> big(Float64(value_of(d))), grid.dx)

pshift(a, d, s) = circshift(a, ntuple(i -> i == d ? s : 0, ndims(a)))
dplus(a, d, Δ) = (pshift(a, d, -1) .- a) ./ Δ    # (a[j + e_d] − a[j])/Δ_d, at index j
dminus(a, d, Δ) = (a .- pshift(a, d, 1)) ./ Δ    # (a[j] − a[j − e_d])/Δ_d

"The discrete curl of the three components `F` with the differences `op` along the resolved axes."
function curl_big(F, Δ, op)
    N = length(Δ)
    cyc = ((2, 3), (3, 1), (1, 2))
    return ntuple(3) do c
        c1, c2 = cyc[c]
        r = zero.(F[c])
        c1 <= N && (r = r .+ op(F[c2], c1, Δ[c1]))
        c2 <= N && (r = r .- op(F[c1], c2, Δ[c2]))
        r
    end
end
Evec(u) = (u.Ex, u.Ey, u.Ez)
Bvec(u) = (u.Bx, u.By, u.Bz)
Jvec(u) = (u.Jx, u.Jy, u.Jz)
field_dot3(a, b) = sum(sum(a[c] .* b[c]) for c in 1:3)

"""
    energy_form(u, v, Δ, dt)

The symmetric bilinear form of the scheme's conserved energy, twice the energy per cell volume:
W(u, v) = Σ E_u·E_v + Σ B_u·B_v − (Δt/2)²·Σ (C⁺E_u)·(C⁺E_v). With Bⁿ±¹ᐟ² = Bⁿ ∓ (Δt/2)C⁺Eⁿ,
W(u, u) = Σ|Eⁿ|² + Σ Bⁿ⁻¹ᐟ²·Bⁿ⁺¹ᐟ² (T-210's energy at N = 1). Over one step,
W(uⁿ⁺¹) − W(uⁿ) = (Eⁿ⁺¹ − Eⁿ)·(Eⁿ⁺¹ + Eⁿ) + Bⁿ⁺¹ᐟ²·(Bⁿ⁺³ᐟ² − Bⁿ⁻¹ᐟ²)
               = Δt(C⁻Bⁿ⁺¹ᐟ² − J)·(Eⁿ + Eⁿ⁺¹) − Δt Bⁿ⁺¹ᐟ²·C⁺(Eⁿ + Eⁿ⁺¹) = −Δt Σ J·(Eⁿ + Eⁿ⁺¹),
since (C⁺)ᵀ = C⁻: conserved in vacuum, and with a current the change is the discrete Poynting
theorem's work term (the flux divergence sums to zero on the periodic grid). The positive
definiteness for Δt below the Courant limit is what bounds the fields.
"""
function energy_form(u, v, Δ, dt)
    h = big(Float64(value_of(dt))) / 2
    Cu, Cv = curl_big(Evec(u), Δ, dplus), curl_big(Evec(v), Δ, dplus)
    return field_dot3(Evec(u), Evec(v)) + field_dot3(Bvec(u), Bvec(v)) - h^2 * field_dot3(Cu, Cv)
end

"Backward-difference divergence Σ_{d≤N} D⁻_d F_d (on ρ's nodes) and forward one Σ D⁺_d F_d (cell centres)."
div_minus(F, Δ) = sum(dminus(F[d], d, Δ[d]) for d in eachindex(Δ))
div_plus(F, Δ) = sum(dplus(F[d], d, Δ[d]) for d in eachindex(Δ))

maxabs_big(F) = maximum(a -> maximum(abs, a), F)

# ---------------------------------------------------------------------------------------------
# The discrete eigenmode.

stagger_shifts(s::ParticleMeshCore.Stagger) = (s isa Center ? 1 // 2 : 0 // 1,)
stagger_shifts(s::Tuple) = map(x -> x isa Center ? 1 // 2 : 0 // 1, s)

"""
    yee_mode(grid, m, pol, dt) -> (; m, Ê, B̂, ω, κ)

The discrete eigenmode of integer wave numbers `m` (per axis) on `grid`: k_d = 2π m_d/(n_dΔ_d),
κ_d = 2 sin(k_dΔ_d/2)/Δ_d. Substituting Eⁿ = Ê e^{i(k·x − ωnΔt)}, Bⁿ⁺¹ᐟ² = B̂½ e^{i(k·x − ω(n+½)Δt)}
into the leapfrog of the header gives Ω·B̂½ = κ×Ê and −Ω·Ê = κ×B̂½ with Ω = 2 sin(ωΔt/2)/Δt, so
Ω² = |κ|² for κ·Ê = 0: the exact discrete relation sin(ωΔt/2)/Δt = √(Σ_d sin²(k_dΔ_d/2)/Δ_d²).
The integer-time B is the mean of its half steps, B̂ = cos(ωΔt/2)·B̂½. Polarization 1 (Ex, Ey, Bz
at N = 2; Ey, Bz at N = 1): Ê = (−κ_y, κ_x, 0)/|κ|, B̂½ = κ×Ê/|κ| = (0, 0, 1). Polarization 2
(Bx, By, Ez; By, Ez at N = 1): Ê = (0, 0, 1), B̂½ = (κ_y, −κ_x, 0)/|κ|. The real fields are
F_c = A·F̂_c·cos(k·x_c − ωt) with x_c the position of each component's stagger.
"""
function yee_mode(grid, m, pol, dt)
    N = length(grid.n)
    Δ = bigΔ(grid)
    κ = ntuple(d -> 2 * sin(big(π) * m[d] / grid.n[d]) / Δ[d], N)      # k_dΔ_d/2 = π m_d/n_d
    κx, κy = κ[1], N >= 2 ? κ[2] : big(0)
    nκ = sqrt(sum(abs2, κ))
    bdt = big(Float64(value_of(dt)))
    ω = 2 / bdt * asin(bdt * nκ / 2)
    c½ = cos(ω * bdt / 2)
    o = big(0)
    Ê, B̂ = pol == 1 ? ((-κy / nκ, κx / nκ, o), (o, o, c½)) : ((o, o, big(1)), (c½ * κy / nκ, -c½ * κx / nκ, o))
    return (; m, Ê, B̂, ω, κ)
end

"The exact value of component `key` of `mode` at interior index `J` at time `t` (BigFloat)."
function mode_value(mode, layout, key, J, n, t)
    s = stagger_shifts(getfield(layout, key))
    θ = 2 * big(π) * sum((J[d] - 1 + s[d]) * mode.m[d] // n[d] for d in eachindex(n)) - mode.ω * t
    c = findfirst(==(key), EB_KEYS)
    amp = c <= 3 ? mode.Ê[c] : mode.B̂[c - 3]
    return amp * cos(θ)
end

"Fields of `T` holding `A` times the mode at t = 0 (guards filled), J zero."
function mode_fields(::Type{T}, grid, mode, A) where {T}
    N = length(grid.n)
    layout = field_layout(Val(N))
    f = em_fields(CPU(), T, grid)
    for key in EB_KEYS
        a = getfield(f, key)
        v = interior_view(a, grid)
        for J in CartesianIndices(v)
            v[J] = A * T(Float64(mode_value(mode, layout, key, J, grid.n, big(0))))
        end
        fill_guards!(a, grid)
    end
    return f
end

"Largest difference over the value and the partials between `x` and `A·exact` (BigFloat exact)."
part_err(x, A, exact) = maximum(k -> Float64(abs(big(Float64(part(x, k))) - big(Float64(part(A, k))) * exact)),
                                0:nparts(typeof(x)))

# ---------------------------------------------------------------------------------------------

@testset "T-320 discrete dispersion: an oblique eigenmode in each polarization follows the exact Yee relation [A2, A7, WP-022]" begin
    C = emC("T-320")
    Cp = emC("T-320", "C_phase")
    nsteps = 400
    for N in (1, 2), T in TYPES, pol in (1, 2), cfl in (1 // 2, 95 // 100)
        B = basetype(T)
        grid = nd_grid(Val(N), B)
        dt = B(cfl) * courant_limit(Yee(), grid)
        m = N == 1 ? (3,) : (3, -2)
        mode = yee_mode(grid, m, pol, dt)
        layout = field_layout(Val(N))
        A = seeded(T, 1)                                   # partials exercised for duals
        f = mode_fields(T, grid, mode, A)
        bdt = big(Float64(dt))
        worst = 0.0
        worstp = 0.0
        key = pol == 1 ? :Bz : :Ez                         # the component of full amplitude
        amp = pol == 1 ? mode.B̂[3] : mode.Ê[3]
        ntot = prod(grid.n)
        for s in 1:nsteps
            solve_fields!(Yee(), f, grid, dt)
            s % 100 == 0 || continue
            t = s * bdt
            err = 0.0
            for k in EB_KEYS
                v = interior_view(getfield(f, k), grid)
                for J in CartesianIndices(v)
                    err = max(err, part_err(v[J], A, mode_value(mode, layout, k, J, grid.n, t)))
                end
            end
            worst = max(worst, err / (base_eps(T) * s))
            # Phase and amplitude of the mode, from its Fourier coefficient Σ F·e^{−ik·x}
            # = A·F̂·(n/2)·e^{−iωt} (the e^{+2ik·x} half sums to zero for these m and n).
            v = interior_view(getfield(f, key), grid)
            sh = stagger_shifts(getfield(layout, key))
            ĉ = sum(big(Float64(value_of(v[J]))) * cis(-2 * big(π) * sum((J[d] - 1 + sh[d]) * m[d] // grid.n[d] for d in 1:N))
                    for J in CartesianIndices(v))
            Δφ = mod(-angle(ĉ) - mode.ω * t + big(π), 2 * big(π)) - big(π)
            Δa = abs(ĉ) * 2 / ntot - abs(amp)
            worstp = max(worstp, Float64(max(abs(Δφ), abs(Δa) / abs(amp))) / (base_eps(T) * s))
        end
        record!("T-320", T, worst)
        record!("T-320 phase and amplitude", T, worstp)
        @test worst <= C
        @test worstp <= Cp
    end
end

# Phase-velocity error 1 − ω/|k| of the code, measured from the phase of the Fourier
# coefficient of Ez advanced through solve_fields! over one period (Float64, polarization 2,
# started on the discrete eigenmode).
function phase_velocity_error(N, scale, m; cfl = 95 // 100)
    grid = N == 1 ? PeriodicGrid(16scale, 2 * π / (16scale), 2) :
           PeriodicGrid((16scale, 12scale), (2 * π / (16scale), 2 * π / (12scale)), 2)
    dt = Float64(cfl) * courant_limit(Yee(), grid)
    mode = yee_mode(grid, m, 2, dt)
    f = mode_fields(Float64, grid, mode, 1.0)
    phases = [cispi(-2 * sum((J[d] - 1) * m[d] / grid.n[d] for d in 1:N)) for J in CartesianIndices(grid.n)]
    proj() = sum(interior_view(f.Ez, grid) .* phases)
    nsteps = round(Int, 2π / dt)
    c0 = proj()
    φ = 0.0
    for _ in 1:nsteps
        solve_fields!(Yee(), f, grid, dt)
        c1 = proj()
        φ += angle(c0 / c1)                                # ωΔt per step, |ωΔt| < π
        c0 = c1
    end
    ω = φ / (nsteps * dt)
    kabs = sqrt(sum(abs2, m))                              # the box is 2π per axis
    return (; err = 1 - ω / kabs, ω, ω_discrete = Float64(mode.ω), dx = grid.dx)
end

@testset "T-321 convergence to Maxwell: the phase-velocity error falls as Δ² at Courant number 0.95 [A2, A7, WP-022]" begin
    tol = emC("T-321", "order_tol")
    worst = 0.0
    for (N, m, label) in ((1, (1,), "N = 1"), (2, (1, 0), "N = 2 axis"), (2, (1, 1), "N = 2 diagonal"))
        r = [phase_velocity_error(N, s, m) for s in (1, 2, 4)]
        errs = [x.err for x in r]
        orders = [log2(errs[i] / errs[i + 1]) for i in 1:2]
        @info "T-321 phase-velocity error 1 − ω/|k|" label errs orders ω_rel_to_discrete = [x.ω / x.ω_discrete - 1 for x in r]
        PARITY_T["T-321 $label"] = (; errs, orders)
        @test all(>(0), errs)                              # Yee is slower than light below the limit
        @test all(o -> abs(o - 2) <= tol, orders)
        worst = max(worst, maximum(o -> abs(o - 2), orders))
    end
    MEASURED["T-321 |order − 2|"] = Dict("Float64" => worst)
    ax, dg = PARITY_T["T-321 N = 2 axis"].errs, PARITY_T["T-321 N = 2 diagonal"].errs
    @info "T-321 anisotropy (diagonal error / axis error)" ratio = dg ./ ax
end

@testset "T-322 divergence identities: ∇·B is kept, and ∇·E changes by −Δt·∇·J [A2, A7, WP-022]" begin
    C = emC("T-322")
    Cj = emC("T-322", "C_current")
    for N in (1, 2), T in TYPES
        grid = nd_grid(Val(N), basetype(T))
        Δ = bigΔ(grid)
        L = sum(d -> 2 / d, Δ)                             # the size of a divergence's terms per unit field
        dt = basetype(T)(95 // 100) * courant_limit(Yee(), grid)
        # ∇·B on the cell centres: D⁺ commute, so div⁺·C⁺ = 0 and every B update keeps it.
        f = random_fields(T, grid; seed = 3220 + N)
        b0 = [bigfields(f, grid, k) for k in 0:nparts(T)]
        div0 = [div_plus(Bvec(u), Δ) for u in b0]
        worst = 0.0
        for s in 1:200
            solve_fields!(Yee(), f, grid, dt)
            s % 50 == 0 || continue
            for k in 0:nparts(T)
                u = bigfields(f, grid, k)
                M = max(maxabs_big(Bvec(u)), maxabs_big(Bvec(b0[k + 1])))
                worst = max(worst, Float64(maximum(abs, div_plus(Bvec(u), Δ) .- div0[k + 1]) / (M * L)) / (base_eps(T) * s))
            end
        end
        record!("T-322 div B", T, worst)
        @test worst <= C
        # ∇·E on ρ's nodes with a prescribed J: Eⁿ⁺¹ − Eⁿ = Δt(C⁻B − J) and div⁻·C⁻ = 0, so
        # div⁻Eⁿ⁺¹ − div⁻Eⁿ = −Δt·div⁻J over every step.
        f = random_fields(T, grid; seed = 3225 + N)
        worstj = 0.0
        for s in 1:10
            before = [bigfields(f, grid, k) for k in 0:nparts(T)]
            solve_fields!(Yee(), f, grid, dt)
            for k in 0:nparts(T)
                u0, u1 = before[k + 1], bigfields(f, grid, k)
                bdt = big(Float64(dt))
                res = div_minus(Evec(u1), Δ) .- div_minus(Evec(u0), Δ) .+ bdt .* div_minus(Jvec(u0), Δ)
                M = max(maxabs_big(Evec(u0)), maxabs_big(Evec(u1)), maxabs_big(Bvec(u0)), maxabs_big(Bvec(u1)),
                        bdt * maxabs_big(Jvec(u0)))
                worstj = max(worstj, Float64(maximum(abs, res) / (M * L)) / base_eps(T))
            end
        end
        record!("T-322 div E with J", T, worstj)
        @test worstj <= Cj
    end
end

@testset "T-323 discrete energy: conserved in vacuum, changed by the discrete Poynting work term with J [A7, WP-022]" begin
    C = emC("T-323")
    Cj = emC("T-323", "C_current")
    for N in (1, 2), T in TYPES
        grid = nd_grid(Val(N), basetype(T))
        Δ = bigΔ(grid)
        dt = basetype(T)(95 // 100) * courant_limit(Yee(), grid)
        P = nparts(T)
        # Vacuum: W(u, u) is invariant under the one-step map M, so is the symmetric form
        # W(u, v) (polarization); for duals W(value, partial_k) is checked with W(value, value).
        f = random_fields(T, grid; seed = 3230 + N, current = false)
        u0 = [bigfields(f, grid, k) for k in 0:P]
        W0 = [energy_form(u0[1], u0[k + 1], Δ, dt) for k in 0:P]
        nrm = [sqrt(energy_form(u0[1], u0[1], Δ, dt) * energy_form(u0[k + 1], u0[k + 1], Δ, dt)) for k in 0:P]
        worst = 0.0
        for s in 1:400
            solve_fields!(Yee(), f, grid, dt)
            s % 100 == 0 || continue
            u = [bigfields(f, grid, k) for k in 0:P]
            for k in 0:P
                worst = max(worst, Float64(abs(energy_form(u[1], u[k + 1], Δ, dt) - W0[k + 1]) / nrm[k + 1]) / (base_eps(T) * s))
            end
        end
        record!("T-323 vacuum", T, worst)
        @test worst <= C
        # A prescribed current: W(uⁿ⁺¹, vⁿ⁺¹) − W(uⁿ, vⁿ) = −(Δt/2)·[J_u·(E_vⁿ + E_vⁿ⁺¹) + J_v·(E_uⁿ + E_uⁿ⁺¹)]
        # (the polarized form of the header identity), each step.
        f = random_fields(T, grid; seed = 3235 + N)
        bdt = big(Float64(dt))
        worstj = 0.0
        for s in 1:10
            a = [bigfields(f, grid, k) for k in 0:P]
            solve_fields!(Yee(), f, grid, dt)
            b = [bigfields(f, grid, k) for k in 0:P]
            for k in 0:P
                u0, v0, u1, v1 = a[1], a[k + 1], b[1], b[k + 1]
                Esum_u = Evec(u0) .+ Evec(u1)
                Esum_v = Evec(v0) .+ Evec(v1)
                work = -bdt / 2 * (field_dot3(Jvec(u0), Esum_v) + field_dot3(Jvec(v0), Esum_u))
                res = energy_form(u1, v1, Δ, dt) - energy_form(u0, v0, Δ, dt) - work
                absw = bdt / 2 * (field_dot3(map(x -> abs.(x), Jvec(u0)), map(x -> abs.(x), Esum_v)) +
                                  field_dot3(map(x -> abs.(x), Jvec(v0)), map(x -> abs.(x), Esum_u)))
                scale = sqrt(energy_form(u0, u0, Δ, dt) * energy_form(v0, v0, Δ, dt)) +
                        sqrt(energy_form(u1, u1, Δ, dt) * energy_form(v1, v1, Δ, dt)) + absw
                worstj = max(worstj, Float64(abs(res) / scale) / base_eps(T))
            end
        end
        record!("T-323 with J", T, worstj)
        @test worstj <= Cj
    end
end

# The axis exchange (x, y, z) ↦ (y, x, z), a reflection (det = −1): a vector field maps as
# E′(r) = P·E(P⁻¹r), a pseudovector as B′(r) = det(P)·P·B(P⁻¹r) = −P·B(P⁻¹r):
#     E′x = Eyᵀ, E′y = Exᵀ, E′z = Ezᵀ, J likewise;   B′x = −Byᵀ, B′y = −Bxᵀ, B′z = −Bzᵀ,
# with ᵀ the transposed array (guards included) on the grid with n and Δ exchanged. Maxwell's
# equations are invariant: ∂ₜB′z = −(∂x E′y − ∂y E′x) holds because ∂x E′y = (∂y Ex)ᵀ and
# ∂y E′x = (∂x Ey)ᵀ, and ∂ₜE′z = ∂x B′y − ∂y B′x − J′z because ∂x B′y = −(∂y Bx)ᵀ and
# ∂y B′x = −(∂x By)ᵀ. The staggers map onto each other: Ey at (Node, Center) transposed is
# (Center, Node), Ex's; Bx ↔ By likewise; Ez and Bz map onto themselves.
function exchange_axes(f)
    t(a) = permutedims(a, (2, 1))
    return (Ex = t(f.Ey), Ey = t(f.Ex), Ez = t(f.Ez), Bx = -t(f.By), By = -t(f.Bx), Bz = -t(f.Bz),
            Jx = t(f.Jy), Jy = t(f.Jx), Jz = t(f.Jz))
end
maxpart(x) = maximum(abs, components(x))

@testset "T-324 reductions and symmetry: a y-invariant 2D run is the 1D run; the axis exchange maps runs onto runs [A7, WP-022]" begin
    C = emC("T-324")
    Cx = emC("T-324", "C_exchange")
    nsteps = 100
    for T in TYPES
        B = basetype(T)
        g1 = nd_grid(Val(1), B)
        g2 = PeriodicGrid((ncells(g1), 8), (cellsize(g1), B(3) / B(10)), guards(g1))
        dt = B(95 // 100) * courant_limit(Yee(), g2)
        f1 = random_fields(T, g1; seed = 3240)
        f2 = em_fields(CPU(), T, g2)
        for k in keys(f2), j in axes(getfield(f2, k), 2)
            getfield(f2, k)[:, j] .= getfield(f1, k)
        end
        for _ in 1:nsteps
            solve_fields!(Yee(), f1, g1, dt)
            solve_fields!(Yee(), f2, g2, dt)
        end
        M = maximum(k -> maximum(maxpart, getfield(f1, k)), EB_KEYS)
        err = maximum(k -> maximum(j -> maximum(absdiff.(getfield(f2, k)[:, j], getfield(f1, k))), axes(getfield(f2, k), 2)),
                      EB_KEYS)
        bitwise = all(k -> all(j -> same_bits(getfield(f2, k)[:, j], getfield(f1, k)), axes(getfield(f2, k), 2)), keys(f2))
        PARITY_T["T-324 y-invariant bitwise $(type_label(T))"] = bitwise
        record!("T-324 y-invariant", T, err / (base_eps(T) * nsteps * M))
        @test err <= C * base_eps(T) * nsteps * M

        g = nd_grid(Val(2), B)
        gx = PeriodicGrid(reverse(g.n), reverse(g.dx), guards(g))
        dt = B(95 // 100) * courant_limit(Yee(), g)
        f = random_fields(T, g; seed = 3245)
        fx = exchange_axes(f)
        for _ in 1:nsteps
            solve_fields!(Yee(), f, g, dt)
            solve_fields!(Yee(), fx, gx, dt)
        end
        mapped = exchange_axes(f)
        M = maximum(k -> maximum(maxpart, getfield(f, k)), EB_KEYS)
        err = maximum(k -> maximum(absdiff.(getfield(mapped, k), getfield(fx, k))), EB_KEYS)
        PARITY_T["T-324 exchange bitwise $(type_label(T))"] = all(k -> same_bits(getfield(mapped, k), getfield(fx, k)), EB_KEYS)
        record!("T-324 exchange", T, err / (base_eps(T) * nsteps * M))
        @test err <= Cx * base_eps(T) * nsteps * M
    end
end

# The highest mode of the grid, k_d = π/Δ_d on every resolved axis (the diagonal at N = 2): the
# fields alternate in sign from node to node, F = a·(−1)^{Σ_d (j_d − 1)} on every stagger, and
# D⁺_d, D⁻_d act as −2/Δ_d and +2/Δ_d. Polarization 2 reduces to the amplitude a of Ez and the
# combination β = (b_y/Δx − b_x/Δy)/ν of the B amplitudes, ν = √(Σ_d 1/Δ_d²), with S = Δt·ν (the
# Courant-scaled symbol, Δt·|κ|/2 at the highest mode): the three sub-steps are β ← β − S·a,
# a ← a + 2S·β, β ← β − S·a, the 2×2 one-step matrix M = [1 − 2S², 2S; −2S(1 − S²), 1 − 2S²] with
# det M = 1 and tr M = 2(1 − 2S²). Its eigenvalues λ solve λ² − 2(1 − 2S²)λ + 1 = 0: on the unit
# circle for S ≤ 1, real and negative for S > 1 with |λ| = 2S² − 1 + 2S√(S² − 1) = (S + √(S² − 1))²
# for the growing one, eigenvector β = −√(S² − 1)·a. Polarization 1 (Ex, Ey, Bz) is the same map
# for (ε, −b) with ε = (e_y/Δx − e_x/Δy)/ν. At N = 1 the y terms are absent and ν = 1/Δx.
# The energy form of T-323 is conserved by this map for every S (above S = 1 it is indefinite and
# the growing eigenvector is a null vector), so the energy alone does not show the instability:
# the test also holds the run to the closed form Mⁿ and asserts |tr M/2| ≤ 1 below the limit.
"""
    nyquist_fields(T, grid, A, a, r)

`A` times the highest mode with Ez amplitude `a` and β = −r (polarization 2), and ε = a, b = r
(polarization 1), the static combinations zero: E = a·(−1/(νΔy), 1/(νΔx), 1),
B = r·(1/(νΔy), −1/(νΔx), 1) times the alternating sign.
"""
function nyquist_fields(::Type{T}, grid, A, a, r) where {T}
    N = length(grid.n)
    Δ = map(d -> big(Float64(value_of(d))), grid.dx)
    ν = sqrt(sum(d -> 1 / d^2, Δ))
    iΔy = N >= 2 ? 1 / Δ[2] : big(0)
    amp = (Ex = -a * iΔy / ν, Ey = a / Δ[1] / ν, Ez = a, Bx = r * iΔy / ν, By = -r / Δ[1] / ν, Bz = r)
    f = em_fields(CPU(), T, grid)
    for key in EB_KEYS
        x = getfield(f, key)
        v = interior_view(x, grid)
        for J in CartesianIndices(v)
            v[J] = A * T(Float64((-1)^sum(Tuple(J) .- 1) * getfield(amp, key)))
        end
        fill_guards!(x, grid)
    end
    return f
end

"The one-step matrix of the highest mode on (a, β) for the Courant-scaled symbol `S`, from the three sub-steps (BigFloat)."
function nyquist_matrix(S)
    sub(a, β) = (β1 = β - S * a; a1 = a + 2S * β1; (a1, β1 - S * a1))
    c1, c2 = sub(big(1), big(0)), sub(big(0), big(1))
    return [c1[1] c2[1]; c1[2] c2[2]]
end

"Largest deviation of every E and B entry of `f` from the reference fields `g` (BigFloat), over the parts."
function fields_dev(f, g, grid)
    err = big(0)
    for key in EB_KEYS
        v, w = interior_view(getfield(f, key), grid), interior_view(getfield(g, key), grid)
        for J in eachindex(v), k in 0:nparts(eltype(v))
            err = max(err, abs(big(Float64(part(v[J], k))) - big(Float64(part(w[J], k)))))
        end
    end
    return err
end

@testset "T-325 the Courant limit is where the scheme becomes unstable: bounded at 0.99, growth (S + √(S² − 1))² at 1.01 [A2, A7, WP-022]" begin
    C = emC("T-325")
    Cr = emC("T-325", "C_rotation")
    Cg = emC("T-325", "C_growth")
    for N in (1, 2), T in TYPES
        B = basetype(T)
        grid = nd_grid(Val(N), B)
        Δ = bigΔ(grid)
        ν = sqrt(sum(d -> 1 / d^2, Δ))
        A = seeded(T, 1)
        # 0.99 of the limit: |tr M/2| ≤ 1, so |λ| = 1 and the run follows Mⁿ·(1, 0); and the energy
        # form of T-323, which is positive definite only for S < 1, is kept.
        dt = B(99 // 100) * courant_limit(Yee(), grid)
        S = big(Float64(dt)) * ν
        M = nyquist_matrix(S)
        @test abs((M[1, 1] + M[2, 2]) / 2) <= 1
        f = nyquist_fields(T, grid, A, big(1), big(0))
        u0 = bigfields(f, grid, 0)
        W0 = energy_form(u0, u0, Δ, dt)
        nsteps = 400
        for _ in 1:nsteps
            solve_fields!(Yee(), f, grid, dt)
        end
        u = bigfields(f, grid, 0)
        ratio = Float64(abs(energy_form(u, u, Δ, dt) - W0) / W0) / (base_eps(T) * nsteps)
        record!("T-325 energy at 0.99", T, ratio)
        @test ratio <= C
        aβ = M^nsteps * [big(1), big(0)]
        dev = fields_dev(f, nyquist_fields(T, grid, A, aβ[1], -aβ[2]), grid)
        ratio = Float64(dev) / (base_eps(T) * nsteps)
        record!("T-325 closed form at 0.99", T, ratio)
        @test ratio <= Cr
        # 1.01 of the limit: the growing eigenvalue of M against the closed form, and the run
        # seeded on its eigenvector, which grows by λ per step.
        dt = B(101 // 100) * courant_limit(Yee(), grid)
        S = big(Float64(dt)) * ν
        @test S > 1
        M = nyquist_matrix(S)
        λ = (M[1, 1] + M[2, 2]) / 2 - sqrt(((M[1, 1] + M[2, 2]) / 2)^2 - 1)   # det M = 1, the growing root
        @test abs(M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1] - 1) < big(10)^-60
        @test abs(abs(λ) - (S + sqrt(S^2 - 1))^2) < big(10)^-60
        r = (λ - M[1, 1]) / M[1, 2]                         # β/a of the eigenvector, −√(S² − 1)
        @test abs(r + sqrt(S^2 - 1)) < big(10)^-60
        f = nyquist_fields(T, grid, A, big(1), -r)
        f0 = map(copy, f)
        ngrow = 40
        for _ in 1:ngrow
            solve_fields!(Yee(), f, grid, dt)
        end
        λn = λ^ngrow
        err = big(0)
        for key in EB_KEYS
            v, v0 = interior_view(getfield(f, key), grid), interior_view(getfield(f0, key), grid)
            for J in eachindex(v), k in 0:nparts(T)
                err = max(err, abs(big(Float64(part(v[J], k))) - λn * big(Float64(part(v0[J], k)))) / abs(λn))
            end
        end
        M0 = maximum(k -> maximum(maxpart, getfield(f0, k)), EB_KEYS)
        ratio = Float64(err) / (M0 * base_eps(T) * ngrow)
        record!("T-325 growth at 1.01", T, ratio)
        @test ratio <= Cg
        @info "T-325 amplification per step at 1.01 of the limit" N T = type_label(T) S = Float64(S) λ = Float64(λ)
    end
end

@testset "T-326 the state constructor and check_courant reject Δt at or above the Courant limit, naming it [A2, A7, WP-022]" begin
    case = load_case("two-stream")
    for T in TYPES
        num = numerics_from_case(case; mode = Reference())
        prob = problem_from_case(case, T; ppc = 4)
        limit = courant_limit(Yee(), PeriodicGrid(prob.cells, prob.length / prob.cells, 1))
        @test limit == prob.length / prob.cells                  # exactly Δx at N = 1
        mk(f) = problem_from_case(case, T; ppc = 4, dt = Float64(value_of(f * limit)))
        @test EMState(mk(basetype(T)(95 // 100)), num) isa EMState
        for f in (one(basetype(T)), basetype(T)(101 // 100))
            err = try
                EMState(mk(f), num)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin(string(limit), sprint(showerror, err))
        end
        # N = 2: the helper the N = 2 state constructor calls (WP-024).
        g = nd_grid(Val(2), basetype(T))
        lim2 = courant_limit(Yee(), g)
        @test lim2 < minimum(g.dx)
        @test check_courant(Yee(), g, basetype(T)(95 // 100) * lim2) == basetype(T)(95 // 100) * lim2
        for f in (one(basetype(T)), basetype(T)(101 // 100))
            err = try
                check_courant(Yee(), g, f * lim2)
                nothing
            catch e
                e
            end
            @test err isa ArgumentError
            @test occursin(string(lim2), sprint(showerror, err))
        end
    end
end

@testset "T-327 the solver is bitwise across reruns, 1 and 4 threads and launch shapes, at N = 1 and N = 2 [A4, WP-022]" begin
    first = yee_nd_digests()
    @test yee_nd_digests() == first
    for wgs in (16, 64)
        @test yee_nd_digests(; workgroupsize = wgs) == first
    end
    worker = joinpath(@__DIR__, "yee_nd_worker.jl")
    project = dirname(Base.active_project())
    for nt in (1, 4)
        out = read(`$(Base.julia_cmd()) --startup-file=no --threads=$nt --project=$project $worker`, String)
        lines = filter(startswith("YEEND"), split(out, '\n'))
        got = Dict((parse(Int, f[3]), String(f[4])) => parse(UInt64, f[5]) for f in split.(lines))
        @test all(f -> parse(Int, f[2]) == nt, split.(lines))
        @test length(got) == length(first)
        for (N, T, d) in first
            @test get(got, (N, type_label(T)), nothing) == d
        end
    end
end

@testset "T-328 Dual: the partials evolve as the solver applied to the partials; the primal as the Float32 run [A8, WP-022]" begin
    C = emC("T-328")
    nsteps = 100
    for N in (1, 2)
        grid = nd_grid(Val(N), Float32)
        dt = Float32(95 // 100) * courant_limit(Yee(), grid)
        fd = random_fields(D3, grid; seed = 3280 + N)
        fs = [map(a -> map(x -> part(x, k), a), fd) for k in 0:nparts(D3)]
        for _ in 1:nsteps
            solve_fields!(Yee(), fd, grid, dt)
            foreach(f -> solve_fields!(Yee(), f, grid, dt), fs)
        end
        for k in 0:nparts(D3)
            M = maximum(key -> maximum(abs, getfield(fs[k + 1], key)), EB_KEYS)
            err = maximum(key -> maximum(abs.(part.(getfield(fd, key), k) .- getfield(fs[k + 1], key))), EB_KEYS)
            bitwise = all(key -> same_bits(part.(getfield(fd, key), k), getfield(fs[k + 1], key)), keys(fd))
            PARITY_T["T-328 N = $N part $k bitwise"] = bitwise
            record!(k == 0 ? "T-328 primal" : "T-328 partials", D3, err / (eps(Float32) * nsteps * M))
            @test err <= C * eps(Float32) * nsteps * M
        end
    end
end

@testset "T-329 guards are periodic images after solve_fields!, and its allocation does not grow with the grid [A7, WP-022]" begin
    for N in (1, 2), T in TYPES
        grid = nd_grid(Val(N), basetype(T))
        f = random_fields(T, grid; seed = 3290 + N)
        dt = basetype(T)(95 // 100) * courant_limit(Yee(), grid)
        before = map(copy, f)
        for _ in 1:3
            solve_fields!(Yee(), f, grid, dt)
        end
        for key in EB_KEYS
            a = getfield(f, key)
            @test same_bits(a, fill_guards!(copy(a), grid))
        end
        # every component the solve changes is dynamic: all of E and B at N = 2 (Bx included),
        # all but Bx at N = 1
        changed = filter(key -> !same_bits(getfield(f, key), getfield(before, key)), EB_KEYS)
        @test changed == (N == 1 ? (:Ex, :Ey, :Ez, :By, :Bz) : EB_KEYS)
    end
    function solve_bytes(grid)
        f = random_fields(Float64, grid; seed = 3299)
        dt = 0.95 * courant_limit(Yee(), grid)
        solve_fields!(Yee(), f, grid, dt)
        solve_fields!(Yee(), f, grid, dt)
        return @allocated solve_fields!(Yee(), f, grid, dt)
    end
    for N in (1, 2)
        small = solve_bytes(nd_grid(Val(N), Float64))
        large = solve_bytes(nd_grid(Val(N), Float64; scale = 8))
        @info "T-329 bytes per solve_fields!" N small large
        @test large <= small
    end
end

println("\nWP-022 observations (recorded, not asserted):")
for k in sort(collect(keys(PARITY_T)))
    println("  ", k, ": ", PARITY_T[k])
end
