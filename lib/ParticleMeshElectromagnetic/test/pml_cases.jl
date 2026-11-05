# WP-034: the convolutional PML's runs and its discrete theory, shared by pml.jl and
# scripts/em2d/pml.jl. The runs are WP-027's slab packet (boundaries_cases.jl) with a `PML` as the
# open axis' absorber; the theory is the scheme's frequency-domain reflection of the PML slab, as
# `layer_rt` is that of the matched layer.

using ParticleMeshElectromagnetic: pml_profile, pml_sigma_max

"The PML of a reflection run on `geo`: `cells` from the geometry, the rest as given (PIConGPU's profile by default)."
slab_pml(geo; kw...) = PML(; cells = geo.cells, kw...)

"σ, κ, α of `pml` at the half row `p` of the run's box (any integer: wrapped into the box), the kernel's profile in Float64."
function pml_at(geo, pml, p::Integer)
    sm = pml_sigma_max(pml, Float64(geo.dy))
    return pml_profile(sm, pml.kappa_max, pml.alpha_max, (pml.order, pml.kappa_order, pml.alpha_order), mod(p, 2 * geo.ny), geo.ny,
                       geo.cells)
end

"""
    pml_rt(geo, pml, pol, ky) -> (r, t)

The reflection and transmission amplitudes of the PML slab (the top layer, the seam and the bottom
layer) for the scheme's mode (k_x, k_y) at its Yee ω, from the time-harmonic step. With every field
∝ e^{i k_x x}·λⁿ, λ = e^{−iωΔt}, a y-difference D_yF of a sub-step of length τ becomes
(1/κ)D_yF + ψ and ψ ← bψ + c·D_yF with b = e^{−(σ/κ + α)τ}, c = σ(b − 1)/(κ(σ + κα)), written
here from the formula, not from the kernel. Over a step, the B half steps' ψ (values P₁ after the
first, P₂ after the second) give P₁ = c(1 + b)D/(1 − b²/λ), P₂ = bP₁ + cλD for the y-difference D
of E at time n, so the half-step B is X^{n+½} = ∓G·D with

    G = h·(2/κ + c + c(1 + b)(1 + b/λ)/(1 − b²/λ))/(1 − 1/λ)   (h = Δt/2, at B's Center() row),

and the E step's ψ makes its y-difference S·D, S = 1/κ + λc/(λ − b) at E's Node() row (τ = Δt).
Eliminating By (TE) or Ey (TM), whose updates carry x-differences only (D⁻_xD⁺_x = −κ_x²):

    TE: (λ − 1)·e_j + Δt²κ_x²/(1 − 1/λ)·e_j − (ΔtS_j/Δy²)·(G_{j+½}(e_{j+1} − e_j) − G_{j−½}(e_j − e_{j−1})) = 0
    TM: (1 + Δt²κ_x²/((1 − 1/λ)(λ − 1)))·u_j − (ΔtG_j/(Δy²(λ − 1)))·(S_{j+1}(u_{j+1} − u_j) − S_j(u_j − u_{j−1})) = 0

for Ez on the Node() rows (TE) and the half-step Bz on the Center() rows (TM). In vacuum (G = Δt/(1 − 1/λ),
S = 1) both reduce to the Yee dispersion relation. Outgoing above the slab, iterated down through it
and split below it into the incident and reflected vacuum roots, as `layer_rt` does. Independent of
the kernels: it shares only the profile `pml_profile`.
"""
function pml_rt(geo, pml, pol, ky)
    dt, dy = geo.dt, geo.dy
    h = dt / 2
    λ = cis(-yee_omega(geo, ky) * dt)
    κx = 2 * sin(geo.kx * geo.dx / 2) / geo.dx
    function bc(p, τ)
        σ, κ, α = pml_at(geo, pml, p)
        b = exp(-(σ / κ + α) * τ)
        c = σ > 0 ? σ * (b - 1) / (κ * (σ + κ * α)) : 0.0
        return (b, c, κ)
    end
    G(pc) = ((b, c, κ) = bc(pc, h); h * (2 / κ + c + c * (1 + b) * (1 + b / λ) / (1 - b^2 / λ)) / (1 - 1 / λ))
    S(pn) = ((b, c, κ) = bc(pn, dt); 1 / κ + λ * c / (λ - b))
    j0 = geo.ny - geo.cells - 3
    j1 = geo.ny + geo.cells + 3
    if pol === :TE
        coefs = j -> begin
            pn = 2j
            s = dt * S(pn) / dy^2
            cp, cm = -s * G(pn + 1), -s * G(pn - 1)
            (cp, (λ - 1) + dt^2 * κx^2 / (1 - 1 / λ) - cp - cm, cm)
        end
    else
        coefs = j -> begin
            pc = 2j + 1
            s = dt * G(pc) / (dy^2 * (λ - 1))
            cp, cm = -s * S(pc + 1), -s * S(pc - 1)
            (cp, 1 + dt^2 * κx^2 / ((1 - 1 / λ) * (λ - 1)) - cp - cm, cm)
        end
    end
    cp, c0, cm = coefs(j0)
    disc = sqrt(c0^2 - 4 * cp * cm)
    μs = ((-c0 + disc) / (2cp), (-c0 - disc) / (2cp))
    μu, μd = angle(μs[1]) > 0 ? μs : (μs[2], μs[1])
    Fp, F = μu, one(ComplexF64)
    for j in j1:-1:j0
        cp, c0, cm = coefs(j)
        Fm = -(cp * Fp + c0 * F) / cm
        Fp, F = F, Fm
    end
    b = (F - Fp / μu) / (1 / μd - 1 / μu)
    a = Fp - b
    return (b / a, 1 / a)
end

"The energy fractions the PML's discrete theory predicts for the slab packet, weighted as `slab_prediction`."
function pml_prediction(geo, pml, pol)
    num_r = num_t = den = 0.0
    for (ky, A) in slab_modes(geo)
        r, t = pml_rt(geo, pml, pol, ky)
        w = abs2(A) * (1 + cos(yee_omega(geo, ky) * geo.dt / 2)^2)
        num_r += w * abs2(r)
        num_t += w * abs2(t)
        den += w
    end
    return (; R = num_r / den, Tr = num_t / den)
end

"""
    pml_step_digests(; workgroupsize = nothing, be = CPU()) -> Vector{(T, digest)}

T-545: a Reference run of `open_plasma_2d` with a PML of 4 cells (κ, α active; 20 steps, particles
retired at both edges) per element type, one digest of every particle array, field, history and
convolution field each.
"""
function pml_step_digests(; workgroupsize = nothing, be = CPU())
    out = Tuple{DataType,UInt64}[]
    for T in TYPES
        prob, _ = open_plasma_2d(T; nsteps = 20)
        st = open_state(prob, PML(; cells = 4, kappa_max = 2, alpha_max = 0.1); workgroupsize, be)
        run!(st, params(prob), prob.nsteps)
        L = st.aux.layer
        h = state_digest_2d(st)
        for a in (L.ψBx, L.ψBz, L.ψEx, L.ψEz)
            h = hash(bits(a), h)
        end
        push!(out, (T, h))
    end
    return out
end
