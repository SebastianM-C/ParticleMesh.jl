# WP-079: the M4 solver's discrete theory, written from the formulas of Lu et al. (2020) and not
# from the kernels, for m4.jl and m4_step.jl. After yee_nd.jl's helpers (bigΔ, pshift, the mode
# fields) and antenna_cases.jl.

"""
    m4_coefficients(Δ, dt) -> (δ, β)

M4's coefficients for the cell sizes `Δ` and the step `dt` in BigFloat (Lu et al. 2020, appendix
C; Smilei's MF_Solver2D_M4): δ_d = ((Δt/Δ_d)² − 1)/12 and β_dj = (Δt/Δ_j)²/12 for j ≠ d (zero on
the diagonal).
"""
function m4_coefficients(Δ, dt)
    bdt = big(Float64(value_of(dt)))
    N = length(Δ)
    δ = [((bdt / Δ[d])^2 - 1) / 12 for d in 1:N]
    β = [j == d ? big(0) : (bdt / Δ[j])^2 / 12 for d in 1:N, j in 1:N]
    return δ, β
end

"M4's symbol A_d of the difference along d at the angles θ_j = k_jΔ_j, so that D̃_d = D⁺_d·A_d on a mode."
m4_symbol(δ, β, θ, d) = 1 - 2 * δ[d] * (1 - cos(θ[d])) - sum(2 * β[d, j] * (1 - cos(θ[j])) for j in eachindex(θ); init = big(0))

"""
    m4_mode(grid, m, pol, dt) -> (; m, Ê, B̂, ω, κ)

The discrete eigenmode of the M4 scheme of integer wave numbers `m`, in the shape of `yee_mode`:
with κ_d = 2 sin(k_dΔ_d/2)/Δ_d the Yee symbol and κ̃_d = κ_d·A_d M4's, the leapfrog gives
Ω·B̂½ = κ̃×Ê and −Ω·Ê = κ×B̂½, Ω = 2 sin(ωΔt/2)/Δt, so for κ·Ê = 0 (Gauss's law)
κ×(κ̃×Ê) = −(κ·κ̃)·Ê and Ω² = κ·κ̃ = Σ_d κ_d²A_d: every Ê across κ is a mode. Ê is `yee_mode`'s
(polarization 1 in the x–y plane, 2 the other), B̂½ = κ̃×Ê/Ω and the integer-time B̂ = cos(ωΔt/2)·B̂½.
"""
function m4_mode(grid, m, pol, dt)
    N = length(grid.n)
    Δ = bigΔ(grid)
    θ = [2 * big(π) * m[d] / grid.n[d] for d in 1:N]
    κ = ntuple(d -> 2 * sin(θ[d] / 2) / Δ[d], N)
    δ, β = m4_coefficients(Δ, dt)
    κ̃ = ntuple(d -> κ[d] * m4_symbol(δ, β, θ, d), N)
    pad(v) = (v[1], N >= 2 ? v[2] : big(0), N >= 3 ? v[3] : big(0))
    k3, k̃3 = pad(κ), pad(κ̃)
    Ω = sqrt(sum(k3 .* k̃3))
    bdt = big(Float64(value_of(dt)))
    ω = 2 / bdt * asin(bdt * Ω / 2)
    nκ = sqrt(sum(abs2, k3))
    κp = sqrt(k3[1]^2 + k3[2]^2)
    o = big(0)
    inplane = (-k3[2] / κp, k3[1] / κp, o)
    other = ((o - k3[1] * k3[3]) / (κp * nκ), (o - k3[2] * k3[3]) / (κp * nκ), κp / nκ)
    Ê = pol == 1 ? inplane : other
    cross3(a, b) = (a[2] * b[3] - a[3] * b[2], a[3] * b[1] - a[1] * b[3], a[1] * b[2] - a[2] * b[1])
    B̂ = cos(ω * bdt / 2) .* cross3(k̃3, Ê) ./ Ω
    return (; m, Ê, B̂, ω, κ)
end

"""
    m4_diff_big(a, d, Δ, δ, β) -> array

M4's forward difference of the BigFloat interior `a` along d on the periodic grid, divided by
Δ_d: D⁺_d a + δ_d·Δ³_d a/Δ_d + Σ_{j≠d} β_dj·Δ²_j D⁺_d a, written with circular shifts.
"""
function m4_diff_big(a, d, Δ, δ, β)
    sh(x, ax, s) = pshift(x, ax, s)
    d1 = sh(a, d, -1) .- a                                   # a[i+1] − a[i]
    d3 = (sh(a, d, -2) .- sh(a, d, 1)) .- 3 .* d1            # third difference
    r = d1 .+ δ[d] .* d3
    for j in eachindex(Δ)
        j == d && continue
        r = r .+ β[d, j] .* (sh(d1, j, -1) .+ sh(d1, j, 1) .- 2 .* d1)
    end
    return r ./ Δ[d]
end

"The divergence Σ_c D̃_c B_c that M4 keeps (`m4.jl`), on the cell centres, of the BigFloat interiors `F`."
m4_div(F, Δ, δ, β) = sum(m4_diff_big(F[d], d, Δ, δ, β) for d in eachindex(Δ))

"""
    m4_kx(ω, ky, kz, dx, dy, dz, dt) -> complex k_x

The wavenumber along x of the M4 mode of frequency ω and transverse wavenumbers (k_y, k_z) on
the branch that travels or decays along +x. With u = sin²(k_xΔx/2) the relation
Σ_d κ_d²A_d = Ω² is the quadratic −(16δ_x/Δx²)u² + (4a/Δx² − 4β_yxκ_y² − 4β_zxκ_z²)u +
(κ_y²b + κ_z²c − Ω²) = 0, a, b, c the parts of A_x, A_y, A_z without u; its small root is the
branch of `yee_kx` (δ, β → 0). At N = 2, kz = 0 (dz is then unused).
"""
function m4_kx(ω, ky, kz, dx, dy, dz, dt)
    r2(h) = (dt / h)^2
    δx, δy, δz = (r2(dx) - 1) / 12, (r2(dy) - 1) / 12, (r2(dz) - 1) / 12
    βxy, βxz, βyx, βyz, βzx, βzy = r2(dy) / 12, r2(dz) / 12, r2(dx) / 12, r2(dz) / 12, r2(dx) / 12, r2(dy) / 12
    cy, cz = 1 - cos(ky * dy), 1 - cos(kz * dz)
    κy2, κz2 = (2 * sin(ky * dy / 2) / dy)^2, (2 * sin(kz * dz / 2) / dz)^2
    Ω2 = (2 * sin(ω * dt / 2) / dt)^2
    a = 1 - 2βxy * cy - 2βxz * cz
    b = 1 - 2δy * cy - 2βyz * cz
    c = 1 - 2δz * cz - 2βzy * cy
    A = -16δx / dx^2
    B = 4a / dx^2 - 4βyx * κy2 - 4βzx * κz2
    C = κy2 * b + κz2 * c - Ω2
    u = -2C / (B + sqrt(complex(B^2 - 4A * C)))
    kx = 2 / dx * asin(sqrt(complex(u)))
    return imag(kx) < 0 ? conj(kx) : kx
end
m4_kx(ω, ky, dx, dy, dt) = m4_kx(ω, ky, 0.0, dx, dy, 1.0, dt)
