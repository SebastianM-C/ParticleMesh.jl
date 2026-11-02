# Diagnostics: the module's physical quantities, recorded once per step inside the step by one
# work item (O(n), fixed order, so deterministic), and read back on the host after the run.

@inline function energy_sum(a, g::Int32, n::Int32, ::Type{T}) where {T}
    w = zero(T)
    @inbounds for j in Int32(1):n
        w += a[g + j] * a[g + j]
    end
    return w
end

# Fourier amplitude 2/n·Σ a[j]·e^{−ik xⱼ} of a field whose index j sits at xⱼ = (j − 1 + x0)Δx.
@inline function mode_sum(a, g::Int32, n::Int32, kdx, x0, ::Type{T}) where {T}
    re = zero(T)
    im = zero(T)
    @inbounds for j in Int32(1):n
        φ = kdx * (T(j - Int32(1)) + x0)
        re += a[g + j] * cos(φ)
        im -= a[g + j] * sin(φ)
    end
    return re * 2 / T(n), im * 2 / T(n)
end

@kernel function diagnostics_kernel!(h, f, @Const(K), dx, kmode, n::Int32, g::Int32, len::Int32, k::Int32)
    T = eltype(K)
    half = one(T) / 2
    @inbounds begin
        h.WEx[k] = energy_sum(f.Ex, g, n, T) * dx / 2
        h.WEy[k] = energy_sum(f.Ey, g, n, T) * dx / 2
        h.WEz[k] = energy_sum(f.Ez, g, n, T) * dx / 2
        h.WBx[k] = energy_sum(f.Bx, g, n, T) * dx / 2
        h.WBy[k] = energy_sum(f.By, g, n, T) * dx / 2
        h.WBz[k] = energy_sum(f.Bz, g, n, T) * dx / 2
        # The kinetic energy density was deposited with guards and not folded: sum every entry.
        kin = zero(T)
        for j in Int32(1):len
            kin += K[j]
        end
        h.K[k] = kin * dx
        kdx = kmode * dx
        h.Ex_re[k], h.Ex_im[k] = mode_sum(f.Ex, g, n, kdx, half, T)
        h.Ey_re[k], h.Ey_im[k] = mode_sum(f.Ey, g, n, kdx, zero(T), T)
        h.Bz_re[k], h.Bz_im[k] = mode_sum(f.Bz, g, n, kdx, half, T)
    end
end

# --- N = 2: parallel reductions in a fixed association order -------------------------------

"""
History keys of a 2D state: the 1D keys, the Fourier amplitude of the longitudinal projection
k̂·E, those of Bx and By, and that of the in-plane projection ê·B, ê = ẑ × k̂ = (−k̂_y, k̂_x).
"""
const HISTORY_KEYS_2D = (HISTORY_KEYS..., :El_re, :El_im, :Bx_re, :Bx_im, :By_re, :By_im, :Bt_re, :Bt_im)
"Rows of the 2D diagnostics' partial sums: six energies and the real and imaginary sums of Ex, Ey, Bz, Bx, By."
const DIAG_ROWS = 16

# e^{−i(θ₁ + θ₂)} from the per-axis tables: (cos θ₁ cos θ₂ − sin θ₁ sin θ₂, sin θ₁ cos θ₂ + cos θ₁ sin θ₂).
@inline phase(c1, s1, c2, s2) = (c1 * c2 - s1 * s2, s1 * c2 + c1 * s2)

# Stage 1: one work item per index j₁ along axis 1 sums its line along axis 2 serially, from
# zero, in increasing j₂: the squares of the six components and the Fourier sums of Ex, Ey, Bz,
# Bx and By, each with the phases of its own stagger (Ex and By at (Center, Node), Ey and Bx at
# (Node, Center), Bz at (Center, Center)). Adjacent work items read adjacent memory.
@kernel function diagnostics_rows_2d_kernel!(P, f, @Const(t1), @Const(t2), n2::Int32, g::Int32)
    j1 = Int32(@index(Global, Linear))
    T = eltype(P)
    i1 = g + j1
    @inbounds begin
        c1N, s1N, c1C, s1C = t1[j1, 1], t1[j1, 2], t1[j1, 3], t1[j1, 4]
        wEx = wEy = wEz = wBx = wBy = wBz = zero(T)
        exr = exi = eyr = eyi = bzr = bzi = bxr = bxi = byr = byi = zero(T)
        for j2 in Int32(1):n2
            i2 = g + j2
            c2N, s2N, c2C, s2C = t2[j2, 1], t2[j2, 2], t2[j2, 3], t2[j2, 4]
            ex, ey, ez = f.Ex[i1, i2], f.Ey[i1, i2], f.Ez[i1, i2]
            bx, by, bz = f.Bx[i1, i2], f.By[i1, i2], f.Bz[i1, i2]
            wEx += ex * ex
            wEy += ey * ey
            wEz += ez * ez
            wBx += bx * bx
            wBy += by * by
            wBz += bz * bz
            cs, sn = phase(c1C, s1C, c2N, s2N)
            exr += ex * cs
            exi -= ex * sn
            byr += by * cs
            byi -= by * sn
            cs, sn = phase(c1N, s1N, c2C, s2C)
            eyr += ey * cs
            eyi -= ey * sn
            bxr += bx * cs
            bxi -= bx * sn
            cs, sn = phase(c1C, s1C, c2C, s2C)
            bzr += bz * cs
            bzi -= bz * sn
        end
        P[1, j1], P[2, j1], P[3, j1], P[4, j1], P[5, j1], P[6, j1] = wEx, wEy, wEz, wBx, wBy, wBz
        P[7, j1], P[8, j1], P[9, j1], P[10, j1], P[11, j1], P[12, j1] = exr, exi, eyr, eyi, bzr, bzi
        P[13, j1], P[14, j1], P[15, j1], P[16, j1] = bxr, bxi, byr, byi
    end
end

@inline function row_sum(P, q::Int, n1::Int32, ::Type{T}) where {T}
    acc = zero(T)
    @inbounds for j in Int32(1):n1
        acc += P[q, j]
    end
    return acc
end

# Stage 2: one work item adds the rows' partial sums in increasing j₁ and the kinetic-energy
# chunks in order, and writes entry k of the histories.
@kernel function diagnostics_total_2d_kernel!(h, @Const(P), @Const(K), n1::Int32, nK::Int32, dA, amp, khat, k::Int32)
    T = eltype(K)
    @inbounds begin
        h.WEx[k] = row_sum(P, 1, n1, T) * dA / 2
        h.WEy[k] = row_sum(P, 2, n1, T) * dA / 2
        h.WEz[k] = row_sum(P, 3, n1, T) * dA / 2
        h.WBx[k] = row_sum(P, 4, n1, T) * dA / 2
        h.WBy[k] = row_sum(P, 5, n1, T) * dA / 2
        h.WBz[k] = row_sum(P, 6, n1, T) * dA / 2
        kin = zero(T)
        for j in Int32(1):nK
            kin += K[j]
        end
        h.K[k] = kin * dA
        exr, exi = row_sum(P, 7, n1, T) * amp, row_sum(P, 8, n1, T) * amp
        eyr, eyi = row_sum(P, 9, n1, T) * amp, row_sum(P, 10, n1, T) * amp
        h.Ex_re[k], h.Ex_im[k] = exr, exi
        h.Ey_re[k], h.Ey_im[k] = eyr, eyi
        h.Bz_re[k], h.Bz_im[k] = row_sum(P, 11, n1, T) * amp, row_sum(P, 12, n1, T) * amp
        h.El_re[k] = khat[1] * exr + khat[2] * eyr
        h.El_im[k] = khat[1] * exi + khat[2] * eyi
        bxr, bxi = row_sum(P, 13, n1, T) * amp, row_sum(P, 14, n1, T) * amp
        byr, byi = row_sum(P, 15, n1, T) * amp, row_sum(P, 16, n1, T) * amp
        h.Bx_re[k], h.Bx_im[k] = bxr, bxi
        h.By_re[k], h.By_im[k] = byr, byi
        h.Bt_re[k] = khat[1] * byr - khat[2] * bxr
        h.Bt_im[k] = khat[1] * byi - khat[2] * bxi
    end
end

"""
    record_diagnostics!(state::EMState{2}, k)

Entry `k` of the 2D histories from the current fields and the kinetic-energy chunks: the field
energies ½Σ(component)²·ΔxΔy (per unit length along z), the kinetic energy Σ m·w·(γ − 1)·ΔxΔy,
and at the state's diagnostic mode the Fourier amplitude 2/(n₁n₂)·Σ a·e^{−ik·x} of Ex, Ey, Bz,
Bx and By, each over the positions of its own stagger, of the longitudinal projection
k̂ₓÊₓ + k̂_yÊ_y and of the in-plane magnetic projection on ê = ẑ × k̂, k̂ₓB̂_y − k̂_yB̂ₓ (the 2D
Weibel mode).
Two launches with a fixed association order (stage 1: O(n₁) work items of O(n₂) serial work;
stage 2: one work item of O(n₁ + chunks)), so the bits do not depend on the thread count or the
launch shape; the phases come from the per-axis tables of the state (no trigonometric
evaluation per step).
"""
function record_diagnostics!(st::EMState{2}, k::Integer)
    (; grid, aux) = st
    T = eltype(st.ρ)
    n = ncells(grid)
    dx = cellsize(grid)
    be = get_backend(st.ρ)
    launcher(diagnostics_rows_2d_kernel!, be, st.numerics.workgroupsize)(aux.partials, st.fields, aux.tables[1], aux.tables[2],
                                                                        Int32(n[2]), Int32(guards(grid)); ndrange = n[1])
    diagnostics_total_2d_kernel!(be)(st.hist, aux.partials, st.K, Int32(n[1]), Int32(length(st.K)), dx[1] * dx[2],
                                     2 / T(n[1] * n[2]), aux.khat, Int32(k); ndrange = 1)
    return st
end

"""
    histories(state) -> NamedTuple

Host copies of the per-step diagnostics, entry `k` recorded at the start of step `k`, at
t = (k − 1)Δt: the field energies `WEx … WBz` = ½Σ(component)²Δx of Eⁿ and Bⁿ (in mₑc²·n_c·c/ω₀,
A1), the kinetic energy `K` = Σ m·w·(γ − 1)·Δx at t − Δt/2, and the real and imaginary parts of the
Fourier amplitude at `kmode` of Ex, Ey and Bz (each at its own stagger). A 2D state also records
`El_re`, `El_im`, the amplitude of the longitudinal projection k̂·E, `Bx_*`, `By_*` and
`Bt_re`, `Bt_im`, the amplitude of ê·B with ê = ẑ × k̂, and sums its energies over ΔxΔy
([`record_diagnostics!`](@ref)).
"""
histories(st::EMState) = map(Array, st.hist)

"Field energy per step: the sum of the six component energies."
field_energy(h::NamedTuple) = h.WEx .+ h.WEy .+ h.WEz .+ h.WBx .+ h.WBy .+ h.WBz

"""
    kinetic_energy_history(h) -> Vector

The kinetic energy at integer times n = 0 … N − 2 (steps 1 … N − 1), as the mean of the values at
n − 1/2 and n + 1/2 recorded by steps n + 1 and n + 2.
"""
kinetic_energy_history(h::NamedTuple) = (h.K[1:(end - 1)] .+ h.K[2:end]) ./ 2

"Total energy (fields plus kinetic) at integer times, steps 1 … N − 1."
total_energy_history(h::NamedTuple) = field_energy(h)[1:(end - 1)] .+ kinetic_energy_history(h)

"Amplitude history |â| of the mode at `kmode` of `:Ex`, `:Ey` or `:Bz` (at N = 2 also `:El`, `:Bx`, `:By`, `:Bt`)."
mode_amplitude_history(h::NamedTuple, c::Symbol) = hypot.(getfield(h, Symbol(c, :_re)), getfield(h, Symbol(c, :_im)))
