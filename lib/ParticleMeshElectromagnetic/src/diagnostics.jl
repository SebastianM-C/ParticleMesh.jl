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

"""
    histories(state) -> NamedTuple

Host copies of the per-step diagnostics, entry `k` recorded at the start of step `k`, at
t = (k − 1)Δt: the field energies `WEx … WBz` = ½Σ(component)²Δx of Eⁿ and Bⁿ (in mₑc²·n_c·c/ω₀,
A1), the kinetic energy `K` = Σ m·w·(γ − 1)·Δx at t − Δt/2, and the real and imaginary parts of the
Fourier amplitude at `kmode` of Ex, Ey and Bz (each at its own stagger).
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

"Amplitude history |â| of the mode at `kmode` of `:Ex`, `:Ey` or `:Bz`."
mode_amplitude_history(h::NamedTuple, c::Symbol) = hypot.(getfield(h, Symbol(c, :_re)), getfield(h, Symbol(c, :_im)))
