# T-212: the discrete Gauss law after every step, from the cumsum_periodic! initial field onwards.
#
# The initial Ex is the zero-mean periodic prefix sum of ρ⁰, so D⁻Ex⁰ = ρ⁰ − mean ρ⁰ to roundoff
# (T-172). Each step changes Ex by −Δt·Jx only, and deposit_flux! makes
# (ρⁿ⁺¹ − ρⁿ)/Δt + D⁻Jx = 0 to roundoff (T-110), so D⁻Exⁿ⁺¹ = ρⁿ⁺¹ − mean ρ (the mean is
# conserved, Σ D⁻Jx = 0 on a periodic grid). The residual is not re-solved, so the roundoff
# of each step adds up: the scale is n·max|ρ|·√k after k steps (a random walk of per-step
# errors of order eps·n·max|ρ|, since Ex is a prefix sum over n cells); C is measured.

const GAUSS_STEPS = 2   # t_end in 1/ω₀: 214 Landau steps, 263 two-stream steps

function body_T212(be)
    C = emC("T-212")
    for name in ("landau", "two-stream"), mode in (Reference(), Fast()), T in TYPES
        amplitude = T === D3 ? seeded(T, name == "landau" ? 0.01 : 0.001) : nothing
        st, p, prob, _ = case_state(name, T; mode, ppc = 64, t_end = GAUSS_STEPS, amplitude, be)
        r0 = gauss_residual(st)
        n = ncells(st.grid)
        worst = r0.res / (base_eps(T) * n * r0.scale)
        for k in 1:prob.nsteps
            step!(st, p, k)
            r = gauss_residual(st)
            worst = max(worst, r.res / (base_eps(T) * n * r.scale * sqrt(k)))
        end
        record!("T-212 $name", T, worst, be)
        @test worst <= C
    end
end

@testset "T-212 discrete Gauss law after every step [A2, A7, A8, WP-014]" begin
    body_T212(CPU())
end
