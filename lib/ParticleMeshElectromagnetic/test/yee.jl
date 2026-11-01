# T-210: the Yee solver in vacuum against its exact discrete dispersion relation.
#
# In A1 units (c = 1, time 1/ω₀, length c/ω₀) the 1D Yee leapfrog on the layout of FIELD_LAYOUT
# admits the travelling mode Ey = A cos(kx − ωt), Bz = A cos(k(x + Δx/2) − ωt) exactly, with ω
# from sin(ωΔt/2)/Δt = sin(kΔx/2)/Δx (substitute e^{i(kx−ωt)} into the two curl updates). B is
# kept at integer times as the mean of its two half-step values, so Bzⁿ = A·cos(ωΔt/2)·cos(kx_d − ωtₙ)
# at the dual points x_d = xⱼ + Δx/2. Ez/By carries the mode travelling the other way. The
# leapfrog conserves W = ½Σ|Eⁿ|²Δx + ½Σ Bⁿ⁻¹ᐟ²·Bⁿ⁺¹ᐟ²Δx exactly in exact arithmetic, and with
# Bⁿ⁻¹ᐟ²·Bⁿ⁺¹ᐟ² = |Bⁿ|² − (Δt/2)²|D⁺Eⁿ|² it is computed from the integer-time fields.

"Discrete ω of the 1D Yee scheme (BigFloat)."
yee_omega(k, dx, dt) = 2 / big(dt) * asin(big(dt) / big(dx) * sin(big(k) * big(dx) / 2))

# The staggered energy in BigFloat from the fields' values (value parts for duals).
function staggered_energy(f, grid, dt)
    g, n, dx = guards(grid), ncells(grid), big(Float64(value_of(cellsize(grid))))
    v(a, j) = big(Float64(value_of(a[g + mod1(j, n)])))
    h = big(Float64(value_of(dt))) / 2
    W = big(0)
    for j in 1:n
        W += v(f.Ex, j)^2 + v(f.Ey, j)^2 + v(f.Ez, j)^2 + v(f.Bx, j)^2
        W += v(f.By, j)^2 - h^2 * ((v(f.Ez, j + 1) - v(f.Ez, j)) / dx)^2
        W += v(f.Bz, j)^2 - h^2 * ((v(f.Ey, j + 1) - v(f.Ey, j)) / dx)^2
    end
    return W * dx / 2
end

function body_T210(be)
    C = emC("T-210")
    Cw = emC("T-210", "C_energy")
    n, m = 64, 3
    for T in TYPES, cfl in (1 // 2, 9 // 10)
        B = basetype(T)
        L = 2 * B(π)
        grid = PeriodicGrid(n, L / n, 2)
        dx = cellsize(grid)
        dt = B(cfl) * dx
        k = 2 * B(π) * m / L
        ω = yee_omega(Float64(k), Float64(dx), Float64(dt))
        A = seeded(T, 1)                                    # partials exercised for duals
        f = em_fields(backend, T, grid)
        g = guards(grid)
        xn(j) = big(j - 1) * big(Float64(dx))
        xd(j) = xn(j) + big(Float64(dx)) / 2
        c½ = cos(ω * big(Float64(dt)) / 2)
        for j in 1:n                                         # host set-up of the initial mode
            f.Ey[g + j] = A * T(Float64(cos(big(Float64(k)) * xn(j))))
            f.Bz[g + j] = A * T(Float64(c½ * cos(big(Float64(k)) * xd(j))))
            f.Ez[g + j] = A * T(Float64(cos(big(Float64(k)) * xn(j))))
            f.By[g + j] = A * T(Float64(c½ * cos(big(Float64(k)) * xd(j))))    # the left-moving mode, By = +Ez
            f.Ex[g + j] = A * T(Float64(cos(big(Float64(k)) * xd(j))) / 4)    # static: no current
        end
        foreach(a -> on(be, fill_guards!, a, grid), f)
        Ex0 = copy(f.Ex)
        W0 = staggered_energy(f, grid, dt)
        nsteps = 400
        worst = 0.0
        worstW = 0.0
        for s in 1:nsteps
            on(be, solve_fields!, Yee(), f, grid, dt)
            if s % 50 == 0
                t = s * big(Float64(dt))
                err = 0.0
                for j in 1:n
                    φr = big(Float64(k)) * xn(j) - ω * t
                    φl = big(Float64(k)) * xn(j) + ω * t
                    err = max(err, absdiff(f.Ey[g + j], A * T(Float64(cos(φr)))),
                              absdiff(f.Bz[g + j], A * T(Float64(c½ * cos(big(Float64(k)) * xd(j) - ω * t)))),
                              absdiff(f.Ez[g + j], A * T(Float64(cos(φl)))),
                              absdiff(f.By[g + j], A * T(Float64(c½ * cos(big(Float64(k)) * xd(j) + ω * t)))))
                end
                ratio = err / (base_eps(T) * s)
                worst = max(worst, ratio)
                Wr = Float64(abs(staggered_energy(f, grid, dt) - W0) / W0) / (base_eps(T) * s)
                worstW = max(worstW, Wr)
            end
        end
        record!("T-210", T, worst, be)
        record!("T-210 energy", T, worstW, be)
        @test worst <= C
        @test worstW <= Cw
        @test same_bits(f.Ex, Ex0)                           # Ex has no curl term in 1D and no current here
        # The phase per step, from the Fourier coefficient of Ey: equal to ω·Δt of the discrete relation.
        ê = sum(Float64(value_of(f.Ey[g + j])) * cis(-Float64(k) * Float64(xn(j))) for j in 1:n)
        φ = angle(ê)                                         # = −ω·N·Δt (mod 2π) for the right-moving wave
        Δφ = mod(-φ - Float64(ω * nsteps * big(Float64(dt))) + π, 2π) - π
        @test abs(Δφ) <= C * base_eps(T) * nsteps * 4
    end
end

@testset "T-210 Yee in vacuum: exact discrete dispersion and staggered energy [A1, A2, A7, A8, WP-014]" begin
    body_T210(CPU())
end
