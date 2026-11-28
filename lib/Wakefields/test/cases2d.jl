# The 2D optimization cases of WP-064 and WP-065 (src/cases2d.jl), their builders, objectives and
# tables; these tests moved here with the cases from scripts/optim/test (WP-092).
using ForwardDiff: ForwardDiff
using ParticleMeshCore
using ParticleMeshCore: ALIVE
using ParticleMeshElectromagnetic
using Wakefields: BCASE, TCASE, TRAPPED_BOX, budget, start_of, run_to_end, check_fade, check_rise, check_from,
                  slot_energy, slot_trapped, species_trapped

# Tests of the WP-064 case: the objective on ionization-born electrons. T-833 and
# T-834 run the case at λ₀/8 to t = 190, about a minute in all.

"The case's state at λ₀/8 before its first step, with `n` hand-made born electrons written into the first slots."
function born_test_state(xs, us, ws)
    st, p, prob = born_start(; ppl = 8, t_end = 1.0)
    e = ionization_electrons(st)
    h = cellsize(st.grid)[1]
    for (i, (x, u, w)) in enumerate(zip(xs, us, ws))
        c = floor(Int, x / h)
        e.cell[1][i] = c + 1; e.offset[1][i] = x / h - c
        e.cell[2][i] = 60 + i; e.offset[2][i] = 0.5
        e.u[1][i], e.u[2][i], e.u[3][i] = u
        e.w[i] = w; e.status[i] = ALIVE
    end
    return st
end

@testset "T-830 the position weight is a C² step: 0 and 1 outside, antisymmetric about the middle, flat to second order at both ends [A2, WP-064]" begin
    lo, hi = 12.0, 24.0
    @test smooth_rise(lo, lo, hi) == 0 && smooth_rise(-3.0, lo, hi) == 0 && smooth_rise(hi, lo, hi) == 1 && smooth_rise(99.0, lo, hi) == 1
    @test smooth_rise(18.0, lo, hi) == 0.5
    xs = range(lo, hi; length = 97)
    @test all(x -> isapprox(smooth_rise(x, lo, hi) + smooth_rise(lo + hi - x, lo, hi), 1; atol = 4eps()), xs)
    @test issorted(smooth_rise.(xs, lo, hi))
    d1(x) = ForwardDiff.derivative(y -> smooth_rise(y, lo, hi), x)
    d2(x) = ForwardDiff.derivative(d1, x)
    @test d1(lo) == 0 && d1(hi) == 0 && d2(lo) == 0 && d2(hi) == 0
    # just inside the ends the second derivative is O(distance): 60·s/(hi − lo)² to first order
    @test d2(lo + 1e-6) ≈ 60 * (1e-6 / 12) / 144 rtol = 1e-4
    @test d2(hi - 1e-6) ≈ -60 * (1e-6 / 12) / 144 rtol = 1e-4
    @test d1(18.0) ≈ 15 / 8 / 12 rtol = 4eps()                       # the steepest slope, 15/8 per width
    @test smooth_rise(15.0f0, 12.0f0, 24.0f0) isa Float32
    @test smooth_rise(15.0f0, 12.0f0, 24.0f0) ≈ 0.25^3 * (10 - 15 * 0.25 + 6 * 0.25^2) rtol = 4eps(Float32)
end

@testset "T-831 the objective is Δx·Δy·Σ w(γ − 1)S(x) over the born electrons alone [A3, WP-064]" begin
    xs = [5.0, 12.0, 15.0, 18.0, 30.0, 60.0]
    us = [(3.0, 0.0, 0.0), (2.0, 1.0, 0.0), (1.0, -2.0, 0.5), (0.0, 0.0, 0.0), (10.0, 0.3, -0.2), (0.4, 0.0, 0.0)]
    ws = [1.0e-3, 2.0e-3, 5.0e-4, 7.0e-4, 3.0e-4, 1.1e-3]
    st = born_test_state(xs, us, ws)
    h = cellsize(st.grid)
    @test h[1] ≈ 2π / 8 && h[2] ≈ π / 4
    S(x) = (s = clamp((x - 12) / 12, 0, 1); s^3 * (10 - 15s + 6s^2))
    expected = h[1] * h[2] * sum(w * (sqrt(1 + sum(abs2, u)) - 1) * S(x) for (x, u, w) in zip(xs, us, ws))
    J = born_energy(st)
    @test J ≈ expected rtol = 8eps()
    @test J > 0
    # the electrons behind fade[1] (x = 5 and 12) count for nothing: without them the same bits
    e = ionization_electrons(st)
    e.status[1] = 0x00; e.status[2] = 0x00
    @test born_energy(st) === J
    e.status[1] = ALIVE; e.status[2] = ALIVE
    # a free slot's leftover numbers do not enter
    e.u[1][7] = 1.0e6; e.w[7] = 1.0; e.cell[1][7] = 40
    @test born_energy(st) === J
    # the plasma's electrons do not enter, whatever their energy (a label, not an energy cut)
    s = st.species[1]
    s.u[1] .= 50.0
    @test born_energy(st) === J
    # another weight: with fade = (0.9·x, x) of the electron at 30 only it and the one at 60 count, in full
    @test born_energy(st; fade = (27.0, 30.0)) ≈ h[1] * h[2] * (ws[5] * (sqrt(1 + sum(abs2, us[5])) - 1) + ws[6] * (sqrt(1 + 0.16) - 1)) rtol = 8eps()
    # a weight that reaches into the window's back band, or runs backwards, is refused
    band = window_back_band(st) * h[1]
    @test band ≈ 9 * 2π / 8
    @test_throws ArgumentError born_energy(st; fade = (band - 0.01, 24.0))
    @test_throws ArgumentError born_energy(st; fade = (24.0, 12.0))
    @test_throws ArgumentError born_energy(st; fade = (24.0, 24.0))
    @test born_energy(st; fade = (band, 24.0)) > J
    # duals in the momenta and positions pass through: ∂J/∂u_x of one electron is Δx·Δy·w·S·u_x/γ
    g5 = sqrt(1 + sum(abs2, us[5]))
    dJ = ForwardDiff.derivative(0.0) do t
        slot_energy(ALIVE, Int32(1), t + 30.0, 10.0 + t, 0.3, -0.2, ws[5], 1.0, 12.0, 24.0)
    end
    @test dJ ≈ ws[5] * 10.0 / g5 rtol = 8eps()                                    # S = 1 there: only the momentum's part
    dS = ForwardDiff.derivative(t -> slot_energy(ALIVE, Int32(1), 18.0 + t, 10.0, 0.3, -0.2, ws[5], 1.0, 12.0, 24.0), 0.0)
    @test dS ≈ ws[5] * (g5 - 1) * 15 / 8 / 12 rtol = 8eps()                       # mid-fade: only the weight's slope
end

@testset "T-832 the guard reads the plasma's weighted electrons ahead of a chosen place, not the born ones [WP-064]" begin
    st = born_test_state([30.0], [(40.0, 0.0, 0.0)], [1.0e-3])
    h = cellsize(st.grid)
    @test plasma_peak_energy(st) == 0 && plasma_fast_weight(st) == 0            # a cold plasma; the born electron at γ ≈ 40 is not read
    s = st.species[1]
    col(i) = (s.cell[1][i] - 1 + s.offset[1][i]) * h[1]
    weighted = findall(i -> s.status[i] == ALIVE && s.w[i] > 0 && col(i) >= 12.0, eachindex(s.w))
    empty = findall(i -> s.status[i] == ALIVE && s.w[i] == 0 && col(i) >= 12.0, eachindex(s.w))
    behind = findall(i -> s.status[i] == ALIVE && s.w[i] > 0 && col(i) < 12.0, eachindex(s.w))
    @test !isempty(empty)                                                       # the vacuum's particles have the weight zero
    if isempty(weighted) || isempty(behind)                                     # at the start the plasma is ahead of the window: place two
        i, j = empty[1], empty[2]
        s.w[i] = 2.0e-3; s.w[j] = 4.0e-3
        s.cell[1][j] = 3; s.offset[1][j] = 0.5                                  # x ≈ 1.96, behind the fade
        weighted, behind = [i], [j]
        empty = empty[3:end]
    end
    i, j, k = weighted[1], behind[1], empty[1]
    s.u[1][k] = 30.0                                                            # a fast particle of weight zero
    s.u[1][j] = 20.0                                                            # a fast one behind the fade
    @test plasma_peak_energy(st) == 0 && plasma_fast_weight(st) == 0
    s.u[1][i] = 3.0; s.u[2][i] = 4.0; s.u[3][i] = 1.0                           # γ − 1 = √27 − 1
    @test plasma_peak_energy(st) ≈ sqrt(27) - 1 rtol = 4eps()
    @test plasma_fast_weight(st) == 0                                           # below the 5 mₑc² of the tables
    @test plasma_fast_weight(st; energy = 4.0) ≈ h[1] * h[2] * s.w[i] rtol = 4eps()
    @test plasma_fast_weight(st; energy = 4.19) > 0 && plasma_fast_weight(st; energy = 4.2) == 0   # √27 − 1 = 4.196: all three momenta count
    # the fade's edge to within a cell: the same particle 0.3 cells behind fade[1] is not read, 0.3 cells ahead it is
    c0, o0 = s.cell[1][i], s.offset[1][i]
    s.cell[1][i] = floor(Int, 12.0 / h[1]) + 1; s.offset[1][i] = 12.0 / h[1] - floor(12.0 / h[1]) - 0.3
    @test plasma_peak_energy(st) == 0 && plasma_fast_weight(st; energy = 4.0) == 0
    s.offset[1][i] += 0.6
    @test plasma_peak_energy(st) ≈ sqrt(27) - 1 rtol = 4eps()
    @test plasma_fast_weight(st; energy = 4.0) ≈ h[1] * h[2] * s.w[i] rtol = 4eps()
    s.cell[1][i] = c0; s.offset[1][i] = o0
    # a retired particle is not read
    s.status[i] = 0x00
    @test plasma_peak_energy(st) == 0 && plasma_fast_weight(st; energy = 4.0) == 0
    s.status[i] = ALIVE
    @test_throws ArgumentError plasma_fast_weight(st; from = 1.0)
    @test plasma_peak_energy(st; from = 8.0) == plasma_peak_energy(st)           # x ≈ 1.96 is still behind
    @test_throws ArgumentError plasma_peak_energy(st; from = 1.0)               # inside the back band
    @test_throws ArgumentError plasma_fast_energy(st; from = 1.0)
    # from the back band's end the whole window is read: the fast particle behind the fade too (u_x = 20)
    band = back_band_end(st)
    s.cell[1][j] = floor(Int, band / h[1]) + 2; s.offset[1][j] = 0.5
    @test plasma_peak_energy(st) ≈ sqrt(27) - 1 rtol = 4eps()
    @test plasma_peak_energy(st; from = band) ≈ sqrt(401) - 1 rtol = 4eps()
    @test plasma_fast_weight(st; from = band) ≈ h[1] * h[2] * s.w[j] rtol = 4eps()
    @test plasma_fast_energy(st; from = band) ≈ h[1] * h[2] * s.w[j] * (sqrt(401) - 1) rtol = 4eps()
    @test plasma_fast_energy(st; from = band, energy = 4.0) ≈ h[1] * h[2] * (s.w[j] * (sqrt(401) - 1) + s.w[i] * (sqrt(27) - 1)) rtol = 4eps()
    @test plasma_fast_energy(st) == 0
end

"The state of the case at λ₀/8 at t = 190."
born_small_state(a0, dopant = BCASE.dopant; kw...) = born_state(a0, dopant; ppl = 8, t_end = 190.0, capacity = 60_000, kw...)
"The objective of the case at λ₀/8 read at t = 190."
born_small(a0, dopant = BCASE.dopant; kw...) = born_objective(a0, dopant; ppl = 8, t_end = 190.0, capacity = 60_000, kw...)
# At t = 190 the born electrons of this run lie at window x = 27 to 53: the case's weight is 1 on
# all of them, and this one cuts through them.
const ACROSS = (30.0, 50.0)

@testset "T-833 ForwardDiff through the run gives the objective's slope in a₀ and in the dopant density, with the position weight across the electrons too [A3, A7, WP-064]" begin
    a0 = 2.2
    std = born_small_state(ForwardDiff.Dual{Nothing}(a0, 1.0, 0.0), ForwardDiff.Dual{Nothing}(BCASE.dopant, 0.0, 1.0))
    st = born_small_state(a0)
    ha = a0 * 2.0^-12
    sta = (born_small_state(a0 + ha), born_small_state(a0 - ha))
    for fade in (BCASE.fade, ACROSS)
        Jd, J = born_energy(std; fade), born_energy(st; fade)
        @test J > 1.0e-4                                                        # the K shell ionizes and the electrons gain energy
        # the duals do not change the run, to roundoff: inside a kernel ForwardDiff's sin of a dual
        # calls sincos, which differs from the plain laser's sin and cos in the last bits (WP-083)
        @test ForwardDiff.value(Jd) ≈ J rtol = 64eps()
        fd_a = (born_energy(sta[1]; fade) - born_energy(sta[2]; fade)) / 2ha
        @test ForwardDiff.partials(Jd)[1] ≈ fd_a rtol = 1.0e-5
        @test ForwardDiff.partials(Jd)[1] > 0
    end
    @test 0.1 * born_energy(st) < born_energy(st; fade = ACROSS) < 0.5 * born_energy(st)   # the second weight does cut (0.22 measured)
    Jd, J = born_energy(std), born_energy(st)
    ad = ForwardDiff.partials(Jd)
    hd = BCASE.dopant * 2.0^-8
    fd_d = (born_small(a0, BCASE.dopant + hd) - born_small(a0, BCASE.dopant - hd)) / 2hd
    @test ad[2] ≈ fd_d rtol = 1.0e-5
    # the dopant is 1 % of the plasma: the born electrons barely load the wake, so J is close to
    # proportional to their density, and not exactly
    @test 0.9 < ad[2] * BCASE.dopant / J < 1.1
    @test ad[2] * BCASE.dopant != J
end

@testset "T-834 the run's objective repeats bit for bit, its keywords reach it, and a full buffer is an error [A4, WP-064]" begin
    st = born_small_state(2.2)
    s = born_summary(st)
    @test s.energy === born_energy(st) === born_small(2.2)                      # a second run: the same bits in J
    @test born_small(2.2; fade = ACROSS) === born_energy(st; fade = ACROSS) < s.energy
    @test s.born ≈ s.alive rtol = 1.0e-12                                       # at t = 190 no born electron has left the window
    @test s.particles == count(==(ALIVE), ionization_electrons(st).status) > 4000
    @test s.plasma_peak == plasma_peak_energy(st) > 0
    @test_throws ErrorException born_state(2.2; ppl = 8, t_end = 190.0, capacity = 1000)   # the run needs more than 4000 slots
end

@testset "T-835 the summary's weights on hand-placed electrons: alive, counted, fast, and the plasma's [WP-064]" begin
    # T-831's electrons, and one fast electron in the middle of the fade (S = 1/2)
    xs = [5.0, 12.0, 15.0, 18.0, 30.0, 60.0, 18.0]
    us = [(3.0, 0.0, 0.0), (2.0, 1.0, 0.0), (1.0, -2.0, 0.5), (0.0, 0.0, 0.0), (10.0, 0.3, -0.2), (0.4, 9.0, 0.0), (7.0, 0.0, 0.0)]
    ws = [1.0e-3, 2.0e-3, 5.0e-4, 7.0e-4, 3.0e-4, 1.1e-3, 4.0e-4]
    st = born_test_state(xs, us, ws)
    h = cellsize(st.grid)
    area = h[1] * h[2]
    S(x) = (q = clamp((x - 12) / 12, 0, 1); q^3 * (10 - 15q + 6q^2))
    s = born_summary(st)
    @test s.particles == 7
    @test s.born == 0                                                           # the atoms have released nothing: these were placed by hand
    @test s.alive ≈ area * sum(ws) rtol = 8eps()
    @test s.counted ≈ area * sum(ws .* S.(xs)) rtol = 8eps()
    @test s.fast ≈ area * (ws[5] + ws[7]) rtol = 8eps()                         # u_x > 5: not the one with u_y = 9
    @test s.fast_counted ≈ area * (ws[5] + ws[7] / 2) rtol = 8eps()
    @test s.alive > s.counted > s.fast > s.fast_counted > 0
    @test s.u_max == 10.0
    @test s.energy === born_energy(st)
    @test s.mean_energy ≈ s.energy / s.counted rtol = 4eps()
    @test born_summary(st; u_fast = 0.5).fast ≈ area * (sum(ws) - ws[4] - ws[6]) rtol = 8eps()
    @test born_summary(st; fade = (27.0, 30.0)).counted ≈ area * (ws[5] + ws[6]) rtol = 8eps()
    # the plasma's part: one weighted fast particle ahead of the fade
    p = st.species[1]
    i = findfirst(j -> p.status[j] == ALIVE && (p.cell[1][j] - 1 + p.offset[1][j]) * h[1] >= 12.0, eachindex(p.w))
    p.w[i] = 2.0e-3; p.u[1][i] = 8.0
    s = born_summary(st)
    @test s.plasma_peak ≈ sqrt(65) - 1 rtol = 4eps()
    @test s.plasma_fast ≈ area * 2.0e-3 rtol = 4eps()
    @test s.plasma_fast_energy ≈ area * 2.0e-3 * (sqrt(65) - 1) rtol = 4eps()
    @test (s.plasma_peak_window, s.plasma_fast_window, s.plasma_fast_energy_window) == (s.plasma_peak, s.plasma_fast, s.plasma_fast_energy)
    @test born_summary(st; u_fast = 8.0).plasma_fast == 0
    # a second fast one between the back band and the fade is in the window's numbers only
    k = findfirst(j -> p.status[j] == ALIVE && back_band_end(st) + 1 <= (p.cell[1][j] - 1 + p.offset[1][j]) * h[1] < 11.0, eachindex(p.w))
    p.w[k] = 5.0e-3; p.u[2][k] = 12.0
    s2 = born_summary(st)
    @test (s2.plasma_peak, s2.plasma_fast, s2.plasma_fast_energy) == (s.plasma_peak, s.plasma_fast, s.plasma_fast_energy)
    @test s2.plasma_peak_window ≈ sqrt(145) - 1 rtol = 4eps()
    @test s2.plasma_fast_window ≈ area * 7.0e-3 rtol = 4eps()
    @test s2.plasma_fast_energy_window ≈ area * (2.0e-3 * (sqrt(65) - 1) + 5.0e-3 * (sqrt(145) - 1)) rtol = 4eps()
end

# Tests of the WP-065 case: every trapped electron at a fixed laser energy.
# T-912 runs the case at λ₀/8 to t = 190 eleven times, a few minutes in all.

@testset "T-910 a₀ from the fixed energy inverts laser_energy in duration and waist, with no energy derivative along either [A3, WP-065]" begin
    E = budget(2.175)
    @test a0_for_energy(TCASE.x0[1], TCASE.x0[2]; energy = E) ≈ 2.175 rtol = 4eps()
    @test budget() == pulse_energy(1.9, 6.0, 10.0)
    for (τ, w) in ((6.0, 10.0), (4.5, 7.0), (8.0, 14.0))
        prob = trapped_problem(τ, w, 80.0, 54.0, 0.072; energy = E, ppl = 8, t_end = 1.0)
        l = prob.lasers[1]
        @test l.duration == τ && l.waist == w && l.focus == 80.0 && l.delay == 54.0
        @test laser_energy(l) ≈ E rtol = 8eps()                               # the engine's energy of the pulse it launches
        # the energy does not move with τ or w: its derivative through the problem's laser is zero to roundoff
        g = ForwardDiff.gradient(v -> laser_energy(trapped_problem(v[1], v[2], 80.0, 54.0, 0.072; energy = E, ppl = 8, t_end = 1.0).lasers[1]), [τ, w])
        @test maximum(abs, g) < 1.0e-12 * E
        # and a₀ does: −a₀/(2τ) along τ
        @test ForwardDiff.derivative(t -> a0_for_energy(t, w; energy = E), τ) ≈ -a0_for_energy(τ, w; energy = E) / (2τ) rtol = 8eps()
    end
    # the dopant follows the plasma's density, and the other parameters reach the problem
    prob = trapped_problem(6.0, 10.0, 70.0, 50.0, 0.05; ppl = 8, t_end = 1.0)
    @test prob.ionization.density ≈ TCASE.dopant_fraction * 0.05 rtol = 4eps()
    @test prob.species[1].density == 0.05 && prob.lasers[1].focus == 70.0 && prob.lasers[1].delay == 50.0
    @test prob.lasers[1].a0 ≈ 1.9 rtol = 4eps()
    # each laser parameter alone can carry a dual (the element type promotes over all of them)
    for (k, v) in ((3, 70.0), (4, 50.0), (2, 12.0))
        f(t) = (x = Any[6.0, 10.0, 70.0, 50.0, 0.05]; x[k] = t; trapped_problem(x...; energy = E, ppl = 8, t_end = 1.0))
        l = f(ForwardDiff.Dual{Nothing}(v, 1.0)).lasers[1]
        @test ForwardDiff.value(getfield(l, (:a0, :waist, :focus, :delay)[k])) == v
        @test ForwardDiff.partials(getfield(l, (:a0, :waist, :focus, :delay)[k]))[1] == 1
    end
    # the state's run is the problem's: at a point away from the base, its antenna and its dopant are trapped_problem's
    st, _, sp = start_of(trapped_problem(5.0, 12.0, 70.0, 50.0, 0.05; energy = E, ppl = 8, t_end = 1.0); ppl = 8, ny = 1)
    @test trapped_state(5.0, 12.0, 70.0, 50.0, 0.05; energy = E, ppl = 8, t_end = 1.0, capacity = 60_000) isa typeof(st)
    @test sp.lasers[1].a0 ≈ a0_for_energy(5.0, 12.0; energy = E) rtol = 4eps()
    @test sp.ionization.density ≈ 0.0005 rtol = 4eps()
end

@testset "T-911 the objective is Δx·Δy·Σ w(γ − 1)S(x)R(u_x) over the born and the plasma's electrons alike [A3, WP-065]" begin
    xs = [5.0, 15.0, 18.0, 30.0, 30.0, 60.0]
    us = [(8.0, 0.0, 0.0), (6.0, 1.0, 0.0), (4.0, -2.0, 0.5), (2.0, 3.0, 0.0), (10.0, 0.3, -0.2), (3.5, 0.0, 0.0)]
    ws = [1.0e-3, 2.0e-3, 5.0e-4, 7.0e-4, 3.0e-4, 1.1e-3]
    st = born_test_state(xs, us, ws)
    h = cellsize(st.grid)
    # the case's position weight is WP-064's first-bucket weight (12, 24)
    @test TCASE.fade == BCASE.fade
    tparts(st; kw...) = trapped_parts(st; fade = BCASE.fade, kw...)
    tenergy(st; kw...) = trapped_energy(st; fade = BCASE.fade, kw...)
    S(x) = (q = clamp((x - 12) / 12, 0, 1); q^3 * (10 - 15q + 6q^2))
    R(u) = (q = clamp((u - 3) / 2, 0, 1); q^3 * (10 - 15q + 6q^2))
    γm(u) = sqrt(1 + sum(abs2, u)) - 1
    born = h[1] * h[2] * sum(w * γm(u) * S(x) * R(u[1]) for (x, u, w) in zip(xs, us, ws))
    parts = tparts(st)
    @test parts.born ≈ born rtol = 8eps()
    @test parts.plasma == 0                                                    # a cold plasma: no electron above u_x = 3
    @test tenergy(st) == parts.born + parts.plasma
    # the electron at x = 30 with u_y = 3 and u_x = 2 has γ − 1 = 2.74 and counts for nothing: it is the forward momentum that decides
    us2 = copy(us); us2[4] = (2.0, 0.0, 0.0)
    @test tparts(born_test_state(xs, us2, ws)).born === parts.born
    # a fast plasma electron counts as a born one would: the label does not matter
    p = st.species[1]
    i = findfirst(j -> p.status[j] == ALIVE && (p.cell[1][j] - 1 + p.offset[1][j]) * h[1] >= 30.0, eachindex(p.w))
    xi = (p.cell[1][i] - 1 + p.offset[1][i]) * h[1]
    p.w[i] = 2.0e-3; p.u[1][i] = 4.5; p.u[2][i] = 1.0
    # and its summary: the plasma's numbers are its own, the born ones unchanged
    sp = trapped_summary(st; fade = BCASE.fade)
    Rp = h[1] * h[2] * 2.0e-3 * S(xi) * R(4.5)
    @test sp.weight_plasma ≈ Rp rtol = 8eps()
    @test sp.mean_energy_plasma ≈ sqrt(1 + 4.5^2 + 1) - 1 rtol = 8eps()
    @test sp.ux_max_plasma == 4.5
    @test sp.fast_plasma_window == 0                                            # u_x = 4.5 is below the rise's top
    p.u[1][i] = 5.5
    @test trapped_summary(st; fade = BCASE.fade).fast_plasma_window ≈ h[1] * h[2] * 2.0e-3 rtol = 8eps()
    @test trapped_summary(st; fade = BCASE.fade).weight_born === sp.weight_born
    p.u[1][i] = 4.5
    @test tparts(st).plasma ≈ h[1] * h[2] * 2.0e-3 * (sqrt(1 + 4.5^2 + 1) - 1) * S(xi) * R(4.5) rtol = 8eps()
    @test tparts(st).born === parts.born
    @test tenergy(st) ≈ parts.born + h[1] * h[2] * 2.0e-3 * (sqrt(1 + 4.5^2 + 1) - 1) * S(xi) * R(4.5) rtol = 8eps()
    # a retired one is not read, nor a free slot's leftovers
    p.status[i] = 0x00
    @test tparts(st).plasma == 0
    e = ionization_electrons(st)
    e.u[1][7] = 1.0e6; e.w[7] = 1.0; e.cell[1][7] = 40
    @test tparts(st).born === parts.born
    # another trapping weight: (1, 2) counts the electron at u_x = 2 in full and every other in full
    @test tparts(st; rise = (1.0, 2.0)).born ≈ h[1] * h[2] * sum(w * γm(u) * S(x) for (x, u, w) in zip(xs, us, ws)) rtol = 8eps()
    @test_throws ArgumentError tenergy(st; rise = (5.0, 3.0))
    @test_throws ArgumentError tenergy(st; rise = (3.0, 3.0))
    @test_throws ArgumentError trapped_energy(st; fade = (1.0, 24.0))     # inside the back band
    # ∂J/∂u_x of one electron mid-rise is the energy's part and the rise's: w·S·(u_x/γ·R + (γ − 1)·R′)
    d = ForwardDiff.derivative(t -> slot_trapped(ALIVE, Int32(1), 30.0, 4.0 + t, 0.0, 0.0, 1.0, 1.0, 12.0, 24.0, 3.0, 5.0), 0.0)
    g = sqrt(17.0)
    @test d ≈ 4 / g * R(4.0) + (g - 1) * 15 / 8 / 2 rtol = 8eps()
    # the summary's numbers on the same electrons
    s = trapped_summary(st; fade = BCASE.fade)
    @test s.J === tenergy(st) && s.J_born === parts.born && s.J_plasma == 0
    @test s.weight_born ≈ h[1] * h[2] * sum(w * S(x) * R(u[1]) for (x, u, w) in zip(xs, us, ws)) rtol = 8eps()
    @test s.mean_energy_born ≈ s.J_born / s.weight_born rtol = 4eps()
    @test s.fast_born_window ≈ h[1] * h[2] * (ws[2] + ws[5]) rtol = 8eps()   # u_x > 5 ahead of the back band (x = 5 is inside it at λ₀/8)
    @test s.ux_max_born == 10.0 && s.particles == 6 + count(==(ALIVE), p.status)
    # a fast plasma electron between the back band's end (7.1) and the weight (12) is dark current, not J
    k = findfirst(j -> p.status[j] == ALIVE && back_band_end(st) + 0.5 <= (p.cell[1][j] - 1 + p.offset[1][j]) * h[1] < 11.5, eachindex(p.w))
    p.w[k] = 5.0e-3; p.u[1][k] = 6.0
    sd = trapped_summary(st; fade = BCASE.fade)
    @test sd.dark_weight ≈ h[1] * h[2] * 5.0e-3 rtol = 8eps()
    @test sd.dark_energy ≈ h[1] * h[2] * 5.0e-3 * (sqrt(37.0) - 1) rtol = 8eps()
    @test sd.J === s.J
    @test sd.fast_plasma_window ≈ h[1] * h[2] * 5.0e-3 rtol = 8eps()
    @test s.dark_weight == 0
    # a fast born electron between the back band's end (7.1) and the weight (12) is in the window's number only
    st9 = born_test_state([xs; 9.0], [us; (6.0, 0.0, 0.0)], [ws; 4.0e-4])
    s9 = trapped_summary(st9; fade = BCASE.fade)
    @test s9.fast_born_window ≈ h[1] * h[2] * (ws[2] + ws[5] + 4.0e-4) rtol = 8eps()
    @test s9.J_born === trapped_summary(born_test_state(xs, us, ws); fade = BCASE.fade).J_born
end

"The WP-065 objective at λ₀/8 read at t = 190, with a trapping weight from u_x = 0.5 to 3 that cuts through the electrons of that early time (the born ones reach u_x = 2.7, the plasma's 1.0)."
trapped_small(x...; kw...) = trapped_objective(x...; ppl = 8, t_end = 190.0, capacity = 60_000, rise = (0.5, 3.0), fade = BCASE.fade, energy = budget(2.2), kw...)

@testset "T-912 ForwardDiff through the run gives the objective's gradient in all five parameters [A3, A7, WP-065]" begin
    x0 = collect(TCASE.x0)
    xd = [ForwardDiff.Dual{Nothing}(x0[i], ntuple(j -> Float64(i == j), 5)) for i in 1:5]
    Jd = trapped_small(xd...)
    J = trapped_small(x0...)
    @test ForwardDiff.value(Jd) ≈ J rtol = 64eps()                              # the duals do not change the run, to roundoff (T-833)
    @test J > 1.0e-4
    st = trapped_state(x0...; ppl = 8, t_end = 190.0, capacity = 60_000, energy = budget(2.2))
    parts = trapped_parts(st; rise = (0.5, 3.0), fade = BCASE.fade)
    @test parts.born > 0 && parts.plasma > 0                                    # both species are read
    ad = ForwardDiff.partials(Jd)
    # At this early time J is mostly the plasma's quiver in the pulse (the rise from 0.5 reaches
    # it): the test is of the derivative through both species, not of trapping. J is rough in the
    # delay below its scale (central differences there reach ForwardDiff only at steps below about
    # 10⁻³), so every direction is checked at a relative step of 2⁻¹⁸.
    for i in 1:5
        hi = x0[i] * 2.0^-18
        xp, xm = copy(x0), copy(x0)
        xp[i] += hi; xm[i] -= hi
        fd = (trapped_small(xp...) - trapped_small(xm...)) / 2hi
        @test ad[i] ≈ fd rtol = 1.0e-5
        @test abs(ad[i]) > 1.0e-3 * J / x0[i]                                  # every parameter moves J
    end
end

@testset "T-913 a finer y keeps the box, the absorbers' and the dopant's widths, and ny = 1 is WP-064's case [WP-065]" begin
    p1 = born_problem(; ppl = 16, t_end = 1.0)
    p2 = born_problem(; ppl = 16, t_end = 1.0, ny = 2)
    @test p2.cells == (p1.cells[1], 2 * p1.cells[2])
    @test p2.length[2] ≈ p1.length[2] rtol = 4eps()
    @test p2.length[1] == p1.length[1]
    @test p2.dt < p1.dt
    @test p2.lasers[1].center ≈ p1.lasers[1].center rtol = 4eps()
    r1, r2 = p1.ionization.rows, p2.ionization.rows
    @test r2[1] ≈ r1[1] rtol = 8eps()
    @test r2[2] ≈ r1[2] rtol = 8eps()
    @test born_numerics(16, 2).absorber.cells == 2 * born_numerics(16).absorber.cells
    @test born_numerics(16, 2).back_absorber.cells == born_numerics(16).back_absorber.cells
    @test born_start(; ppl = 16, ny = 2, t_end = 1.0, capacity = 10)[1].numerics.absorber.cells == 2 * BCASE.pml_y
    @test_throws ArgumentError born_problem(; ppl = 16, ny = 0)
    # a longer window keeps more wake behind: its front, and so the lab frame's pulse and plasma, stay where they were
    pe = born_problem(; ppl = 16, t_end = 1.0, extend = 4)
    @test pe.cells == (18 * 16, 128)
    @test pe.window.start ≈ BCASE.start + 8π rtol = 4eps()
    @test pe.length[1] - pe.window.start ≈ p1.length[1] - p1.window.start rtol = 8eps()   # the front's lab position at any time after both start
    @test pe.lasers[1].position == p1.lasers[1].position
    @test_throws ArgumentError born_problem(; ppl = 16, extend = -1)
    # the case's position weight starts ahead of the back band at λ₀/16 and finer
    for (ppl, ny) in ((16, 1), (32, 2), (64, 4))
        st, _, _ = born_start(; ppl, ny, t_end = 1.0, capacity = 10)
        @test back_band_end(st) < TCASE.fade[1]
    end
    # at ny = 1 the grid is WP-064's to the bit: its formulas before the keywords
    for ppl in (8, 16, 32)
        q = born_problem(; ppl, t_end = 1.0)
        h = 2 * π / ppl
        hy = (ppl ÷ 8) * h
        @test q.cells == (14 * ppl, 128)
        @test q.dt === 95 / 100 / sqrt(1 / h^2 + 1 / hy^2)
        @test q.length === (14 * ppl * h, 128 * hy)
        yc = 128 * hy / 2
        @test q.ionization.rows === (yc - 18 * hy, yc + 18 * hy)
        @test born_numerics(ppl).absorber.cells == 12
    end
end

@testset "T-914 the case runs in Float32, with J and its derivative within 10⁻⁴ of Float64 [A8, WP-065]" begin
    kw = (; ppl = 8, t_end = 190.0, capacity = 60_000, energy = budget(2.2), float = Float32)
    st32 = trapped_state(TCASE.x0...; kw...)
    @test eltype(st32.species[1].w) == Float32 && eltype(ionization_electrons(st32).w) == Float32
    J32 = trapped_energy(st32; fade = BCASE.fade, rise = (0.5, 3.0))
    @test J32 isa Float32
    J64 = trapped_small(TCASE.x0...)
    @test J32 ≈ J64 rtol = 1.0e-4                                               # 1.6·10⁻⁵ measured
    # a Float32 dual keeps the run in Float32, and its derivative in the focus is Float64's
    d32 = trapped_energy(trapped_state(6.0f0, 10.0f0, ForwardDiff.Dual{Nothing}(80.0f0, 1.0f0), 54.0f0, 0.072f0; kw...); fade = BCASE.fade, rise = (0.5, 3.0))
    d64 = trapped_small(6.0, 10.0, ForwardDiff.Dual{Nothing}(80.0, 1.0), 54.0, 0.072)
    @test d32 isa ForwardDiff.Dual{Nothing,Float32}
    @test ForwardDiff.partials(d32)[1] ≈ ForwardDiff.partials(d64)[1] rtol = 1.0e-4
    # a Float32 dual in the density stays Float32 through the dopant's fraction of it
    stn = trapped_state(6.0f0, 10.0f0, 80.0f0, 54.0f0, ForwardDiff.Dual{Nothing}(0.072f0, 1.0f0); kw..., t_end = 1.0)
    @test eltype(ionization_electrons(stn).w) == ForwardDiff.Dual{Nothing,Float32,1}
    @test trapped_capacity(16, 1) == BCASE.capacity && trapped_capacity(64, 4) == 2_560_000
end
