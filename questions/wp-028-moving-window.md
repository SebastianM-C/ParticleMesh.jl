# Choices WP-028 took that its text did not settle

## 1. Which axis, how fast, from when

**Assumption:** the window moves along axis 1, the antenna's propagation axis (+x), at c, from a time `start` of the problem (`MovingWindow(; start)`), by whole cells: after step k it has moved `max(0, floor((kΔt − start)/Δx₁))` cells. This is Smilei's convention, where the window moves along x from a start time. It is not checked in Smilei's source here. PIConGPU's window slides along y only (`questions/wp-017-lwfa-case.md` §3), so a cross-code generator has to swap the axes. A speed other than c, and a start given as a position instead of a time, are not implemented.

## 2. A memory shift, not a ring buffer

B1 left the mechanism open. **Assumption:** at the end of a step that moves the window, every E and B component is copied one column back through the scratch ρ, and the particles' cells go down by one. The grid stays a `PeriodicGrid{2}`, and no kernel signature or index computation changes. A ring buffer (a moving index offset) would avoid the six field copies per shift, about two shifts every three steps at 0.95 of the 2D Courant limit. In exchange, every kernel that reads a neighbour along x would need to know the offset. The measured cost of a shift is in `reports/wp-028-moving-window.md`. The choice can be revisited without changing the problem description.

## 3. The x edges

The window's axis is not periodic. **Assumption:** the outside is field-free and has no current. After every Yee sub-step the x-guards of the updated components are zero, and the current and charge in the x-guards are dropped before the fold. That is exact at the front, where nothing moves faster than the window. At the back it reflects whatever reaches the edge. In 2D that includes the trailing field a pulse leaves behind it, about 15 % of the pulse's peak at the back edge in T-472's run. The reflection travels forward at most at the numerical speed of Yee's stencil, falls off within about 24 columns, and the shifts leave it behind. An absorbing layer at the back, or the PML of A2's "later", would remove it; that is not part of WP-028.

## 4. Particles at the x edges

**Assumption:** a particle that crosses an x edge in a step is retired before the current of that move is deposited. The open y edge of WP-027 retires after the deposit, because both sides of its seam are layer rows. Here the two sides of the seam are the window's back and front. The engine places a particle's flux stencil at its new cell, so the current of a particle that left through the back would land at the front. Retired first, the particle takes its charge out of the box with no current. That breaks Gauss's law only on the two back nodes its stencil reached, and the next shifts drop those nodes.

The fresh plasma is loaded on the injection column n₁ − 1 − r − P, where r is how far the shape's stencil reaches past its cell (`stencil_reach(shape, Node())[2]`: 1 for the linear, 2 for the quadratic shape) and P the filter passes; the columns ahead of it receive none. The discrete front lags the continuous one by less than a cell, and a particle slower than c only falls further behind the continuous front, so once the window moves no particle gets more than one column ahead of the injection column. Then nothing the plasma deposits, filtered or not, reaches the front face (node n₁ + 1, an x-guard), where the step drops the current and holds Ey at zero, so that node's Gauss's law is not kept. The next shift makes that node the new front node. It also needs Ex on the front face (the edge n₁ + ½, the last array column of Ex): the shift carries it along instead of zeroing it, so that the new front node's law holds against the zero Ey of the fresh front column. A cold plasma at rest needs neither, and a warm one needs both (T-473, the report's §3). The price is r + P + 1 plasma-free columns at the front, three for the quadratic shape without a filter.

Before the window starts moving, the front is a fixed edge, and a particle that reaches it leaves the same residual. `EMProblem` therefore refuses `start > 0` for a species with a drift or a thermal speed. A cold plasma can still be set moving by the laser before the window starts, which no static check sees: a case should start the window before the laser reaches the plasma at the front, as LWFA cases do.

## 5. The plasma, the background and the initial field

**Assumption:** a window problem starts neutral. Each species is neutralized by an immobile background of its own profile (A2's immobile ions), so E⁰ = 0 and no initial solve is made. Gauss's law then reads D⁻·E = ρ − ρ₀(s), where ρ₀(s) is the charge of the cold quiet start of the window's plasma columns at its position s. A perturbed species (`amplitude ≠ 0`) and background fields (`E0`, `B0`) are refused. The first would need a non-periodic initial solve, and the second a non-zero outside. The longitudinal profile `DensityRamp(; vacuum, ramp)` is zero, then sin² over the ramp, then one. It is C¹, as A2's smoothness rule asks, where a linear ramp has a kink in its derivative. A polygonal profile with a down-ramp, as LWFA cases sometimes use, would be a second profile type. The profile is a property of the problem and multiplies every species.

## 6. The refill's momenta

The fresh plasma's drawn velocities, converted to γv, are taken as the momenta at the half step. That is the convention of Smilei and PIConGPU (`momenta_at = :t_minus_half`); no pusher kick is applied. It is exact where the fields at the front vanish, which §4 requires anyway. A refilled thermal plasma also does not reproduce the static run's state at that cell, which has evolved since t = 0. Only a cold plasma at rest is the same with and without the window (T-471).

A drawn speed can reach c, since the thermal spread is drawn in velocity (#257). The initial load refuses it, as a static load does. A refill cannot raise inside a step (§0.4), and dropping the particle would leave its charge out of the plasma against the background, so it is loaded at the drift velocity and counted (`window_superluminal`, T-479). This biases the tail of the refilled distribution by the counted particles only.

## 7. The antenna

**Assumption:** an antenna plane stays at its place in the lab frame. Its window column moves back with the shifts, and it stops once the window has passed it. The state refuses a laser whose pulse is not over by then, meaning six durations after its delay, where the envelope is below Float64 roundoff. A2's "antenna plane at the window front" is read as a plane fixed in the lab, ahead of the plasma, which the pulse leaves before the window passes it. An antenna that moves with the window would need the antenna's TF/SF corrections to follow a moving plane, and is not implemented.

## 8. Buffer capacity

The particle buffers hold `(1 + headroom)·prod(cells)·prod(ppc)` slots, with `EMNumerics(; headroom = 1/4)` by default. A plasma at rest keeps its count when the window moves. A plasma streaming forward at v stays 1/(1 − v) times as long in the window, so it needs a headroom of at least v/(1 − v). Trapped LWFA electrons are a small fraction of the plasma. A refill that finds no free slot does not stop the run, which could only happen through host branching on device data (§0.4). Instead the missing particles are counted on the device, and `window_lost(state)` reads the count after the run. A case file key for the headroom does not exist yet.

## 9. Not in this package

- The case-file schema: `grid.boundaries = ["window", ·]`, a `[window]` table and a profile per case. The cross-code generators would then need the axis swap of §1.
- An absorbing back edge (§3).
- Threads: the bitwise property is tested across reruns and launch shapes, not at 1 and 4 threads (T-474 has no worker run).
