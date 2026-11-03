# Choices WP-026 took that its text did not settle

## 1. Which source term "antenna plane" means

A2 asks for a "laser by antenna plane at the window front". An electric current sheet alone radiates the same pulse in both directions, half the energy backwards, where in a periodic box it wraps round and in an LWFA run it would cross the plasma behind the plane. **Assumption:** the antenna is a total-field/scattered-field (Huygens) plane, an electric and a magnetic current sheet half a cell apart, as PIConGPU 0.8.0's `incidentField` is (`questions/wp-017-lwfa-case.md` §3). It launches one way. Measured where the leak is (in a periodic box it wraps round, so the runs keep it apart from the pulse), the energy behind its plane is 3·10⁻⁵ of the pulse for a waist of 6 c/ω₀ and a duration of 8/ω₀ (T-391). Its cost is a transient surface charge on the plane's E column and B column, which T-396 computes exactly. Smilei injects through a boundary condition instead; the injected field should be compared in vacuum before any cross-code plasma run, as that question file says.

## 2. Where the antenna lives

§0.1 puts the antenna in the LWFA module (`Wakefields`), which does not exist yet. **Assumption:** the antenna is a source of the electromagnetic module (`lib/ParticleMeshElectromagnetic/src/antenna.jl`): `EMProblem{2}` takes `lasers = (GaussianLaser(...),)`, and the 2D step applies it inside the Yee solve. A `Wakefields` package can re-export it or move it when the moving window (WP-028) gives the LWFA layer a reason to exist.

## 3. Definitions of the laser parameters

The package names a₀, waist, duration, focus and polarization without conventions. **Assumptions** (all in A1 units, `GaussianLaser`'s docstring):

- `a0` is the peak field on the axis at the focal plane, eE/(mₑcω₀), for a linear polarization.
- `waist` is the 1/e radius of the field at the focus. The beam is the paraxial Gaussian beam in two dimensions, with z_R = waist²/2 and the 2D Gouy phase −½ atan(ξ/z_R), not the 3D −atan.
- `duration` is the 1/e half width of the field envelope exp(−(t − delay)²/duration²), so the intensity FWHM is duration·√(2 ln 2). It is a Gaussian, not a sin² envelope, because the AD-smoothness rule of A2 prefers it. It has infinite support, so the delay sets where the pulse starts: about six durations put it below Float64 roundoff at t = 0.
- `chirp` is the coefficient b of a quadratic phase b·(t − delay)² at the plane, a linear chirp with instantaneous frequency 1 + 2b(t − delay) at the plane.
- `polarization` is the angle of a linear polarization from y towards z. No elliptical or circular polarization yet.
- `phase` is the carrier-envelope phase.
- The pulse travels along +x only, and the plane is snapped to the nearest primal node, so `position` is not differentiable.

Each of these is a parameter of the antenna source, so it can be a dual (A3). The plane's position along x, the axis's position along y and the direction are configuration.

## 4. The antenna's B for a focused beam

The paraxial beam prescribes the transverse E at the plane. A plane-wave component at angle α has Bz/Ey = 1/cos α (in-plane polarization) and By/Ez = −cos α (out of plane). With B = x̂ × E the antenna launched a beam 1.3 % off its prescribed profile at waist 6. **Assumption:** the antenna's B is E ∓ ½∂²E/∂y², exact to order k_y², computed from the analytic second derivative of the profile. Measured, the remaining error of the launched field at ω₀ is:

| waist | error | ratio to (3/16)·√105/w0⁴ |
|---|---|---|
| 6 | 1.65·10⁻³ | 1.1 |
| 12 | 1.2·10⁻⁴ | 1.3 |
| 24 | 1.3·10⁻⁵ | 2.2 |
| 30 | 7·10⁻⁶ | 3.0 |

Up to a waist of 12 the error is the (k_y/k)⁴ term, (3/16)·√105/w0⁴ of the field (T-392). At larger waists a floor of order Δ²/w0², from the grid, takes over: at waist 24 the error is 8·10⁻⁶ at Courant number 0.5 and 7·10⁻⁶ at 32 × 16 cells per wavelength. A nonlocal (per-k_y) B would remove the k_y⁴ term, at the cost of an FFT along the plane per step; that waits until a case needs it.

## 5. The antenna is exact at ω₀ only

The B the antenna injects, both its ∓½∂²E/∂y² term (whose exact coefficient is 1/(2ω²)) and its half-cell delay k(ω)·Δx/2, is that of the carrier frequency ω₀ = 1. A plane wave at ω₀ is therefore launched exactly (T-392); the rest of the pulse's band, about 1/duration wide and wider with chirp, is not. A breaker run measured this (waist 6, 16 × 8 cells, the backward leak measured where it is):

| duration, chirp | energy behind / pulse | W/Φ − 1 | group-velocity error |
|---|---|---|---|
| 8, 0 | 3·10⁻⁵ | 3·10⁻⁵ | 5·10⁻⁵ |
| 20, 0 | 6·10⁻⁶ | | |
| 8, 0.03 | 8·10⁻⁴ | −3.3·10⁻⁴ | 6·10⁻⁴ |
| 3, 0 | 3.6·10⁻³ | −2.9·10⁻³ | 1·10⁻³ |

At duration 3 the Gaussian's zero-frequency content, exp(−duration²/4) = 0.1, also leaves a static surface charge on the plane. **Assumption:** the antenna stays narrow-band. A pulse of 30 fs at 0.8 µm lasts about 70/ω₀, far narrower in band than these runs. A broadband or strongly chirped pulse would need a B per frequency, the same nonlocal operation as in §4, in time.

Affected: `lib/ParticleMeshElectromagnetic/src/antenna.jl`, `problem.jl`, `step2d.jl`; `lib/ParticleMeshElectromagnetic/test/antenna.jl`; `trace/wp/WP-026-laser-antenna.md`.
