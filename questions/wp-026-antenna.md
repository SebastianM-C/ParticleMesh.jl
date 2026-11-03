# Choices WP-026 took that its text did not settle

## 1. Which source term "antenna plane" means

A2 asks for a "laser by antenna plane at the window front". An electric current sheet alone radiates the same pulse in both directions, half the energy backwards, where in a periodic box it wraps round and in an LWFA run it would cross the plasma behind the plane. **Assumption:** the antenna is a total-field/scattered-field (Huygens) plane, an electric and a magnetic current sheet half a cell apart, as PIConGPU 0.8.0's `incidentField` is (`questions/wp-017-lwfa-case.md` §3). It launches one way, and measured, the energy behind its plane is 4·10⁻⁷ of the pulse for a waist of 6 c/ω₀ (T-391). Its cost is a transient surface charge on the plane's E column and B column, which T-396 computes exactly. Smilei injects through a boundary condition instead; the injected field should be compared in vacuum before any cross-code plasma run, as that question file says.

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

The paraxial beam prescribes the transverse E at the plane. A plane-wave component at angle α has Bz/Ey = 1/cos α (in-plane polarization) and By/Ez = −cos α (out of plane). With B = x̂ × E the antenna launched a beam 1.3 % off its prescribed profile at waist 6. **Assumption:** the antenna's B is E ∓ ½∂²E/∂y², exact to order k_y², computed from the analytic second derivative of the profile. The remaining (k_y/k)⁴ term is the measured error of T-392, (3/16)·√105/w0⁴ of the field: 1.5·10⁻³ at w0 = 6, 1·10⁻⁷ at an LWFA waist of 30. A nonlocal (per-k_y) B would remove it, at the cost of an FFT along the plane per step; that waits until a case needs it.

Affected: `lib/ParticleMeshElectromagnetic/src/antenna.jl`, `problem.jl`, `step2d.jl`; `lib/ParticleMeshElectromagnetic/test/antenna.jl`; `trace/wp/WP-026-laser-antenna.md`.
