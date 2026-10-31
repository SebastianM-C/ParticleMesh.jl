# Particle pushers: what the literature says, for WP-015

Background for WP-015: the relativistic pushers worth comparing, the claims each paper makes, and the test problems worth reusing. Papers are cited by journal, volume and page. The DOIs were checked against the bibliographic records, not completed from memory. Page and section numbers are the journal's unless a note says the copy read was a preprint. "Here" means among the papers read for this note, not the whole literature.

## The candidates for the in-loop push

The PIC push is one explicit update per particle at a global, CFL-set Δt, with one field gather and no iteration of unbounded length, generic in the element type. That rules out implicit, adaptive and multi-evaluation methods for the in-loop push. They stay available for frozen-field test-particle runs through SciML (`sciml-reuse.md` §2).

| Pusher | Volume-preserving in (x, u) | E×B velocity | Energy, E = 0 | Gyrophase per step, uniform B | Extra cost over Boris |
|---|---|---|---|---|---|
| Boris (Boris 1970, Proc. 4th Conf. Num. Sim. Plasmas, pp. 3–67) | yes | exact nonrelativistically, wrong once relativistic | exact | 2·atan(θ/2), lags | – |
| Vay (Phys. Plasmas 15, 056701 (2008)) | no | correct | exact | same as Boris (Vay App. B) | one more square root |
| Higuera–Cary (Phys. Plasmas 24, 052104 (2017)) | yes | correct | exact | error ≤ Boris's; lags for γ < √3, leads above (their Eq. 27) | one more square root |
| Boris with exact rotation, "Boris-C" (Zenitani and Umeda, Phys. Plasmas 25, 112110 (2018)) | yes (their App. A) | wrong at O((ΩΔt)²), even nonrelativistically | exact | exact | sin and cos; needs a form smooth through B = 0 for duals |
| Quadratic-unitary (Gordon and Hafizi, arXiv:2006.10146, 2020; venue not verified) | not discussed | not discussed | exact | – | steps in proper time; a lab-Δt step needs their conversion (Eq. 22) |

The table entries come from these papers:

- **Higuera and Cary.**
  - Volume preservation and the E×B velocity: their Secs. IV–V.
  - Their pusher differs from Boris only in the γ used for the rotation: their Eq. 20 instead of Eq. 21, with γ₋ ≥ γ_new (Eq. 25).
  - Near resonant orbits of their integrable test system (Eq. 48, H = 4, Δt = 1/10), Vay's energy error oscillates with amplitude δH ≈ 0.1, about 3% of H. They describe these errors as bounded and periodic. Boris and Higuera–Cary stay about 10² smaller (Figs. 3–4).
  - They show the per-step volume change of Vay's scheme is a ratio J(x, u_i)/J(x, u_f) (Eqs. 40–42), which telescopes in constant fields.
- **Boris-C (Zenitani and Umeda).** It removes the gyrophase error, but it no longer reproduces the E×B drift that plain Boris gets right nonrelativistically: its fixed point is off by a factor of about (θ/2)/tan(θ/2). The exact-rotation compositions of He et al. (below) are unstable at hω = kπ. Zenitani and Umeda nevertheless call Boris-C the safer choice at large Δt compared with their tan-half-angle variant.
- **Ripperda et al. (ApJS 235, 21 (2018), DOI 10.3847/1538-4365/aab114)** compare Boris, Vay and Higuera–Cary on relativistic test-particle cases.
  - The three explicit schemes cost about the same (Sec. 2.5).
  - In uniform B at γ = 10⁶, Higuera–Cary has about half the phase error of Boris and Vay (Sec. 3.1.2). Vay and Higuera–Cary have γ errors larger than Boris there, all near machine precision.
  - In the force-free case, the Boris error grows with Δt and the others barely depend on it (Sec. 3.1.3, Table 2).
  - In crossed fields at drift Lorentz factor κ = 100, no explicit scheme recovers the comoving gyroradius (Sec. 3.1.4).

  Their γ and Δt/T lie far outside the LWFA range, so their numbers do not transfer. Their list of tests does.
- **Gordon and Hafizi** test a plane wave at a₀ = 10 and 1000. At a₀ = 10, with 63 proper-time steps of ω₀Δs = 0.1:
  - Boris and Vay are "not viable".
  - Higuera–Cary shows visible errors.
  - Their pusher follows the exact orbit.

  Two caveats apply:
  - Their quadratic-unitary update equals the exact update for a locally uniform null field (E·B = 0, E² = B²). A real plane-wave orbit still has errors from the field varying across a step and from the position update.
  - A constant proper-time step is a constant phase step. Their run therefore resolves one laser period with about 63 steps. A fixed lab Δt gives a phase step Δη = Δt/γ, so one figure-8 period takes 2π(1 + a₀²/4)/Δt steps: about 820 at a₀ = 10 and ω₀Δt = 0.2. At PIC resolution the orbit is resolved more than ten times more finely than in their comparison, so their verdict does not carry over unchecked.

  They also attribute the Boris-type error in strong fields to splitting the electric boost from the magnetic rotation. They state that the condition persists even when each factor is evaluated exactly, so an exact rotation (Boris-C) does not address it.
- **Zhang et al. (Phys. Plasmas 22, 044501 (2015))** state that relativistic Boris is not volume-preserving. Higuera and Cary (Sec. V) answer that the scheme Zhang et al. construct is itself the Boris scheme, and that relativistic Boris preserves volume. Zenitani and Umeda (Sec. I) remark that Zhang's solver appears to be another expression of the Boris solver. Zhang's Eq. 3 is consistent with that: γ is taken after the first electric half-kick.

## Structure and long-time energy: what is proved

- **Qin et al. (Phys. Plasmas 20, 084503 (2013))** show that the nonrelativistic Boris map is volume-preserving but not symplectic.
  - The long-time energy bound is shown numerically and "expected", not proved.
  - Implementers should derive the update from u⁺ − u⁻ = (u⁺ + u⁻) × t/γ with t = qBΔt/(2m) (Zenitani and Umeda, Eq. 4). They should not copy the matrix form of Qin's Eqs. 8–12, where the relative signs of Ω̂ and the factor qΔt/(2m) need care.
  - Qin's symplecticity conditions (Eqs. 16–18) are the block form of M J Mᵀ = J. That is equivalent to the Mᵀ J M = J of their Eq. 14, whose block form is S₁ᵀS₃ = S₃ᵀS₁, S₂ᵀS₄ = S₄ᵀS₂, S₁ᵀS₄ − S₃ᵀS₂ = I.
- **Webb (J. Comput. Phys. 270, 570 (2014))** claims that Boris and Vay are symplectic. Two comments dispute this:
  - Zhang, Jia and Sun (J. Comput. Phys. 282, 43 (2015)). Only its role as cited by Ellison et al. was checked.
  - Ellison, Burby and Qin (J. Comput. Phys. 301, 489 (2015)). They prove that Boris is variational only for constant B, and say the same objection applies to Webb's relativistic argument.

  So for non-uniform fields, no paper here supports calling Boris or Vay symplectic.
- **Hairer and Lubich (BIT Numer. Math. 58, 969 (2018), DOI 10.1007/s10543-018-0713-1)** prove near-conservation of energy, |ΔE| ≤ C_N h² for nh ≤ h^−N, for nonrelativistic Boris in two cases: constant B, or a quadratic electric potential with any B(x).
  - The proof assumes smooth, time-independent fields and a numerical solution that stays in an h-independent compact set.
  - They show numerically that the bound can fail outside these cases: linear drift O(h²t) and random-walk errors, for a non-uniform B with a cubic-plus-quartic potential (Examples 5.1–5.2, preprint pages).
  - In a time-dependent laser field the exact energy is not conserved, so there is nothing to bound. What can be measured there is the error against a reference solution and the plane-wave invariants.
- **He, Sun, Liu and Qin (J. Comput. Phys. 281, 135 (2015))** treat Boris as a composition of volume-preserving splitting steps. The exact-rotation variant is unstable at hω = kπ, where the Cayley form (Boris) is not. Knapp et al. (Phys. Rev. E 92, 063310 (2015)) report instability of the exact-exponential and related schemes in a Penning trap near f_c h ≈ n.
- **Qin et al. 2013 and Zenitani and Umeda** show RK4 damping the gyration over long runs. This is the baseline WP-015's scripts use for a scheme that does not preserve volume.

## Not for the in-loop push

Each of these needs iteration, field values at sub-step times, static fields or per-particle step sizes:

- Boris-SDC (Winkel, Speck and Ruprecht, J. Comput. Phys. 295, 456 (2015));
- line integral methods (Brugnano, Montijano and Rández, J. Comput. Phys. 396, 209 (2019));
- adaptive symplectic methods (Shi et al., J. Comput. Dyn. 6, 429 (2019));
- the filtered Boris and the variational integrator for strong magnetic fields (Hairer, Lubich and Wang, DOI 10.1007/s00211-020-01105-3; Hairer and Lubich, Numer. Math. 144, 699 (2020)). Both target a strong, slowly varying B, which a laser is not;
- adaptive Boris–Buneman (Toggweiler et al., J. Comput. Phys. 273, 255 (2014)). It adapts the space-charge solve, not the pusher, and it writes the relativistic Boris kick in momentum variables (their Alg. 3).

Boris-SDC is the one that handles time-dependent fields. It is the candidate if frozen-field test-particle runs ever need high order. Buneman (J. Comput. Phys. 1, 517 (1967)) uses the tan-half-angle rotation that Boris builds on.

## Test problems worth reusing

- **Uniform E, B = 0, relativistic.**
  - Exact x(t) and γ(t) are known.
  - Boris, Vay and Higuera–Cary get u exactly.
  - Ripperda et al. (Sec. 3.1.1) show that the position error comes from the half-step offset of the leapfrog start, and that it goes away when x at the half step is taken from the analytic solution.
- **Uniform B.**
  - |u| is exact.
  - The per-step angle is 2·atan(θ/2) for Boris and Vay, exact for Boris-C, and 2·atan(β/γ_new) for Higuera–Cary, with γ_new from their Eq. 20.
- **Crossed E and B.**
  - A particle moving at E×B/B² stays there for Vay and Higuera–Cary (Higuera–Cary Eqs. 28–29). This includes Ripperda's force-free case E = −v×B with v ⊥ B.
  - A force-free particle with a component along B is a separate case.
- **The plane-wave figure-8 orbit** (Sarachik and Schappert, Phys. Rev. D 1, 2738 (1970)).
  - The closed form assumes the electron was at rest before an adiabatically switched-on wave.
  - In a constant-amplitude wave, a particle started at rest must start at a zero of the vector potential, or it picks up a transverse drift.
- **Higuera and Cary's integrable system** (their Eq. 48, Δt = 1/10, H = 4, p_z = 0, p_y,0 ∈ {1.625, 1.775}). It shows Vay's energy errors near resonant orbits. The remaining initial conditions are not stated in the text.
- **Nonrelativistic long-run cases with stated parameters:**
  - Qin et al. 2013 (B = r e_z, φ = 10⁻²/r);
  - Hairer and Lubich 2018, Examples 5.1–5.2;
  - the Penning traps of Knapp et al. 2015 and Winkel et al. 2015.
