"""Yee's on-axis group-velocity error for the laser (k = omega_0) against the physical wake phase
velocity deficit, 1 - v_phi ~ n/(2 n_c), over density and resolution; and the slip over a
propagation length. 2D cells dy/dx = 2 (this case) and 8; 3D dy = dz = 2 dx; dt = 0.95 CFL."""
import math, json, sys
sys.path.insert(0, "/home/user/lowdisp/src")
import dispersion as D

def yee_error(ppl, ratio, ndim=2, cfl=0.95):
    dx = 2 * math.pi / ppl; dy = ratio * dx
    dt = cfl / math.sqrt(1 / dx**2 + (ndim - 1) / dy**2)
    return 1 - D.velocities("Yee", dx, dy, dt)[1], dt / dx

def m4_error(ppl, ratio, cfl=0.95):
    dx = 2 * math.pi / ppl; dy = ratio * dx
    dt = cfl / math.sqrt(1 / dx**2 + 1 / dy**2)
    return 1 - D.velocities("M4", dx, dy, dt)[1]

if __name__ == "__main__":
    out = {}
    for label, ratio, nd in (("2D dy=2dx", 2, 2), ("2D dy=8dx", 8, 2), ("3D dy=dz=2dx", 2, 3)):
        for ppl in (16, 24, 32, 48, 64, 96, 128):
            e, r = yee_error(ppl, ratio, nd)
            out.setdefault(label, {})[ppl] = dict(yee=e, dt_over_dx=r, m4=m4_error(ppl, ratio) if nd == 2 else None)
    json.dump(out, open("/home/user/lowdisp/regimes.json", "w"), indent=1)
    L3mm = 3e-3 * 2 * math.pi / 0.8e-6
    print("3 mm =", round(L3mm), "c/omega0")
    for label, d in out.items():
        print(label)
        for ppl, v in d.items():
            print(f"  lambda/{ppl:3d} dt/dx={v['dt_over_dx']:.3f} Yee 1-vg={v['yee']:.2e}  slip over 3 mm={v['yee']*L3mm:6.1f} c/w0" + (f"  M4 1-vg={v['m4']:.1e}" if v['m4'] is not None else ""))
    for n in (0.0018, 0.003, 0.072):
        print(f"n/nc={n}: physical 1-v_phi ~ n/2nc = {n/2:.1e}, gamma_phi ~ {1/math.sqrt(n):.1f}, slip over 3 mm = {n/2*L3mm:.0f} c/w0, lambda_p = {2*math.pi/math.sqrt(n):.0f} c/w0")
