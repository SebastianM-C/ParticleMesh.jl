"""Laser velocity in vacuum: slope of the axis centroid of Ey^2 over t in [150, 290]; and the
arrival-time slip at t = 280 against the reference run (by interpolation)."""
import json, sys, numpy as np, math
sys.path.insert(0, "/home/user/lowdisp/src")
import dispersion as D

def fit(run, lo=150, hi=290):
    s = json.load(open(f"{run}/summary.json"))
    t = np.array([e["t"] for e in s["series"]]); c = np.array([e["centroid_lab"] for e in s["series"]])
    a = np.array([e["a_peak"] for e in s["series"]])
    k = (t >= lo) & (t <= hi)
    p = np.polyfit(t[k], c[k], 1)
    return s["meta"], p[0], float(np.interp(280.0, t, c)), t, c, a

if __name__ == "__main__":
    ref = None
    rows = []
    for r in sys.argv[1:]:
        m, v, c280, *_ = fit(r)
        vp, vg = D.velocities(m["solver"], m["dx"], m["dy"], m["dt"])
        rows.append((r.split("/")[-1], v, c280, vg))
    refc = dict((x[0], x[2]) for x in rows).get("vac-r64x4-M4-yee")
    for n, v, c, vg in rows:
        print(f"{n:22s} 1-v_meas={1-v:+.2e}  1-vg_theory(k=1,ky=0)={1-vg:+.2e}  centroid(280)-ref={c-refc:+.3f}")
