"""Collect the run summaries, ParticleMesh's grid campaign and the dispersion theory into data.json for the page."""
import json, glob, math, os, sys, numpy as np
sys.path.insert(0, "/home/user/lowdisp/src")
import dispersion as D, regimes as R

RUNS = "/home/user/runs"
PM = "/mnt/project-files/grid-ppc-report/data/gpu/grid"
OUT = "/mnt/project-files/lowdisp-solver/data/data.json"

def wake_fit(series, lo=150.0, hi=300.0, band=(20, 34)):
    t, z = [], []
    for e in series:
        if e["t"] < lo or e["t"] > hi:
            continue
        zs = [q for q in e["Ex_zeros"] if band[0] < q < band[1]]
        if zs:
            t.append(e["t"]); z.append(zs[0] + e.get("pm_back", e.get("x0", 0.0)))
    if len(t) < 5:
        return None
    p = np.polyfit(t, z, 1)
    return dict(one_minus_v=float(1 - p[0]), z300=float(np.polyval(p, 300.0)))

def zero_at(series, T=297.0, band=(20, 34)):
    """The E_x zero closing the first bucket (window x), linearly interpolated to time T."""
    pts = []
    for e in series:
        zs = [q for q in e["Ex_zeros"] if band[0] < q < band[1]]
        if zs and abs(e["t"] - T) < 6:
            pts.append((e["t"], zs[0]))
    pts.sort()
    for (t0, z0), (t1, z1) in zip(pts, pts[1:]):
        if t0 <= T <= t1:
            return z0 + (T - t0) / (t1 - t0) * (z1 - z0)
    return None

def key(m):
    dt = m["dtmode"]
    return f'{m["solver"]}-{dt}'

def series_lite(s):
    return [dict(t=round(e["t"], 3), c=round(e["centroid_lab"], 4), a=round(e["a_peak"], 5)) for e in s["series"]]

data = dict(smilei={}, vacuum={}, pm={}, theory={}, regimes={})
for f in sorted(glob.glob(f"{RUNS}/pl-*/summary.json")):
    s = json.load(open(f)); m = s["meta"]
    res = f'r{m["ppl"]}x{m["nyfac"]}'
    row = dict(seed=m["seed"], J=s["J"], J_born=s["J_born"], J_plasma=s["J_plasma"], a0scale=m["a0scale"],
               n_born=s["born"]["n"], n_trapped=s["born"]["n_trapped"], neff=s["born"]["neff"],
               weight_born=s["born"]["weight"], weight_trapped=s["born"]["weight_trapped"],
               mean_energy=s["born"]["mean_energy_trapped"], mean_x=s["born"]["mean_x_trapped"], ux_max=s["born"]["ux_max"],
               zeros=s["series"][-1]["Ex_zeros"][:4], zback=zero_at(s["series"]), wake=wake_fit(s["series"]), a_end=s["series"][-1]["a_peak"],
               dt=m["dt"], dx=m["dx"], dy=m["dy"], seconds=int(open(os.path.dirname(f) + "/container.txt").read().split(",")[1].split()[0]),
               spectrum=s.get("spectrum"), trapped_x=s.get("trapped_x"))
    data["smilei"].setdefault(key(m), {}).setdefault(res, []).append(row)
for f in sorted(glob.glob(f"{RUNS}/vac-*/summary.json")):
    s = json.load(open(f)); m = s["meta"]
    res = f'r{m["ppl"]}x{m["nyfac"]}'
    t = np.array([e["t"] for e in s["series"]]); c = np.array([e["centroid_lab"] for e in s["series"]])
    k = (t >= 150) & (t <= 290); v = np.polyfit(t[k], c[k], 1)[0]
    vp, vg = D.velocities(m["solver"], m["dx"], m["dy"], m["dt"])
    data["vacuum"].setdefault(key(m), {})[res] = dict(series=series_lite(s), v=float(v), vg_theory=vg, vphi_theory=vp, dt_over_dx=m["dt"] / m["dx"])
for f in sorted(glob.glob(f"{PM}/grid-r*x*.json")):
    if "grad" in f or "E2" in f:
        continue
    d = json.load(open(f))
    data["pm"][d["id"].replace("grid-", "")] = dict(J=d["J"], zeros=d["axis"]["Ex_zeros"], zback=zero_at(d["series"]), wake=wake_fit(d["series"]), a_end=d["axis"]["a_peak"],
                                                    dt=d["dt"], dx=d["dx"], dy=d["dy"], ppl=d["ppl"], ny=d["ny"],
                                                    spectrum=d["bunch"].get("spectrum"))
# theory: laser 1 - v_g vs cells per wavelength for the three solvers on this case's cells
for s in ("Yee", "M4", "Lehe"):
    pts = []
    for ppl in range(12, 161, 4):
        dx = 2 * math.pi / ppl; dy = 2 * dx
        dty = 0.95 / math.sqrt(1 / dx**2 + 1 / dy**2)
        for lab, dt in (("yee", dty), ("0.92", 0.92 * dx)):
            if s != "Lehe" and lab != "yee":
                continue
            vp, vg = D.velocities(s, dx, dy, dt)
            pts.append(dict(ppl=ppl, dt=lab, one_minus_vg=1 - vg, one_minus_vphi=1 - vp))
    data["theory"][s] = pts
for lab, ratio, nd in (("2D, Δy = 2Δx (this case)", 2, 2), ("2D, Δy = 8Δx", 8, 2), ("3D, Δy = Δz = 2Δx", 2, 3)):
    data["regimes"][lab] = [dict(ppl=p, yee=R.yee_error(p, ratio, nd)[0], m4=(R.m4_error(p, ratio) if nd == 2 else None)) for p in range(12, 161, 4)]
json.dump(data, open(OUT, "w"))
print("wrote", OUT, os.path.getsize(OUT), "bytes")
for k, v in data["smilei"].items():
    for r, rows in sorted(v.items(), key=lambda x: int(x[0][1:].split("x")[0])):
        J = [x["J"] for x in rows]
        w = rows[0]["wake"]
        print(f"{k:10s} {r:7s} J={np.mean(J):.5f}±{(np.std(J, ddof=1) if len(J) > 1 else 0):.5f} (n={len(J)}) zback={rows[0]['zback']}")
for r, p in data["pm"].items():
    print("PM", r, round(p["J"], 5), p["zback"])
