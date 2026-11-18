"""Summary of one Smilei run of the case: the objective J as ParticleMesh's trapped_energy
(scripts/optim/trapped2d_case.jl: sum w (gamma - 1) S(x_window; 12, 24) R(u_x; 3, 5), window x
counted from ParticleMesh's box back, floor((t - 108)/dx) cells), the born electrons, and from the
axis probe the laser (centroid of Ey^2, Hilbert envelope peak) and the wake (zero crossings of Ex)
over time.

    python3.12 analyze.py <rundir>...   -> <rundir>/summary.json
"""
import h5py, json, math, sys, numpy as np

def smooth_rise(x, lo, hi):
    s = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return s**3 * (10 - 15 * s + 6 * s * s)

def tracks(path, species):
    try:
        f = h5py.File(path)
    except OSError:
        return None
    k = sorted(f["data"].keys())[-1]
    g = f["data"][k]["particles"][species]
    return {a: np.array(g[b][c]) if c else np.array(g[b]) for a, b, c in
            (("x", "position", "x"), ("y", "position", "y"), ("px", "momentum", "x"), ("py", "momentum", "y"),
             ("pz", "momentum", "z"), ("w", "weight", None))}, int(k)

def hilbert_env(a):
    n = a.size
    A = np.fft.fft(a)
    h = np.zeros(n); h[0] = 1
    if n % 2 == 0:
        h[n // 2] = 1; h[1:n // 2] = 2
    else:
        h[1:(n + 1) // 2] = 2
    return np.abs(np.fft.ifft(A * h))

def crossings(x, f):
    s = np.sign(f)
    i = np.where(s[:-1] * s[1:] < 0)[0]
    return x[i] - f[i] * (x[i + 1] - x[i]) / (f[i + 1] - f[i])

def probe_series(run, m):
    f = h5py.File(f"{run}/Probes0.h5")
    pos = np.array(f["positions"])[:, 0]
    out = []
    for k in sorted(kk for kk in f.keys() if kk.isdigit()):
        d = f[k]
        step = int(k)
        t = step * m["dt"]
        xm = float(d.attrs["x_moved"])
        Ex, Ey, Bz, Rho = (np.array(d[i]) for i in range(4))
        I = Ey**2
        tot = I.sum()
        if tot <= 0:
            continue
        xc = float((pos * I).sum() / tot)  # window-relative probe coordinates
        env = hilbert_env(Ey)
        ip = int(np.argmax(env))
        # ParticleMesh's window back in lab coordinates at this step
        shift = max(0, math.floor((t - 108.0) / m["dx"])) * m["dx"]
        # wake: Ex crossings behind the laser (window coords of ParticleMesh's box)
        xl = pos + xm - shift
        z = crossings(xl, np.convolve(Ex, np.ones(4) / 4, mode="same"))
        z = z[(z < xc + xm - shift - 4) & (z > 1.0)]
        out.append(dict(step=step, t=t, x_moved=xm, pm_back=shift, centroid_lab=xc + xm, a_peak=float(env[ip]),
                        a_peak_lab=float(pos[ip] + xm), Ex_zeros=sorted(z.tolist(), reverse=True)[:6],
                        Ex_max=float(Ex.max()), Ex_min=float(Ex.min()), Ey_absmax=float(np.abs(Ey).max())))
    return out

def objective(P, back, fade=(12.0, 24.0), rise=(3.0, 5.0)):
    if P is None or P["w"].size == 0:
        return dict(J=0.0, weight=0.0, n=0, J_n=0.0)
    xw = P["x"] - back
    g = np.sqrt(1 + P["px"]**2 + P["py"]**2 + P["pz"]**2)
    c = P["w"] * (g - 1) * smooth_rise(xw, *fade) * smooth_rise(P["px"], *rise)
    trapped = c > 0
    return dict(J=float(c.sum()), weight=float(P["w"].sum()), n=int(P["w"].size), n_trapped=int(trapped.sum()),
                weight_trapped=float((P["w"] * smooth_rise(xw, *fade) * smooth_rise(P["px"], *rise)).sum()),
                mean_energy_trapped=float(((g - 1) * c).sum() / c.sum()) if c.sum() > 0 else 0.0,
                mean_x_trapped=float((xw * c).sum() / c.sum()) if c.sum() > 0 else 0.0,
                ux_max=float(P["px"].max()), neff=float(c.sum()**2 / (c**2).sum()) if c.sum() > 0 else 0.0)

def summarize(run):
    m = json.load(open(f"{run}/meta.json"))
    out = dict(meta=m)
    out["series"] = probe_series(run, m)
    if not m["vacuum"]:
        tb = tracks(f"{run}/TrackParticlesDisordered_born.h5", "born")
        te = tracks(f"{run}/TrackParticlesDisordered_electron.h5", "electron")
        step = (tb or te)[1]
        t = step * m["dt"]
        back = max(0, math.floor((t - 108.0) / m["dx"])) * m["dx"]
        ob = objective(tb[0] if tb else None, back)
        oe = objective(te[0] if te else None, back)
        out.update(t_end=t, pm_back=back, J=ob["J"] + oe["J"], J_born=ob["J"], J_plasma=oe["J"], born=ob, plasma=oe)
        if tb:
            P = tb[0]; xw = P["x"] - back
            g = np.sqrt(1 + P["px"]**2 + P["py"]**2 + P["pz"]**2) - 1
            sel = smooth_rise(xw, 12, 24) * smooth_rise(P["px"], 3, 5)
            h, e = np.histogram(g, bins=60, range=(0, 30), weights=P["w"] * sel)
            out["spectrum"] = dict(edges=e.tolist(), weight=h.tolist())
            hx, ex = np.histogram(xw, bins=88, range=(0, 88), weights=P["w"] * smooth_rise(P["px"], 3, 5))
            out["trapped_x"] = dict(edges=ex.tolist(), weight=hx.tolist())
    json.dump(out, open(f"{run}/summary.json", "w"))
    return out

if __name__ == "__main__":
    for r in sys.argv[1:]:
        s = summarize(r)
        last = s["series"][-1]
        print(r.rstrip("/").split("/")[-1], "J=%.5f" % s.get("J", 0), "Jb=%.5f" % s.get("J_born", 0), "Jp=%.5f" % s.get("J_plasma", 0),
              "nborn=%d" % s.get("born", {}).get("n", 0), "ntrap=%d" % s.get("born", {}).get("n_trapped", 0),
              "neff=%.0f" % s.get("born", {}).get("neff", 0),
              "zeros", [round(z, 2) for z in last["Ex_zeros"][:4]], "apk=%.4f" % last["a_peak"], "cen=%.3f" % (last["centroid_lab"]))
