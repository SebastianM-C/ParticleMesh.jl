"""Copy a run directory for publishing: text files as they are, HDF5 outputs repacked as float32
with gzip (Probes0.h5, Fields0.h5 of vacuum runs, tracks and binning kept in float64), plasma runs'
Fields0.h5 left out (only the vacuum runs' t = 100 dump is used)."""
import h5py, numpy as np, os, shutil, sys

def repack(src, dst, f32=True):
    with h5py.File(src) as a, h5py.File(dst, "w") as b:
        for k, v in a.attrs.items():
            b.attrs[k] = v
        def visit(name, obj):
            if isinstance(obj, h5py.Group):
                g = b.require_group(name)
                for k, v in obj.attrs.items(): g.attrs[k] = v
            else:
                data = obj[()]
                if f32 and getattr(data, "dtype", None) is not None and data.dtype == np.float64 and data.ndim >= 1 and data.size > 64:
                    data = data.astype(np.float32)
                d = b.create_dataset(name, data=data, compression="gzip" if np.ndim(data) >= 1 and np.size(data) > 64 else None)
                for k, v in obj.attrs.items(): d.attrs[k] = v
        a.visititems(visit)

def pack(run, out):
    os.makedirs(out, exist_ok=True)
    vac = os.path.basename(run).startswith("vac-")
    for f in sorted(os.listdir(run)):
        p = os.path.join(run, f)
        if f in ("info.shelf",) or (f == "Fields0.h5" and not vac):
            continue
        if f in ("Probes0.h5", "Fields0.h5"):
            repack(p, os.path.join(out, f.replace(".h5", "-f32.h5")))
        elif f.endswith(".h5"):
            repack(p, os.path.join(out, f), f32=False)
        else:
            shutil.copy2(p, out)

if __name__ == "__main__":
    dst = sys.argv[1]
    for r in sys.argv[2:]:
        pack(r, os.path.join(dst, "lowdisp-" + os.path.basename(r.rstrip("/"))))
