"""Vacuum calibration: Ey energy at t = 100 against the case's pi a0^2 w tau / 8; prints the a0 scale."""
import h5py, json, numpy as np, glob, math, sys
def calib(r):
    m=json.load(open(r+'/meta.json')); g=h5py.File(r+'/Fields0.h5'); k=list(g['data'].keys())[0]
    a=np.array(g['data'][k]['Ey']); U=0.5*np.sum(a**2)*m['dx']*m['dy']
    T=math.pi*1.9**2*10*6/8
    return m, U/T, m['a0scale']*math.sqrt(T/U)
if __name__ == "__main__":
    out = {}
    for r in sys.argv[1:]:
        m, ratio, s = calib(r)
        rows=[l.split() for l in open(r+'/scalars.txt') if l.strip() and not l.startswith('#')]
        print(r.split('/')[-1], 'dt/dx=%.3f'%(m['dt']/m['dx']), 'UEy/T=%.5f'%ratio, 'scale=%.5f'%s, 'Uelm t100..end', rows[len(rows)//3][8], rows[-1][8])
