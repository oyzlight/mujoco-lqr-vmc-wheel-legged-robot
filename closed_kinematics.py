"""Two-input, two-loop linkage, measured from the corrected native CAD assembly.

All planar vectors are [x,z], metres. Positive joint rotation is about robot +Y.
q=[A output, B output]; actual motor-module output angles are -q (external 1:1).
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parent
POINTS = {k: np.asarray(v, float) for k, v in json.loads(
    (ROOT/'mechanism_analysis/geometry_corrected_preview.json').read_text(encoding='utf-8'))['points_m'].items()}
DROT = np.array([[0., 1.], [-1., 0.]])


def rot(q):
    c, s = np.cos(q), np.sin(q)
    return np.array([[c, s], [-s, c]])


def wrap(q):
    return np.arctan2(np.sin(q), np.cos(q))


def cross(a, b):
    return a[0]*b[1]-a[1]*b[0]


def relative_angle(a, b):
    return np.arctan2(-cross(a, b), np.dot(a, b))


def intersection(a, b, ra, rb, branch):
    v = b-a
    length = np.linalg.norm(v)
    if length < 1e-9 or length > ra+rb-1e-9 or length < abs(ra-rb)+1e-9:
        raise ValueError('Linkage unreachable or at a circle-intersection singularity')
    u = v/length
    x = (length**2+ra**2-rb**2)/(2*length)
    h = np.sqrt(max(0., ra**2-x**2))
    return a+x*u+branch*h*np.array([-u[1], u[0]])


class ClosedLeg:
    def __init__(self):
        self.p = {k: v-POINTS['O'] for k, v in POINTS.items()}
        p = self.p
        self.sc, self.qc = np.linalg.norm(p['C']-p['S']), np.linalg.norm(p['C']-p['Q'])
        self.ed, self.kd = np.linalg.norm(p['D']-p['E']), np.linalg.norm(p['D']-p['K'])
        self.branch_c = np.sign(cross(p['Q']-p['S'], p['C']-p['S']))
        self.branch_d = np.sign(cross(p['K']-p['E'], p['D']-p['E']))

    def forward(self, q, jacobian=False):
        p = self.p
        a, b = q
        o = np.zeros(2)
        s, qpt, k = rot(a)@p['S'], rot(b)@p['Q'], rot(b)@p['K']
        c = intersection(s, qpt, self.sc, self.qc, self.branch_c)
        rocker = relative_angle(p['C']-p['Q'], c-qpt)
        e = qpt+rot(rocker)@(p['E']-p['Q'])
        d = intersection(e, k, self.ed, self.kd, self.branch_d)
        lower = relative_angle(p['D']-p['K'], d-k)
        w = k+rot(lower)@(p['W']-p['K'])
        rod29 = relative_angle(p['C']-p['S'], c-s)
        rod35 = relative_angle(p['D']-p['E'], d-e)
        result = dict(O=o, S=s, Q=qpt, K=k, C=c, E=e, D=d, W=w,
                      angles=np.array([a, b, rocker-b, rod29-a, lower-b, rod35-rocker]))
        if jacobian:
            ds, dq, dk = np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((2, 2))
            ds[:, 0], dq[:, 1], dk[:, 1] = DROT@s, DROT@qpt, DROT@k
            dc = np.linalg.solve(np.vstack([c-s, c-qpt]), np.vstack([(c-s)@ds, (c-qpt)@dq]))
            dr = ((c-qpt)[1]*(dc-dq)[0]-(c-qpt)[0]*(dc-dq)[1])/self.qc**2
            de = dq+np.outer(DROT@(e-qpt), dr)
            dd = np.linalg.solve(np.vstack([d-e, d-k]), np.vstack([(d-e)@de, (d-k)@dk]))
            dl = ((d-k)[1]*(dd-dk)[0]-(d-k)[0]*(dd-dk)[1])/self.kd**2
            result['Jcart'] = dk+np.outer(DROT@(w-k), dl)
        return result

    def polar(self, q):
        f = self.forward(q, True)
        r, j = f['W'], f['Jcart']
        length = np.linalg.norm(r)
        beta = np.arctan2(-r[0], -r[1])
        return length, beta, np.vstack([r@j/length, np.array([r[1], -r[0]])@j/length**2])

    def ik(self, length, beta, seed=None):
        target = np.array([-length*np.sin(beta), -length*np.cos(beta)])
        def residual(q):
            try:
                return self.forward(q)['W']-target
            except ValueError:
                return np.ones(2)*10
        seeds = [seed] if seed is not None else [np.array([a, b]) for a in [-2., -1., 0., 1., 2.] for b in [-2., -1., 0., 1., 2.]]
        for start in seeds:
            fit = least_squares(residual, start, max_nfev=150, diff_step=1e-5, xtol=1e-12, ftol=1e-12, gtol=1e-12)
            if np.linalg.norm(residual(fit.x)) < 1e-7:
                return wrap(fit.x)
        raise ValueError(f'No regular solution on the corrected assembly branch: L={length}, beta={beta}')
