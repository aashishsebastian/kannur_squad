"""Python port of signmath.js (tracker -> features -> resampled sequence -> model vector).
Must stay numerically identical to the JavaScript used in the app; tested by the parity check."""
import math
import numpy as np

N = 24
RAISED = 1.5
DROP_BELOW = 2.0
OUTSIDE_X, OUTSIDE_UP = 3.2, 3.5
FEAT_VERSION, FEAT_PER_HAND, FEAT_DIM = 1, 82, 164
CHAINS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]


def _hypot(*a):
    return math.sqrt(sum(x * x for x in a))


def hand_angles(shape):
    def P(j):
        return (0.0, 0.0, 0.0) if j == 0 else (shape[(j-1)*3], shape[(j-1)*3+1], shape[(j-1)*3+2] * 0.5)
    def sub(a, b):
        return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
    def ang(u, v):
        d = u[0]*v[0] + u[1]*v[1] + u[2]*v[2]
        n = _hypot(*u) * _hypot(*v) or 1.0
        return math.acos(max(-1.0, min(1.0, d / n)))
    out = []
    for c in CHAINS:
        pts = [P(j) for j in [0] + c]
        for i in range(1, 4):
            out.append(ang(sub(pts[i], pts[i-1]), sub(pts[i+1], pts[i])))
    for i in range(4):
        a, b = CHAINS[i], CHAINS[i+1]
        out.append(ang(sub(P(a[3]), P(a[0])), sub(P(b[3]), P(b[0]))))
    return out


class Tracker:
    """hands: list of dicts {label, score, lm: [(x,y,z)*21]} in normalised image coords; pose: dict {11:(x,y,vis), 12:(x,y,vis)} or None."""
    def __init__(self):
        self.body = dict(cx=0.0, cy=0.0, s=0.0, ok=False, seen=False)
        self.prev = dict(R=None, L=None, tR=-1e9, tL=-1e9)

    def _update_body(self, pose, W, H):
        b = self.body
        a = pose.get(11) if pose else None
        c = pose.get(12) if pose else None
        if a and c and a[2] > 0.5 and c[2] > 0.5:
            cx, cy = (a[0] + c[0]) / 2 * W, (a[1] + c[1]) / 2 * H
            s = _hypot((a[0] - c[0]) * W, (a[1] - c[1]) * H)
            if s > 0.08 * W and s < 0.9 * W and (not b['seen'] or abs(s / b['s'] - 1) < 0.6 or not b['ok']):
                k = 0.35 if b['seen'] else 1.0
                b['cx'] += (cx - b['cx']) * k; b['cy'] += (cy - b['cy']) * k; b['s'] += (s - b['s']) * k
                b['ok'] = True; b['seen'] = True
                return
        b['ok'] = False
        if not b['seen']:
            b['cx'], b['cy'], b['s'] = W / 2, H * 0.62, W * 0.3

    def _hand_feat(self, lm, W, H):
        b = self.body
        w = lm[0]
        size = _hypot((lm[9][0] - w[0]) * W, (lm[9][1] - w[1]) * H) or 1.0
        shape = []
        for j in range(1, 21):
            shape += [(lm[j][0] - w[0]) * W / size, (lm[j][1] - w[1]) * H / size, (lm[j][2] - w[2]) * W / size]
        return dict(loc=[(w[0] * W - b['cx']) / b['s'], (w[1] * H - b['cy']) / b['s']], shape=shape, ang=hand_angles(shape))

    def frame(self, hands, pose, W, H, t):
        self._update_body(pose, W, H)
        f = dict(t=t, R=None, L=None)
        c = [dict(h=self._hand_feat(h['lm'], W, H), label=h.get('label') or '', score=h.get('score', 0.5)) for h in (hands or [])]
        c = [x for x in c if abs(x['h']['loc'][0]) < OUTSIDE_X and x['h']['loc'][1] > -OUTSIDE_UP and x['h']['loc'][1] < DROP_BELOW]
        if len(c) == 2 and _hypot(c[0]['h']['loc'][0] - c[1]['h']['loc'][0], c[0]['h']['loc'][1] - c[1]['h']['loc'][1]) < 0.12:
            c = [c[0] if c[0]['score'] >= c[1]['score'] else c[1]]
        def label_side(x):
            if x['label'] == 'Left': return 'R'
            if x['label'] == 'Right': return 'L'
            return 'R' if x['h']['loc'][0] < 0 else 'L'
        p = self.prev
        pR = p['R'] if t - p['tR'] < 400 else None
        pL = p['L'] if t - p['tL'] < 400 else None
        def d(h, q):
            return None if q is None else _hypot(h['loc'][0] - q['loc'][0], h['loc'][1] - q['loc'][1])
        if len(c) == 2:
            a, b = c
            if pR is not None and pL is not None:
                if d(a['h'], pR) + d(b['h'], pL) > d(a['h'], pL) + d(b['h'], pR): a, b = b, a
            elif label_side(a) == label_side(b):
                if a['h']['loc'][0] > b['h']['loc'][0]: a, b = b, a
            elif label_side(a) == 'L':
                a, b = b, a
            f['R'], f['L'] = a['h'], b['h']
        elif len(c) == 1:
            x = c[0]; dR, dL = d(x['h'], pR), d(x['h'], pL)
            if dR is not None and dL is not None: side = 'R' if dR <= dL else 'L'
            elif dR is not None and dR < 0.8: side = 'R'
            elif dL is not None and dL < 0.8: side = 'L'
            else: side = label_side(x)
            f[side] = x['h']
        if f['R'] is not None: p['R'], p['tR'] = f['R'], t
        if f['L'] is not None: p['L'], p['tL'] = f['L'], t
        return f


def has_hand(f): return f['R'] is not None or f['L'] is not None
def is_raised(f): return any(f[k] is not None and f[k]['loc'][1] < RAISED for k in ('R', 'L'))


def _lerp_hand(a, b, k):
    if a is None or b is None:
        return (a if a is not None else b) if k < 0.5 else (b if b is not None else a)
    return dict(loc=[a['loc'][i] + (b['loc'][i] - a['loc'][i]) * k for i in range(2)],
                shape=[v + (b['shape'][i] - v) * k for i, v in enumerate(a['shape'])],
                ang=[v + (b['ang'][i] - v) * k for i, v in enumerate(a['ang'])])


def to_sample(frames, n=N):
    def first_last(pred):
        idx = [i for i, f in enumerate(frames) if pred(f)]
        return (idx[0], idx[-1]) if idx else None
    span = first_last(is_raised) or first_last(has_hand)
    if not span: return None
    seg = frames[span[0]:span[1] + 1]
    if sum(1 for f in seg if has_hand(f)) < 4: return None
    t0, t1 = seg[0]['t'], seg[-1]['t']
    out = []
    for i in range(n):
        tt = t0 + (t1 - t0) * i / (n - 1)
        j = 0
        while j < len(seg) - 2 and seg[j + 1]['t'] < tt: j += 1
        a, b = seg[j], seg[min(j + 1, len(seg) - 1)]
        k = min(1.0, max(0.0, (tt - a['t']) / (b['t'] - a['t']))) if b['t'] > a['t'] else 0.0
        out.append(dict(R=_lerp_hand(a['R'], b['R'], k), L=_lerp_hand(a['L'], b['L'], k)))
    return out


def _mirror(h):
    return None if h is None else dict(loc=[-h['loc'][0], h['loc'][1]], shape=[-v if i % 3 == 0 else v for i, v in enumerate(h['shape'])], ang=h['ang'])


def to_vector(seq):
    out = np.zeros((len(seq), FEAT_DIM), dtype=np.float32)
    for t, f in enumerate(seq):
        for side, k in enumerate(('R', 'L')):
            h = f[k]
            if h is None: continue
            o = side * FEAT_PER_HAND
            out[t, o] = 1; out[t, o+1:o+3] = h['loc']; out[t, o+3:o+63] = h['shape']; out[t, o+63:o+82] = h['ang']
    return out


def features_from_raw(raw_frames, n):
    """raw_frames: list of {t, W, H, hands:[{label,score,lm}], pose:{11:(x,y,v),12:(x,y,v)}|None} -> [n x 164] or None"""
    tr = Tracker()
    frames = [tr.frame(r['hands'], r['pose'], r['W'], r['H'], r['t']) for r in raw_frames]
    seq = to_sample(frames, n)
    return None if seq is None else to_vector(seq)
