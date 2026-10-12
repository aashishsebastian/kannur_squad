// Pure sign-recognition maths (no DOM): landmark frames -> features -> fixed-length sequences -> DTW few-shot classifier.
// Used by camera.js in the browser and by the offline tests, so both run exactly the same code.

export const N = 24;            // every sign is resampled to this many frames
export const MIN_SAMPLES = 3;   // recordings needed before a sign can be recognised
export const RAISED = 1.5;      // a hand counts as "signing" when its wrist is less than this many shoulder-widths below the shoulder line
const DROP_BELOW = 2.0;         // hands lower than this (resting in lap / at the hip) are ignored
const OUTSIDE_X = 3.2, OUTSIDE_UP = 3.5;   // detections far outside the signing space are treated as false positives

/* ---------- per-frame tracking ---------- */
// Keeps the body frame of reference (centre between shoulders, shoulder width) and follows each hand over time so that
// a momentary left/right label flip from the detector doesn't swap the hands.
export function createTracker(){
  const body = { cx: 0, cy: 0, s: 0, ok: false, seen: false };
  let prev = { R: null, L: null, tR: -1e9, tL: -1e9 };
  function updateBody(poseLm, W, H){
    const a = poseLm && poseLm[11], b = poseLm && poseLm[12];
    if (a && b && (a.visibility ?? 1) > 0.5 && (b.visibility ?? 1) > 0.5) {
      const cx = (a.x + b.x) / 2 * W, cy = (a.y + b.y) / 2 * H, s = Math.hypot((a.x - b.x) * W, (a.y - b.y) * H);
      // ignore implausible shoulder widths (too small/large for the picture, or a sudden jump)
      if (s > 0.08 * W && s < 0.9 * W && (!body.seen || Math.abs(s / body.s - 1) < 0.6 || !body.ok)) { const k = body.seen ? 0.35 : 1; body.cx += (cx - body.cx) * k; body.cy += (cy - body.cy) * k; body.s += (s - body.s) * k; body.ok = true; body.seen = true; return; }
    }
    body.ok = false;
    if (!body.seen) { body.cx = W / 2; body.cy = H * 0.62; body.s = W * 0.3; }
  }
  function handFeat(lm, W, H){
    const w = lm[0], size = Math.hypot((lm[9].x - w.x) * W, (lm[9].y - w.y) * H) || 1, shape = new Array(60);
    for (let j = 1; j <= 20; j++) { const k = (j - 1) * 3; shape[k] = (lm[j].x - w.x) * W / size; shape[k+1] = (lm[j].y - w.y) * H / size; shape[k+2] = (lm[j].z - w.z) * W / size; }
    return { loc: [(w.x * W - body.cx) / body.s, (w.y * H - body.cy) / body.s], shape, ang: handAngles(shape) };
  }
  // hands: MediaPipe HandLandmarker result; poseLm: 33 pose landmarks (or null)
  function frame(hands, poseLm, W, H, t){
    updateBody(poseLm, W, H);
    const f = { t, R: null, L: null };
    let c = (hands?.landmarks || []).map((lm, i) => ({ h: handFeat(lm, W, H), label: hands.handedness?.[i]?.[0]?.categoryName || '', score: hands.handedness?.[i]?.[0]?.score ?? 0.5 }));
    c = c.filter(x => Math.abs(x.h.loc[0]) < OUTSIDE_X && x.h.loc[1] > -OUTSIDE_UP && x.h.loc[1] < DROP_BELOW);
    if (c.length === 2 && Math.hypot(c[0].h.loc[0] - c[1].h.loc[0], c[0].h.loc[1] - c[1].h.loc[1]) < 0.12) c = [c[0].score >= c[1].score ? c[0] : c[1]];
    // MediaPipe labels assume a mirrored selfie image; our frames are not mirrored, so its "Left" is the signer's right hand.
    const labelSide = x => x.label === 'Left' ? 'R' : x.label === 'Right' ? 'L' : (x.h.loc[0] < 0 ? 'R' : 'L');
    const pR = t - prev.tR < 400 ? prev.R : null, pL = t - prev.tL < 400 ? prev.L : null;
    const d = (h, p) => p ? Math.hypot(h.loc[0] - p.loc[0], h.loc[1] - p.loc[1]) : null;
    if (c.length === 2) {
      let a = c[0], b = c[1];
      if (pR && pL) { if (d(a.h, pR) + d(b.h, pL) > d(a.h, pL) + d(b.h, pR)) [a, b] = [b, a]; }
      else if (labelSide(a) === labelSide(b)) { if (a.h.loc[0] > b.h.loc[0]) [a, b] = [b, a]; }   // same label: signer's right hand is on image-left
      else if (labelSide(a) === 'L') [a, b] = [b, a];
      f.R = a.h; f.L = b.h;
    } else if (c.length === 1) {
      const x = c[0], dR = d(x.h, pR), dL = d(x.h, pL);
      let side;
      if (dR != null && dL != null) side = dR <= dL ? 'R' : 'L';
      else if (dR != null && dR < 0.8) side = 'R';
      else if (dL != null && dL < 0.8) side = 'L';
      else side = labelSide(x);
      f[side] = x.h;
    }
    if (f.R) { prev.R = f.R; prev.tR = t; } if (f.L) { prev.L = f.L; prev.tL = t; }
    return f;
  }
  return { body, frame, reset(){ prev = { R: null, L: null, tR: -1e9, tL: -1e9 }; } };
}

// Handshape descriptor that ignores hand size and rotation: bend at each finger joint (15) + spread between neighbouring fingers (4)
const CHAINS = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16],[17,18,19,20]];
function handAngles(shape){
  const P = j => j === 0 ? [0, 0, 0] : [shape[(j-1)*3], shape[(j-1)*3+1], shape[(j-1)*3+2] * 0.5];
  const sub = (a, b) => [a[0]-b[0], a[1]-b[1], a[2]-b[2]];
  const ang = (u, v) => { const d = u[0]*v[0] + u[1]*v[1] + u[2]*v[2], n = Math.hypot(...u) * Math.hypot(...v) || 1; return Math.acos(Math.max(-1, Math.min(1, d / n))); };
  const out = [];
  for (const c of CHAINS) { const pts = [0, ...c].map(P); for (let i = 1; i < 4; i++) out.push(ang(sub(pts[i], pts[i-1]), sub(pts[i+1], pts[i]))); }
  for (let i = 0; i < 4; i++) { const a = CHAINS[i], b = CHAINS[i+1]; out.push(ang(sub(P(a[3]), P(a[0])), sub(P(b[3]), P(b[0])))); }
  return out;
}

/* ---------- sequences ---------- */
export const hasHand = f => !!(f.R || f.L);
export const isRaised = f => ['R', 'L'].some(k => f[k] && f[k].loc[1] < RAISED);
function lerpHand(a, b, k){
  if (!a || !b) return k < 0.5 ? (a || b) : (b || a);
  return { loc: [a.loc[0] + (b.loc[0] - a.loc[0]) * k, a.loc[1] + (b.loc[1] - a.loc[1]) * k], shape: a.shape.map((v, i) => v + (b.shape[i] - v) * k), ang: a.ang.map((v, i) => v + (b.ang[i] - v) * k) };
}
// Keep only the part where the hands are up and signing, then resample to N frames over time.
export function toSample(frames, n = N){
  const firstLast = pred => { const i0 = frames.findIndex(pred); if (i0 < 0) return null; let i1 = frames.length - 1; while (!pred(frames[i1])) i1--; return [i0, i1]; };
  const span = firstLast(isRaised) || firstLast(hasHand);
  if (!span) return { ok: false, reason: 'no_hands' };
  const seg = frames.slice(span[0], span[1] + 1), handFrames = seg.filter(hasHand).length;
  if (handFrames < 4) return { ok: false, reason: 'too_short' };
  const t0 = seg[0].t, t1 = seg[seg.length - 1].t, out = [];
  for (let i = 0; i < n; i++) {
    const tt = t0 + (t1 - t0) * i / (n - 1);
    let j = 0; while (j < seg.length - 2 && seg[j + 1].t < tt) j++;
    const a = seg[j], b = seg[Math.min(j + 1, seg.length - 1)], k = b.t > a.t ? Math.min(1, Math.max(0, (tt - a.t) / (b.t - a.t))) : 0;
    out.push({ R: lerpHand(a.R, b.R, k), L: lerpHand(a.L, b.L, k) });
  }
  return { ok: true, seq: out, duration: (t1 - t0) / 1000, coverage: handFrames / seg.length };
}
// ---------- model input (must match islfeat.py exactly) ----------
export const FEAT_VERSION = 1, FEAT_PER_HAND = 82, FEAT_DIM = 164;
// sequence -> Float32Array [T x 164]: per hand (signer's right, then left): present, loc(2), shape(60), angles(19)
// (left-handed signing is handled by the model, which is trained on mirrored copies of every video)
export function toVector(seq){
  const out = new Float32Array(seq.length * FEAT_DIM);
  seq.forEach((f, t) => ['R', 'L'].forEach((k, side) => {
    const h = f[k]; if (!h) return; const o = t * FEAT_DIM + side * FEAT_PER_HAND;
    out[o] = 1; out[o+1] = h.loc[0]; out[o+2] = h.loc[1];
    for (let i = 0; i < 60; i++) out[o+3+i] = h.shape[i];
    for (let i = 0; i < 19; i++) out[o+63+i] = h.ang[i];
  }));
  return out;
}
const r3 = x => Math.round(x * 1000) / 1000;
const packHand = h => h && { loc: h.loc.map(r3), shape: h.shape.map(r3), ang: h.ang.map(r3) };
export const packSeq = seq => seq.map(f => ({ R: packHand(f.R), L: packHand(f.L) }));

/* ---------- distances ---------- */
export const W = { loc: 0.5, shape: 2, z: 0.5, ang: 4, miss: 1.6, swapPen: 0.15 };   // tuned on synthetic tracker data (see tests)
function handDist(x, y){
  if (!x && !y) return 0;
  if (!x || !y) return W.miss;                      // one hand used vs. not used
  const dl = Math.hypot(x.loc[0] - y.loc[0], x.loc[1] - y.loc[1]);           // where the hand is
  let ds = 0;                                                                   // handshape + orientation
  for (let k = 0; k < 60; k += 3) ds += Math.hypot(x.shape[k] - y.shape[k], x.shape[k+1] - y.shape[k+1], W.z * (x.shape[k+2] - y.shape[k+2]));
  let da = 0; if (W.ang) { for (let k = 0; k < 19; k++) da += Math.abs(x.ang[k] - y.ang[k]); da /= 19; }
  return W.loc * dl + W.shape * ds / 20 + W.ang * da;
}
const only = f => (f.R ? 1 : 0) + (f.L ? 1 : 0) === 1 ? (f.R || f.L) : null;
export function frameDist(a, b){
  const ha = only(a), hb = only(b);
  // one hand in each frame: compare those hands even if left/right labels disagree (robust to label flips and left-handed signers)
  if (ha && hb) return handDist(ha, hb) + (!!a.R !== !!b.R ? W.swapPen : 0);
  return handDist(a.R, b.R) + handDist(a.L, b.L);
}
export function dtw(A, B, r = 5){
  const n = A.length, m = B.length, INF = 1e9;
  let prev = new Float64Array(m + 1).fill(INF), cur = new Float64Array(m + 1).fill(INF); prev[0] = 0;
  for (let i = 1; i <= n; i++) {
    cur.fill(INF);
    for (let j = Math.max(1, i - r); j <= Math.min(m, i + r); j++) cur[j] = frameDist(A[i-1], B[j-1]) + Math.min(prev[j], cur[j-1], prev[j-1]);
    [prev, cur] = [cur, prev];
  }
  return prev[m] / (n + m);
}

/* ---------- classifier ---------- */
export const CONF_T = 0.08;    // softmax temperature as a fraction of the typical within-sign distance (calibrated on test data)
export function activeLabels(samples){ return Object.keys(samples).filter(l => samples[l].length >= MIN_SAMPLES); }
export function classScores(samples, seq, exclude){
  const out = [];
  for (const l of activeLabels(samples)) {
    const ds = samples[l].filter(s => s !== exclude).map(s => dtw(seq, s.seq)).sort((a, b) => a - b);
    if (ds.length) out.push({ label: l, d: ds[0] });
  }
  return out.sort((a, b) => a.d - b.d);
}
// typical distance between a sample and its nearest same-sign neighbour
export function computeStats(samples){
  const nn = [];
  for (const l of activeLabels(samples)) for (const s of samples[l]) {
    const ds = samples[l].filter(o => o !== s).map(o => dtw(s.seq, o.seq));
    if (ds.length) nn.push(Math.min(...ds));
  }
  nn.sort((a, b) => a - b);
  return nn.length ? { tau: Math.max(0.05, nn[nn.length >> 1]), n: nn.length } : null;
}
// strict: reject threshold multiplier (lower = stricter)
export function classify(samples, stats, seq, { strict = 1.5, exclude } = {}){
  if (activeLabels(samples).length < 2) return { status: 'not_ready' };
  const tau = stats ? stats.tau : 0.3, scores = classScores(samples, seq, exclude);
  const ex = scores.map(s => Math.exp(-(s.d - scores[0].d) / (CONF_T * tau))), Z = ex.reduce((a, b) => a + b, 0);
  const cands = scores.slice(0, 3).map((s, i) => ({ label: s.label, d: s.d, conf: ex[i] / Z }));
  let status = 'needs_confirmation';
  if (cands[0].d > strict * tau) status = 'unsupported';
  else if (cands[0].conf < 0.6) status = 'uncertain_repeat';
  return { status, cands, tau };
}
export function leaveOneOut(samples, stats){
  let right = 0, total = 0; const confusions = {};
  for (const l of activeLabels(samples)) for (const s of samples[l]) {
    const r = classify(samples, stats, s.seq, { exclude: s, strict: 1e9 }); if (!r.cands) continue;
    total++;
    if (r.cands[0].label === l) right++; else { const k = l + ' → ' + r.cands[0].label; confusions[k] = (confusions[k] || 0) + 1; }
  }
  return { total, right, confusions };
}
