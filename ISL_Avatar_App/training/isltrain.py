"""ISL sign model training: videos -> MediaPipe landmarks -> augmented feature sequences -> 1D-CNN -> ONNX.
Used by the Colab notebook (train_isl_model.ipynb). Feature maths lives in islfeat.py (identical to the app's signmath.js)."""
import gzip, json, math, os, re, random, time, zipfile
from collections import Counter, defaultdict
import numpy as np
import islfeat as F

VIDEO_EXT = ('.mov', '.mp4', '.avi', '.mkv', '.m4v', '.webm')
T = 32                       # frames per sign given to the model (app resamples to the same length)

# ----------------------------------------------------------------------------- landmark extraction
_HL = _PL = None
_TS = 0


def _init_worker(hand_model, pose_model):
    """Create one hand + pose landmarker per worker process (VIDEO mode)."""
    global _HL, _PL
    from mediapipe.tasks import python as mpt
    from mediapipe.tasks.python import vision
    _HL = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=hand_model), running_mode=vision.RunningMode.VIDEO,
        num_hands=2, min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5))
    _PL = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mpt.BaseOptions(model_asset_path=pose_model), running_mode=vision.RunningMode.VIDEO, num_poses=1))


def extract_video(path, max_width=640, target_fps=15):
    """Run hand + pose tracking on one video. Returns list of raw frames (same format the app produces)."""
    global _TS
    import cv2, mediapipe as mp
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, round(fps / target_fps))
    frames, i = [], 0
    r4 = lambda v: round(float(v), 4)
    while True:
        ok, img = cap.read()
        if not ok: break
        if i % step == 0:
            h, w = img.shape[:2]
            if w > max_width:
                img = cv2.resize(img, (max_width, round(h * max_width / w)), interpolation=cv2.INTER_AREA); h, w = img.shape[:2]
            rgb = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            _TS += 40                                     # MediaPipe needs strictly increasing timestamps (ms)
            im = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            hr, pr = _HL.detect_for_video(im, _TS), _PL.detect_for_video(im, _TS)
            hands = [dict(label=hr.handedness[k][0].category_name, score=r4(hr.handedness[k][0].score),
                          lm=[[r4(p.x), r4(p.y), r4(p.z)] for p in lm]) for k, lm in enumerate(hr.hand_landmarks)]
            pose = None
            if pr.pose_landmarks:
                pl = pr.pose_landmarks[0]
                pose = {11: [r4(pl[11].x), r4(pl[11].y), r4(pl[11].visibility or 1)], 12: [r4(pl[12].x), r4(pl[12].y), r4(pl[12].visibility or 1)]}
            frames.append(dict(t=round(i * 1000 / fps, 1), W=w, H=h, hands=hands, pose=pose))
        i += 1
    cap.release()
    return frames


def _extract_job(args):
    path, out_path, meta = args
    try:
        frames = extract_video(path)
        with gzip.open(out_path, 'wt') as f: json.dump(dict(meta, frames=frames), f)
        return out_path, len(frames), None
    except Exception as e:  # keep going; report at the end
        return out_path, 0, repr(e)


def label_from_path(path):
    """INCLUDE layout: <Category>/<NN. Word>/<video>. Label = folder name without the number prefix."""
    parent = os.path.basename(os.path.dirname(path))
    cat = os.path.basename(os.path.dirname(os.path.dirname(path)))
    label = re.sub(r'^\s*\d+\s*[.)_-]?\s*', '', parent).strip().lower()
    return label or parent.lower(), cat


def extract_folder(video_root, out_dir, hand_model, pose_model, workers=None, log=print):
    """Extract landmarks for every video under video_root into out_dir (one .json.gz per video). Skips finished ones."""
    from multiprocessing import Pool
    os.makedirs(out_dir, exist_ok=True)
    jobs = []
    for root, _, files in os.walk(video_root):
        for fn in sorted(files):
            if fn.lower().endswith(VIDEO_EXT) and not fn.startswith('._'):
                p = os.path.join(root, fn)
                label, cat = label_from_path(p)
                key = re.sub(r'[^A-Za-z0-9]+', '_', os.path.relpath(p, video_root))
                out = os.path.join(out_dir, key + '.json.gz')
                if not os.path.exists(out): jobs.append((p, out, dict(label=label, category=cat, video=os.path.relpath(p, video_root))))
    if not jobs: return 0
    workers = workers or max(1, os.cpu_count() or 1)
    t0, done, errors = time.time(), 0, []
    with Pool(workers, initializer=_init_worker, initargs=(hand_model, pose_model)) as pool:
        for out, n, err in pool.imap_unordered(_extract_job, jobs):
            done += 1
            if err: errors.append((out, err))
            if done % 25 == 0 or done == len(jobs):
                el = time.time() - t0
                log(f'  {done}/{len(jobs)} videos  ({el/done:.1f}s each, ~{el/done*(len(jobs)-done)/60:.0f} min left)')
    for out, err in errors[:5]: log('  failed:', out, err)
    return len(jobs) - len(errors)


def load_landmarks(lm_dir):
    items = []
    for fn in sorted(os.listdir(lm_dir)):
        if fn.endswith('.json.gz'):
            with gzip.open(os.path.join(lm_dir, fn), 'rt') as f: d = json.load(f)
            for fr in d['frames']:
                if fr['pose']: fr['pose'] = {int(k): v for k, v in fr['pose'].items()}
            items.append(d)
    return items


# ----------------------------------------------------------------------------- augmentation
def augment(frames, rng, mirror=None):
    """Simulate a different camera/person/take: mirror (left-handed), rotate, scale, shift, jitter, speed changes, dropped frames."""
    if mirror is None: mirror = rng.random() < 0.5
    rot, sc = rng.normal(0, 0.07), rng.uniform(0.8, 1.2)
    tx, ty = rng.normal(0, 0.05), rng.normal(0, 0.05)
    jit, fdrop, hdrop = rng.uniform(0.0, 0.004), rng.uniform(0, 0.12), rng.uniform(0, 0.06)
    c, s = math.cos(rot), math.sin(rot)
    n = len(frames)
    # random temporal crop + non-uniform speed (time warp)
    a = int(rng.integers(0, max(1, n // 10) + 1)); b = n - int(rng.integers(0, max(1, n // 10) + 1))
    sel = frames[a:max(b, a + 2)]
    speeds = rng.uniform(0.7, 1.4, size=4)
    out, t = [], 0.0
    for k, fr in enumerate(sel):
        if k and rng.random() < fdrop: continue
        W, H = fr['W'], fr['H']
        def tf(x, y, z=0.0):
            px, py = x * W - W / 2, y * H - H / 2
            if mirror: px = -px
            qx, qy = (c * px - s * py) * sc + tx * W, (s * px + c * py) * sc + ty * H
            return [(qx + W / 2) / W + rng.normal(0, jit), (qy + H / 2) / H + rng.normal(0, jit), z * sc + rng.normal(0, jit)]
        hands = []
        for h in fr['hands']:
            if rng.random() < hdrop: continue
            lab = h['label']
            if mirror: lab = {'Left': 'Right', 'Right': 'Left'}.get(lab, lab)
            hands.append(dict(label=lab, score=h['score'], lm=[tf(*p) for p in h['lm']]))
        pose = None
        if fr['pose']:
            p11, p12 = fr['pose'][11], fr['pose'][12]
            if mirror: p11, p12 = p12, p11
            q11, q12 = tf(p11[0], p11[1]), tf(p12[0], p12[1])
            pose = {11: [q11[0], q11[1], p11[2]], 12: [q12[0], q12[1], p12[2]]}
        if k: t += (sel[k]['t'] - sel[k - 1]['t']) * speeds[min(3, 4 * k // len(sel))]
        out.append(dict(t=t, W=W, H=H, hands=hands, pose=pose))
    return out


def _feat_job(args):
    frames, n_aug, seed, mirror_test = args
    rng = np.random.default_rng(seed)
    vecs = []
    v = F.features_from_raw(frames, T)
    if v is not None: vecs.append(v)
    if mirror_test:   # clean mirrored copy, used to measure left-handed performance
        v = F.features_from_raw(augment_clean_mirror(frames), T)
        return vecs, ([v] if v is not None else [])
    for _ in range(n_aug):
        v = F.features_from_raw(augment(frames, rng), T)
        if v is not None: vecs.append(v)
    return vecs, []


def augment_clean_mirror(frames):
    out = []
    for fr in frames:
        hands = [dict(label={'Left': 'Right', 'Right': 'Left'}.get(h['label'], h['label']), score=h['score'], lm=[[1 - x, y, z] for x, y, z in h['lm']]) for h in fr['hands']]
        pose = None
        if fr['pose']:
            p11, p12 = fr['pose'][11], fr['pose'][12]
            pose = {11: [1 - p12[0], p12[1], p12[2]], 12: [1 - p11[0], p11[1], p11[2]]}
        out.append(dict(t=fr['t'], W=fr['W'], H=fr['H'], hands=hands, pose=pose))
    return out


def split_items(items, test_frac=0.2, seed=0, official=None):
    """Stratified split by video. If official=(train_names, test_names) is given, use those file names instead."""
    if official:
        tr_names, te_names = official
        tr = [i for i, d in enumerate(items) if os.path.basename(d['video']).lower() in tr_names]
        te = [i for i, d in enumerate(items) if os.path.basename(d['video']).lower() in te_names]
        if len(tr) > 0.5 * len(items) and te: return tr, te
    rng = random.Random(seed); by = defaultdict(list)
    for i, d in enumerate(items): by[d['label']].append(i)
    tr, te = [], []
    for lab, idx in sorted(by.items()):
        rng.shuffle(idx); k = max(1, round(len(idx) * test_frac)) if len(idx) >= 3 else 0
        te += idx[:k]; tr += idx[k:]
    return tr, te


def build_arrays(items, labels, idx, n_aug, seed, workers=None, mirror_test=False):
    from multiprocessing import Pool
    lab2i = {l: i for i, l in enumerate(labels)}
    jobs = [(items[i]['frames'], n_aug, seed * 100003 + i, mirror_test) for i in idx]
    X, Y, XM, YM = [], [], [], []
    with Pool(workers or max(1, os.cpu_count() or 1)) as pool:
        for i, (vecs, mvecs) in zip(idx, pool.imap(_feat_job, jobs, chunksize=4)):
            y = lab2i[items[i]['label']]
            X += vecs; Y += [y] * len(vecs); XM += mvecs; YM += [y] * len(mvecs)
    cat = lambda a: np.stack(a).astype(np.float16) if a else np.zeros((0, T, F.FEAT_DIM), np.float16)
    return cat(X), np.array(Y, np.int64), cat(XM), np.array(YM, np.int64)


# ----------------------------------------------------------------------------- model
def make_model(n_classes, d=192, blocks=4, drop=0.25):
    import torch, torch.nn as nn

    class Block(nn.Module):
        def __init__(s, d, k, drop):
            super().__init__()
            s.dw = nn.Conv1d(d, d, k, padding=k // 2, groups=d); s.bn = nn.BatchNorm1d(d)
            s.pw = nn.Conv1d(d, 2 * d, 1); s.pw2 = nn.Conv1d(2 * d, d, 1); s.act = nn.GELU(); s.drop = nn.Dropout(drop)
        def forward(s, x):
            return x + s.drop(s.pw2(s.act(s.pw(s.bn(s.dw(x))))))

    class SignNet(nn.Module):
        """Per-frame embedding -> temporal depthwise-separable convolutions -> mean+max pooling -> classes."""
        def __init__(s):
            super().__init__()
            s.norm = nn.LayerNorm(F.FEAT_DIM)
            s.inp = nn.Sequential(nn.Linear(F.FEAT_DIM, d), nn.GELU(), nn.Dropout(drop))
            s.blocks = nn.Sequential(*[Block(d, 5 if i % 2 == 0 else 3, drop) for i in range(blocks)])
            s.head = nn.Sequential(nn.Dropout(drop), nn.Linear(2 * d, n_classes))
        def forward(s, x):                        # x: [B, T, 164]
            h = s.inp(s.norm(x)).transpose(1, 2)  # [B, d, T]
            h = s.blocks(h)
            return s.head(torch.cat([h.mean(2), h.amax(2)], 1))
    return SignNet()


def train(Xtr, Ytr, Xte, Yte, n_classes, epochs=60, batch=128, lr=2e-3, log=print, seed=0):
    import torch, torch.nn as nn
    torch.manual_seed(seed)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = make_model(n_classes).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    steps = epochs * math.ceil(len(Xtr) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
    Xtr_t, Ytr_t = torch.tensor(Xtr), torch.tensor(Ytr)
    best, best_state = -1, None
    for ep in range(epochs):
        model.train(); perm = torch.randperm(len(Xtr_t)); tot = 0.0
        for k in range(0, len(perm), batch):
            b = perm[k:k + batch]
            x, y = Xtr_t[b].float().to(dev), Ytr_t[b].to(dev)
            loss = lossf(model(x), y)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(b)
        if (ep + 1) % 5 == 0 or ep == epochs - 1:
            acc = evaluate(model, Xte, Yte)['top1'] if len(Xte) else float('nan')
            log(f'  epoch {ep+1:3d}/{epochs}  loss {tot/len(perm):.3f}  test accuracy {acc*100:.1f}%')
    return model


def predict(model, X, batch=512):
    import torch
    dev = next(model.parameters()).device; model.eval(); out = []
    with torch.no_grad():
        for k in range(0, len(X), batch): out.append(torch.softmax(model(torch.tensor(X[k:k + batch]).float().to(dev)), 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 0))


def evaluate(model, X, Y, labels=None):
    P = predict(model, X)
    top = np.argsort(-P, 1)
    r = dict(top1=float((top[:, 0] == Y).mean()), top3=float((top[:, :3] == Y[:, None]).any(1).mean()), n=len(Y))
    if labels is not None:
        conf = Counter((labels[y], labels[p]) for y, p in zip(Y, top[:, 0]) if y != p)
        r['confusions'] = [f'{a} -> {b} ({c})' for (a, b), c in conf.most_common(12)]
        per = defaultdict(list)
        for y, p in zip(Y, top[:, 0]): per[labels[y]].append(p == y)
        r['worst'] = sorted(((float(np.mean(v)), k) for k, v in per.items()))[:10]
        # confidence: how often the top guess is right when the model is >= 50% sure
        conf_ok = P.max(1) >= 0.5
        r['confident_share'] = float(conf_ok.mean())
        r['confident_accuracy'] = float((top[conf_ok, 0] == Y[conf_ok]).mean()) if conf_ok.any() else float('nan')
    return r


def export_onnx(model, path, labels, extra_meta):
    """Export to ONNX for the browser, then check onnxruntime gives the same outputs as PyTorch."""
    import torch, onnxruntime as ort
    model = model.cpu().eval()
    x = torch.randn(1, T, F.FEAT_DIM)
    kw = dict(input_names=['x'], output_names=['logits'], opset_version=17, dynamic_axes={'x': {0: 'batch'}, 'logits': {0: 'batch'}})
    try:
        torch.onnx.export(model, (x,), path, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(model, (x,), path, **kw)
    sess = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
    xb = torch.randn(4, T, F.FEAT_DIM)
    diff = float(np.abs(sess.run(None, {'x': xb.numpy()})[0] - model(xb).detach().numpy()).max())
    meta = dict(format='isl-model-v1', feat_version=F.FEAT_VERSION, T=T, feat_dim=F.FEAT_DIM, labels=labels, onnx_check_max_diff=diff, **extra_meta)
    with open(os.path.join(os.path.dirname(path), 'isl_model.json'), 'w') as f: json.dump(meta, f, indent=1)
    return diff
