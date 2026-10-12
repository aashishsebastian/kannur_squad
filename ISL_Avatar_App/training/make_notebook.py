"""Builds train_isl_model.ipynb with islfeat.py and isltrain.py embedded (so the notebook is self-contained)."""
import json, sys

HERE = sys.argv[1] if len(sys.argv) > 1 else '.'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'train_isl_model.ipynb'
islfeat = open(f'{HERE}/islfeat.py').read()
isltrain = open(f'{HERE}/isltrain.py').read()

ZIPS = {
    'Adjectives': 8, 'Animals': 2, 'Clothes': 2, 'Colours': 2, 'Days_and_Time': 3, 'Electronics': 2, 'Greetings': 2,
    'Home': 4, 'Jobs': 2, 'Means_of_Transportation': 2, 'People': 5, 'Places': 4, 'Pronouns': 2, 'Seasons': 1, 'Society': 3,
}
QUICK = ['Greetings', 'Pronouns', 'Days_and_Time', 'Society', 'People']

cells = []
md = lambda s: cells.append(dict(cell_type='markdown', metadata={}, source=s.strip('\n')))
code = lambda s, title=None: cells.append(dict(cell_type='code', metadata={'cellView': 'form'} if title else {}, execution_count=None, outputs=[], source=s.strip('\n')))

md("""
# Train the ISL sign-recognition model

This notebook teaches a model to recognise Indian Sign Language words **for anyone**, using the free
**INCLUDE** dataset (videos of 263 ISL words signed by Deaf students of St. Louis School for the Deaf, Chennai —
IIT Madras / AI4Bharat, CC BY 4.0). You don't need to know sign language: the dataset is the teacher.

**What to do**
1. `Runtime → Change runtime type → T4 GPU` (free) → Save.
2. Run each cell from top to bottom (▶ button, or `Runtime → Run all`).
3. At the end a file **`isl_model.zip`** downloads. Unzip it into your app folder so you get
   `ISL_Avatar_App/model/isl_model.onnx` and `isl_model.json`, then restart the app.

**Time:** quick start ≈ 30–45 min, everything ≈ 1.5–2.5 h (mostly downloading and reading the videos).
Progress is saved to your Google Drive, so if Colab disconnects just run all cells again — it continues where it stopped.
""")

code(r"""
#@title 1 · Setup (≈1 min) { display-mode: "form" }
SAVE_TO_GOOGLE_DRIVE = True  #@param {type:"boolean"}
LOCAL_TEST_ZIPS = ""         # leave empty (used only for offline testing of this notebook)

import os, sys, shutil, subprocess, time, glob
IN_COLAB = 'google.colab' in sys.modules
if SAVE_TO_GOOGLE_DRIVE and IN_COLAB and not LOCAL_TEST_ZIPS:
    from google.colab import drive
    drive.mount('/content/drive')
    WORK = '/content/drive/MyDrive/ISL_training'
else:
    WORK = os.path.abspath('ISL_training')
TMP = '/content/tmp' if IN_COLAB else os.path.abspath('tmp_isl')
for d in (WORK, f'{WORK}/landmarks', f'{WORK}/done', TMP, 'models'): os.makedirs(d, exist_ok=True)

def sh(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if r.returncode: print(r.stdout[-2000:], r.stderr[-2000:]); raise RuntimeError('command failed: ' + cmd)
    return r.stdout

try:
    import mediapipe, onnx, onnxruntime
except ImportError:
    sh(f'{sys.executable} -m pip -q install "mediapipe>=0.10.14,<0.11" onnx onnxruntime')
for url in ['https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
            'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task']:
    dst = 'models/' + url.rsplit('/', 1)[1]
    if not os.path.exists(dst): sh(f'wget -q -O {dst} {url}')
import torch
print('Work folder:', WORK)
gpu = subprocess.run('nvidia-smi --query-gpu=name --format=csv,noheader', shell=True, capture_output=True, text=True).stdout.strip()
print('GPU:', gpu or 'none (training will be much slower — Runtime → Change runtime type → T4 GPU)')
print('CPUs:', os.cpu_count())
""", True)

code("%%writefile islfeat.py\n" + islfeat)
code("%%writefile isltrain.py\n" + isltrain)

code(r"""
#@title 2 · Choose which signs to learn { display-mode: "form" }
DATASET = "Quick start (5 categories, ~19 GB download)"  #@param ["Quick start (5 categories, ~19 GB download)", "Everything (15 categories, all 263 signs, ~57 GB download)"]
ZIPS = """ + json.dumps(ZIPS) + r"""
QUICK = """ + json.dumps(QUICK) + r"""
cats = QUICK if DATASET.startswith('Quick') else list(ZIPS)
zip_names = [f'{c}_{i}of{ZIPS[c]}.zip' for c in cats for i in range(1, ZIPS[c] + 1)]
if LOCAL_TEST_ZIPS:
    zip_names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(LOCAL_TEST_ZIPS, '*.zip')))
print(len(zip_names), 'files to process:', ', '.join(zip_names))
""", True)

code(r"""
#@title 3 · Download videos and read hand movements (the long step) { display-mode: "form" }
import sys; sys.path.insert(0, '.')
import importlib, isltrain as L; importlib.reload(L)
HAND, POSE = os.path.abspath('models/hand_landmarker.task'), os.path.abspath('models/pose_landmarker_lite.task')
import requests

def download(url, path, tries=8):
    # resumable download that prints progress every ~10 s (Zenodo can be slow; this shows it's still moving)
    for attempt in range(tries):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        try:
            with requests.get(url, stream=True, timeout=60, headers={'Range': f'bytes={have}-'} if have else {}) as r:
                if r.status_code == 416: return                      # already complete
                r.raise_for_status()
                if have and r.status_code != 206: have = 0             # server ignored resume: start again
                total = have + int(r.headers.get('Content-Length', 0))
                t0, last, got = time.time(), 0, have
                with open(path, 'ab' if have else 'wb') as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk); got += len(chunk)
                        if time.time() - last > 10:
                            last = time.time(); sp = (got - have) / max(1e-6, last - t0) / 1e6
                            eta = (total - got) / 1e6 / sp / 60 if sp > 0 and total else 0
                            print(f'      {got/1e9:.2f} / {total/1e9:.2f} GB  ({sp:.1f} MB/s, ~{eta:.0f} min left)', flush=True)
            if not total or os.path.getsize(path) >= total: return
        except Exception as e:
            print(f'      connection problem ({e.__class__.__name__}), retrying in 10 s…', flush=True); time.sleep(10)
    raise RuntimeError('Download kept failing. Check your internet / try again later: ' + url)
t_all = time.time()
for k, name in enumerate(zip_names, 1):
    marker = f'{WORK}/done/{name}.done'
    if os.path.exists(marker):
        print(f'[{k}/{len(zip_names)}] {name}: already done'); continue
    zp, vdir = f'{TMP}/{name}', f'{TMP}/videos'
    shutil.rmtree(vdir, ignore_errors=True)
    print(f'[{k}/{len(zip_names)}] {name}: downloading…', flush=True)
    if LOCAL_TEST_ZIPS: shutil.copy(os.path.join(LOCAL_TEST_ZIPS, name), zp)
    else: download(f'https://zenodo.org/records/4010759/files/{name}?download=1', zp)
    print('   unzipping…', flush=True); sh(f'unzip -q -o "{zp}" -d "{vdir}"'); os.remove(zp)
    print('   reading hand movements…', flush=True)
    L.extract_folder(vdir, f'{WORK}/landmarks', HAND, POSE, log=print)
    shutil.rmtree(vdir, ignore_errors=True)
    open(marker, 'w').write('ok')
print(f'Done in {(time.time()-t_all)/60:.0f} min.', len(os.listdir(f"{WORK}/landmarks")), 'videos ready.')
""", True)

code(r"""
#@title 4 · Prepare training data { display-mode: "form" }
AUGMENTED_COPIES_PER_VIDEO = 12  #@param {type:"slider", min:2, max:24, step:1}
from collections import Counter
items = L.load_landmarks(f'{WORK}/landmarks')
count = Counter(d['label'] for d in items)
labels = sorted(l for l, c in count.items() if c >= 4)          # need a few videos per sign
items = [d for d in items if d['label'] in labels]
# Optional: the official INCLUDE train/test split. Put train.csv and test.csv in the work folder to use it.
official = None
if os.path.exists(f'{WORK}/train.csv') and os.path.exists(f'{WORK}/test.csv'):
    import re
    names = lambda p: {m.lower() for m in re.findall(r'[\w\-. ]+\.(?:mov|mp4|avi|mkv)', open(p).read(), re.I)}
    official = ({os.path.basename(n) for n in names(f'{WORK}/train.csv')}, {os.path.basename(n) for n in names(f'{WORK}/test.csv')})
tr, te = L.split_items(items, test_frac=0.2, seed=0, official=official)
print(f'{len(labels)} signs, {len(items)} videos → {len(tr)} for training, {len(te)} held out for testing'
      + (' (official split)' if official else ' (random 80/20 split per sign)'))
t0 = time.time()
Xtr, Ytr, _, _ = L.build_arrays(items, labels, tr, n_aug=AUGMENTED_COPIES_PER_VIDEO, seed=1)
Xte, Yte, Xte_m, Yte_m = L.build_arrays(items, labels, te, n_aug=0, seed=2, mirror_test=True)
print(f'Training examples: {len(Xtr)} (incl. augmented copies), test: {len(Xte)}  [{time.time()-t0:.0f}s]')
""", True)

code(r"""
#@title 5 · Train the model { display-mode: "form" }
EPOCHS = 60  #@param {type:"slider", min:10, max:150, step:5}
t0 = time.time()
model = L.train(Xtr, Ytr, Xte, Yte, n_classes=len(labels), epochs=EPOCHS, log=print)
print(f'Training took {(time.time()-t0)/60:.1f} min')
""", True)

code(r"""
#@title 6 · How good is it? { display-mode: "form" }
r = L.evaluate(model, Xte, Yte, labels)
rm = L.evaluate(model, Xte_m, Yte_m) if len(Xte_m) else None
print(f'Held-out test videos: {r["n"]}')
print(f'  Correct first guess:      {r["top1"]*100:.1f}%')
print(f'  Correct within top 3:     {r["top3"]*100:.1f}%')
if rm: print(f'  Left-handed (mirrored):   {rm["top1"]*100:.1f}%')
print(f'  When the model is ≥50% sure ({r["confident_share"]*100:.0f}% of the time) it is right {r["confident_accuracy"]*100:.1f}% of the time')
print('\nHardest signs (accuracy):', ', '.join(f'{k} {v*100:.0f}%' for v, k in r['worst']))
print('Most common mix-ups:', '; '.join(r['confusions']) or 'none')
print('\nNote: INCLUDE does not say which student signed which video, so the same students can appear in training and')
print('test videos. Expect somewhat lower accuracy for new people and different cameras — test with real users.')
""", True)

code(r"""
#@title 7 · Export for the app and download { display-mode: "form" }
import zipfile, datetime
exp = f'{TMP}/export'; os.makedirs(exp, exist_ok=True)
meta = dict(dataset='INCLUDE (IIT Madras / AI4Bharat), CC BY 4.0 — https://zenodo.org/records/4010759',
            categories=cats if not LOCAL_TEST_ZIPS else ['local test'], train_videos=len(tr), test_videos=len(te),
            test_top1=round(r['top1'], 4), test_top3=round(r['top3'], 4), mirrored_top1=round(rm['top1'], 4) if rm else None,
            epochs=EPOCHS, created=datetime.datetime.now().isoformat(timespec='minutes'))
diff = L.export_onnx(model, f'{exp}/isl_model.onnx', labels, meta)
print('ONNX check (max difference vs PyTorch):', f'{diff:.2e}', '✓' if diff < 1e-3 else '✗ — please report this')
zpath = f'{WORK}/isl_model.zip'
with zipfile.ZipFile(zpath, 'w') as z:
    for f in ('isl_model.onnx', 'isl_model.json'): z.write(f'{exp}/{f}', f'model/{f}')
print('Saved', zpath, f'({os.path.getsize(zpath)/1e6:.1f} MB)')
if IN_COLAB:
    from google.colab import files; files.download(zpath)
print('Unzip it into your ISL_Avatar_App folder (it creates model/isl_model.onnx and model/isl_model.json), then restart the app.')
""", True)

nb = dict(nbformat=4, nbformat_minor=5, cells=cells, metadata=dict(
    kernelspec=dict(name='python3', display_name='Python 3', language='python'), language_info=dict(name='python'),
    accelerator='GPU', colab=dict(provenance=[], gpuType='T4')))
for c in nb['cells']:
    c['source'] = [l + '\n' for l in c['source'].split('\n')]; c['source'][-1] = c['source'][-1].rstrip('\n')
    if c['cell_type'] == 'code': c['id'] = None
for i, c in enumerate(nb['cells']): c['id'] = f'cell{i}'
json.dump(nb, open(OUT, 'w'), indent=1)
print('wrote', OUT, len(cells), 'cells')
