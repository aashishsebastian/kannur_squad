// ISL Sign Reader — webcam → MediaPipe hands + pose → landmark sequences → few-shot DTW classifier.
// Everything runs on this laptop. No video is stored or sent anywhere: only landmark numbers are kept,
// and only for the signs you choose to teach (saved in this browser's local storage).
import { FilesetResolver, HandLandmarker, PoseLandmarker } from './lib/mediapipe/vision_bundle.mjs';
import { createTracker, toSample, packSeq, isRaised, computeStats, classify as classifyWith, leaveOneOut as looWith, activeLabels as activeLabelsOf, MIN_SAMPLES, N, toVector, FEAT_VERSION, FEAT_DIM } from './signmath.js';

const $ = s => document.querySelector(s);
const HAND_EDGES = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[17,18],[18,19],[19,20],[0,17]];

/* ======================= storage ======================= */
const DB = {
  db: null,
  open(){ return this.db ? Promise.resolve(this.db) : new Promise((res, rej) => { const r = indexedDB.open('isl-avatar', 1); r.onupgradeneeded = () => r.result.createObjectStore('files'); r.onsuccess = () => res(this.db = r.result); r.onerror = () => rej(r.error); }); },
  async get(k){ try { const db = await this.open(); return await new Promise(res => { const q = db.transaction('files').objectStore('files').get(k); q.onsuccess = () => res(q.result ?? null); q.onerror = () => res(null); }); } catch { return null; } },
  async set(k, v){ try { const db = await this.open(); db.transaction('files', 'readwrite').objectStore('files').put(v, k); } catch {} }
};

/* ======================= recognition state ======================= */
let samples = {};            // label -> [{ seq, created }]
let stats = null;            // typical within-sign distance, from your recordings
const tracker = createTracker(), body = tracker.body;
const getStats = () => stats || (stats = computeStats(samples));
const activeLabels = () => activeLabelsOf(samples);

/* ======================= trained model (optional) ======================= */
// If ./model/isl_model.onnx + isl_model.json exist (made by the Colab training notebook), recognise their signs for anyone.
let model = null;
async function loadModel(){
  const el = $('#modelState');
  try {
    const r = await fetch('./model/isl_model.json', { cache: 'no-store' });
    if (!r.ok) throw Object.assign(new Error('missing'), { missing: true });
    const meta = await r.json();
    if (meta.format !== 'isl-model-v1' || meta.feat_dim !== FEAT_DIM) throw new Error('This model file is from an incompatible version of the training notebook.');
    if (meta.feat_version !== FEAT_VERSION) throw new Error('The model was trained with different features (v' + meta.feat_version + ') than this app uses (v' + FEAT_VERSION + '). Retrain with the matching notebook.');
    const ort = await import('./lib/ort/ort.wasm.min.mjs');
    ort.env.wasm.wasmPaths = new URL('./lib/ort/', location.href).href; ort.env.wasm.numThreads = 1;
    const session = await ort.InferenceSession.create('./model/isl_model.onnx', { executionProviders: ['wasm'] });
    model = { ort, session, meta };
    const acc = meta.test_top1 != null ? ` · ${Math.round(meta.test_top1 * 100)}% on held-out videos` : '';
    el.innerHTML = `<b>Trained model loaded:</b> ${meta.labels.length} signs${acc}. <a href="#" id="modelSignsLink">See signs</a>`;
    el.className = 'okbox';
    $('#modelSignsLink').onclick = e => { e.preventDefault(); $('#modelSigns').classList.toggle('hidden'); };
    $('#modelSigns').textContent = meta.labels.join(' · ');
    $('#sourceRow').classList.remove('hidden');
  } catch (e) {
    model = null; $('#sourceRow').classList.add('hidden');
    el.className = 'sub';
    el.innerHTML = e.missing ? 'No trained model yet — recognition uses only the signs you teach. (Train one with the Colab notebook, see TRAINING.md.)' : 'Trained model could not load: ' + e.message;
  }
}
async function modelPredict(frames){
  if (!model) return null;
  const T = model.meta.T, s = toSample(frames, T);
  if (!s.ok) return null;
  const x = new model.ort.Tensor('float32', toVector(s.seq), [1, T, FEAT_DIM]);
  const out = await model.session.run({ x });
  const logits = out.logits.data, mx = Math.max(...logits), ex = Array.from(logits, v => Math.exp(v - mx)), Z = ex.reduce((a, b) => a + b, 0);
  const order = ex.map((p, i) => [p / Z, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
  return order.map(([p, i]) => ({ label: model.meta.labels[i], conf: p, src: 'model' }));
}
const modelStatus = p => p >= 0.5 ? 'needs_confirmation' : p >= 0.25 ? 'uncertain_repeat' : 'unsupported';

/* ======================= tracking ======================= */
let vision = null, hands = null, pose = null, stream = null, running = false, lastVT = -1, frameNo = 0;
let lastHands = null, lastPose = null;
const video = $('#cam'), canvas = $('#camCanvas'), ctx = canvas.getContext('2d');
const msgEl = $('#camMsg'), countEl = $('#camCount'), recEl = $('#camRec'), fpsEl = $('#camFps');
const lum = document.createElement('canvas'); lum.width = 32; lum.height = 24; const lctx = lum.getContext('2d', { willReadFrequently: true });
let brightness = 128;

async function createTasks(order = ['GPU', 'CPU']){
  vision = vision || await FilesetResolver.forVisionTasks('./lib/mediapipe/wasm');
  const mk = async (Cls, model, extra, delegate) => Cls.createFromOptions(vision, { baseOptions: { modelAssetPath: model, delegate }, runningMode: 'VIDEO', ...extra });
  for (const delegate of order) {
    try {
      hands = hands || await mk(HandLandmarker, './lib/mediapipe/models/hand_landmarker.task', { numHands: 2, minHandDetectionConfidence: 0.5, minHandPresenceConfidence: 0.5, minTrackingConfidence: 0.5 }, delegate);
      pose = pose || await mk(PoseLandmarker, './lib/mediapipe/models/pose_landmarker_lite.task', { numPoses: 1 }, delegate);
      return delegate;
    } catch (e) { console.warn('MediaPipe ' + delegate + ' failed', e); hands = null; pose = null; }
  }
  throw new Error('Hand tracking could not start on this computer.');
}

function state(t){ $('#camState').textContent = t; }
function camHelp(html){ const el = $('#camHelp'); el.innerHTML = html; el.classList.toggle('hidden', !html); }

async function startCamera(){
  camHelp('');
  if (!navigator.mediaDevices?.getUserMedia) { camHelp('This browser cannot use the camera here. Open the app through <b>start_app.bat</b> in Chrome or Edge.'); return; }
  $('#camStart').disabled = true; state('Starting camera…');
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }, audio: false });
  } catch (e) {
    $('#camStart').disabled = false; state('Camera off');
    if (e.name === 'NotAllowedError') camHelp('Camera permission was blocked. Click the <b>camera icon</b> (or lock icon) at the right/left of the address bar, choose <b>Allow</b>, then press Start camera again.');
    else if (e.name === 'NotFoundError') camHelp('No camera was found. Check that your webcam is connected and enabled.');
    else if (e.name === 'NotReadableError') camHelp('The camera is being used by another app (Zoom, Teams, Camera app…). Close that app and try again.');
    else camHelp('Could not start the camera: ' + e.message);
    return;
  }
  video.srcObject = stream; await video.play();
  state('Loading hand tracking…');
  try { const d = await createTasks(); state('Camera on · tracking (' + d + ')'); }
  catch (e) { state('Tracking failed'); camHelp(e.message + ' Try reloading the page.'); stopCamera(); return; }
  running = true; $('#camIdle').classList.add('hidden'); $('#camStart').textContent = 'Stop camera'; $('#camStart').disabled = false;
  requestAnimationFrame(loop);
}
function stopCamera(){
  running = false; capture.active = false; auto.on = false;
  if (stream) stream.getTracks().forEach(t => t.stop()); stream = null; video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  $('#camIdle').classList.remove('hidden'); $('#camStart').textContent = 'Start camera'; $('#camStart').disabled = false; state('Camera off');
  countEl.textContent = ''; recEl.classList.add('hidden'); msgEl.textContent = '';
}

let fpsT = performance.now(), fpsN = 0;
function loop(){
  if (!running) return;
  requestAnimationFrame(loop);
  if (video.readyState < 2 || video.currentTime === lastVT) return;
  lastVT = video.currentTime;
  const W = video.videoWidth, H = video.videoHeight, ts = performance.now();
  if (canvas.width !== W) { canvas.width = W; canvas.height = H; }
  processFrame(video, W, H, ts);
  if (++fpsN >= 15) { const now = performance.now(); fpsEl.textContent = Math.round(fpsN * 1000 / (now - fpsT)) + ' fps'; fpsT = now; fpsN = 0; }
}
function processFrame(src, W, H, ts){
  frameNo++;
  lastHands = hands.detectForVideo(src, ts);
  if (frameNo % 2 === 1 || !lastPose) lastPose = pose.detectForVideo(src, ts);
  if (frameNo % 15 === 1) { try { lctx.drawImage(src, 0, 0, 32, 24); const d = lctx.getImageData(0, 0, 32, 24).data; let s = 0; for (let i = 0; i < d.length; i += 4) s += 0.299*d[i] + 0.587*d[i+1] + 0.114*d[i+2]; brightness = s / (d.length / 4); } catch {} }
  const f = tracker.frame(lastHands, lastPose?.landmarks?.[0], W, H, ts);
  draw(W, H);
  feedback(W, H);
  onFrame(f);
  return f;
}
function draw(W, H){
  ctx.clearRect(0, 0, W, H);
  if (lastPose?.landmarks?.[0]) {
    const lm = lastPose.landmarks[0];
    ctx.strokeStyle = 'rgba(255,255,255,.55)'; ctx.lineWidth = 3; ctx.beginPath();
    ctx.moveTo(lm[11].x * W, lm[11].y * H); ctx.lineTo(lm[12].x * W, lm[12].y * H); ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,.7)'; for (const i of [0, 11, 12]) { ctx.beginPath(); ctx.arc(lm[i].x * W, lm[i].y * H, 4, 0, 7); ctx.fill(); }
  }
  (lastHands?.landmarks || []).forEach((lm, i) => {
    const lab = lastHands.handedness?.[i]?.[0]?.categoryName;
    const col = lab === 'Left' ? '#4fb3ff' : '#f5b84a';   // blue = signer's right hand, amber = left
    ctx.strokeStyle = col; ctx.lineWidth = 3;
    for (const [a, b] of HAND_EDGES) { ctx.beginPath(); ctx.moveTo(lm[a].x * W, lm[a].y * H); ctx.lineTo(lm[b].x * W, lm[b].y * H); ctx.stroke(); }
    ctx.fillStyle = '#fff'; for (const p of lm) { ctx.beginPath(); ctx.arc(p.x * W, p.y * H, 3, 0, 7); ctx.fill(); }
  });
}
function feedback(W, H){
  const tips = [];
  const nh = lastHands?.landmarks?.length || 0;
  if (brightness < 55) tips.push('Too dark — add light in front of you');
  if (!body.ok) tips.push('Show your head and shoulders');
  else if (body.s > W * 0.55) tips.push('Move back a little');
  else if (body.s < W * 0.16) tips.push('Move closer');
  if (nh) for (const lm of lastHands.landmarks) { if (lm.some(p => p.x < 0.02 || p.x > 0.98 || p.y < 0.02 || p.y > 0.98)) { tips.push('A hand is at the edge of the picture'); break; } }
  if (!nh && (capture.active || auto.recording)) tips.push('No hands visible');
  msgEl.textContent = tips[0] || '';
  msgEl.classList.toggle('hidden', !tips.length);
}

/* ======================= capture (button + countdown, or automatic) ======================= */
const capture = { active: false, phase: '', frames: [], t0: 0, len: 2000, purpose: 'recognize' };
const auto = { on: false, recording: false, frames: [], raisedN: 0, lowSince: 0 };
const handsRaised = isRaised;   // wrist above the lower chest
function onFrame(f){
  const now = f.t;
  if (capture.active) {
    if (capture.phase === 'count') {
      const left = Math.ceil((capture.t0 - now) / 500);
      countEl.textContent = left > 0 ? left : '';
      if (now >= capture.t0) { capture.phase = 'rec'; capture.frames = []; recEl.classList.remove('hidden'); countEl.textContent = ''; }
    } else if (capture.phase === 'rec') {
      capture.frames.push(f);
      $('#recBar').style.width = Math.min(100, (now - capture.t0) / capture.len * 100) + '%';
      if (now - capture.t0 >= capture.len) { capture.active = false; recEl.classList.add('hidden'); finish(capture.frames, capture.purpose); }
    }
    return;
  }
  if (!auto.on) return;
  if (!auto.recording) {
    auto.raisedN = handsRaised(f) ? auto.raisedN + 1 : 0;
    if (auto.raisedN >= 3) { auto.recording = true; auto.frames = []; auto.lowSince = 0; recEl.classList.remove('hidden'); }
  }
  if (auto.recording) {
    auto.frames.push(f);
    const dur = now - auto.frames[0].t;
    $('#recBar').style.width = Math.min(100, dur / 4000 * 100) + '%';
    if (handsRaised(f)) auto.lowSince = 0; else if (!auto.lowSince) auto.lowSince = now;
    if ((auto.lowSince && now - auto.lowSince > 350) || dur > 4000) {
      auto.recording = false; auto.raisedN = 0; recEl.classList.add('hidden');
      if (dur > 400) finish(auto.frames, mode === 'teach' ? 'teach' : 'recognize');
    }
  }
}
function startCapture(purpose){
  if (!running) { flash('Start the camera first'); return; }
  if (auto.on) { flash('Automatic mode is on — just sign, then lower your hands'); return; }
  if (capture.active) return;
  Object.assign(capture, { active: true, phase: 'count', frames: [], t0: performance.now() + 1500, len: +$('#recLen').value * 1000, purpose });
}
function flash(t){ msgEl.textContent = t; msgEl.classList.remove('hidden'); }

/* ======================= modes + UI ======================= */
let mode = 'recognize', pending = null, sentence = [];
async function finish(frames, purpose){
  const s = toSample(frames);
  if (purpose === 'teach') {
    if (!s.ok) { teachMsg(s.reason === 'no_hands' ? 'No hands were seen — try again.' : 'Too short — try again.'); return; }
    const label = currentTeachLabel(); if (!label) { teachMsg('Choose a sign first.'); return; }
    (samples[label] = samples[label] || []).push({ seq: packSeq(s.seq), created: Date.now() });
    stats = null; save(); renderSamples();
    const n = samples[label].length;
    teachMsg(`Saved sample ${n} for “${label}”.` + (n < 5 ? ` Record ${5 - n} more.` : ' That\'s enough for this sign — pick the next one.'));
    return;
  }
  if (!s.ok) { showResult({ status: 'no_sign_detected' }); return; }
  const src = model ? $('#source').value : 'mine';
  const mine = src !== 'model' ? classifyWith(samples, getStats(), s.seq, { strict: +$('#strict').value }) : { status: 'not_ready' };
  let r = mine;
  if (src !== 'mine') {
    let mc = null; try { mc = await modelPredict(frames); } catch (e) { console.warn('model failed', e); }
    if (mc) {
      // combine: your own taught signs win when they are a confident match, otherwise the trained model's guess leads
      const mineOk = mine.cands && mine.status === 'needs_confirmation';
      if (mineOk && mine.cands[0].conf >= mc[0].conf) r = { ...mine, cands: dedupe([...mine.cands, ...mc]).slice(0, 3) };
      else r = { status: modelStatus(mc[0].conf), cands: dedupe([...mc, ...(mine.cands || [])]).slice(0, 3) };
    }
  }
  r.seq = s.seq; showResult(r);
}
function dedupe(c){ const seen = new Set(); return c.filter(x => !seen.has(x.label) && seen.add(x.label)); }
function currentTeachLabel(){ const c = $('#customLabel').value.trim().toLowerCase(); return c || $('#teachLabel').value; }
function teachMsg(t){ $('#teachHint').textContent = t; }

const STATUS_TEXT = {
  needs_confirmation: 'Is this right? Confirm or pick another.',
  uncertain_repeat: 'Not sure — please repeat, or pick the right one.',
  unsupported: 'Not recognised — this may not be a taught sign. Repeat or pick one.',
  no_sign_detected: 'No signing was seen. Keep your hands in view and try again.',
  not_ready: 'Teach at least 2 signs (3+ samples each) in “Teach signs” first, or add a trained model.'
};
function showResult(r){
  pending = r;
  const card = $('#resultCard'); card.classList.remove('hidden');
  card.dataset.status = r.status;
  const top = r.cands?.[0];
  $('#resText').textContent = r.status === 'needs_confirmation' || r.status === 'uncertain_repeat' ? top.label : '—';
  $('#resStatus').textContent = STATUS_TEXT[r.status];
  const list = $('#resCands'); list.innerHTML = '';
  for (const c of r.cands || []) {
    const b = document.createElement('button'); b.className = 'cand';
    b.innerHTML = `<span>${c.label}</span><span class="bar"><i style="width:${Math.round(c.conf * 100)}%"></i></span><span class="pct">${Math.round(c.conf * 100)}%</span>`;
    b.onclick = () => confirm(c.label, c !== top || r.status !== 'needs_confirmation');
    list.appendChild(b);
  }
  $('#confirmBtn').disabled = !(r.status === 'needs_confirmation' || r.status === 'uncertain_repeat');
  // stage banner so the signer sees the guess without looking at the sidebar
  const sb = $('#camResult'); sb.textContent = $('#resText').textContent === '—' ? STATUS_TEXT[r.status].split('.')[0] : top.label + '?';
  sb.classList.remove('hidden'); sb.classList.toggle('bad', !(r.status === 'needs_confirmation' || r.status === 'uncertain_repeat'));
}
function confirm(label, corrected){
  if (!pending) return;
  if (corrected && $('#learn').checked && pending.seq) {
    (samples[label] = samples[label] || []).push({ seq: packSeq(pending.seq), created: Date.now(), fromCorrection: true });
    stats = null; save(); renderSamples();
  }
  sentence.push(label); renderSentence();
  if ($('#autoSpeak').checked) speak(label);
  pending = null; $('#resultCard').classList.add('hidden'); $('#camResult').classList.add('hidden');
}
function renderSentence(){
  const el = $('#sentence'); el.innerHTML = '';
  if (!sentence.length) { el.innerHTML = '<span class="sub">Confirmed signs appear here.</span>'; return; }
  sentence.forEach((w, i) => { const s = document.createElement('span'); s.className = 'chip'; s.textContent = w; s.title = 'Click to remove'; s.onclick = () => { sentence.splice(i, 1); renderSentence(); }; el.appendChild(s); });
}
function speak(text){
  if (!('speechSynthesis' in window) || !text) return;
  const u = new SpeechSynthesisUtterance(text);
  const v = speechSynthesis.getVoices().find(v => v.lang === 'en-IN') || speechSynthesis.getVoices().find(v => v.lang?.startsWith('en'));
  if (v) u.voice = v; u.lang = v?.lang || 'en-IN'; u.rate = 0.95;
  speechSynthesis.cancel(); speechSynthesis.speak(u);
}

function renderSamples(){
  const el = $('#sampleList'); el.innerHTML = '';
  const labels = Object.keys(samples).filter(l => samples[l].length).sort();
  if (!labels.length) { el.innerHTML = '<span class="sub">Nothing taught yet.</span>'; }
  for (const l of labels) {
    const n = samples[l].length, b = document.createElement('button');
    b.className = 'btn' + (n >= MIN_SAMPLES ? ' ready' : '');
    b.textContent = `${l} · ${n}`; b.title = n >= MIN_SAMPLES ? 'Ready to recognise. Click to select.' : `Needs ${MIN_SAMPLES - n} more sample(s). Click to select.`;
    b.onclick = () => { if ([...$('#teachLabel').options].some(o => o.value === l)) { $('#teachLabel').value = l; $('#customLabel').value = ''; } else $('#customLabel').value = l; };
    el.appendChild(b);
  }
  const ready = activeLabels().length;
  $('#readyCount').textContent = ready + (ready === 1 ? ' sign' : ' signs') + ' ready';
  // mark counts in the dropdown
  for (const o of $('#teachLabel').options) { const n = samples[o.value]?.length || 0; o.textContent = o.value + (n ? `  (${n})` : ''); }
}
async function save(){ await DB.set('samples', samples); }
async function load(){ const s = await DB.get('samples'); if (s && typeof s === 'object') samples = s; renderSamples(); }

function setMode(m){
  mode = m;
  document.querySelectorAll('#camModeSeg button').forEach(b => b.classList.toggle('on', b.dataset.mode === m));
  $('#teachPanel').classList.toggle('hidden', m !== 'teach');
  $('#recogPanel').classList.toggle('hidden', m !== 'recognize');
  $('#camModeBadge').textContent = m === 'teach' ? 'TEACH: ' + currentTeachLabel() : 'RECOGNISE';
  if (m === 'teach') { pending = null; $('#resultCard').classList.add('hidden'); $('#camResult').classList.add('hidden'); }
}

export function initCamera(signLabels){
  const sel = $('#teachLabel');
  for (const l of signLabels) { const o = document.createElement('option'); o.value = l; o.textContent = l; sel.appendChild(o); }
  $('#camStart').onclick = () => running ? stopCamera() : startCamera();
  document.querySelectorAll('#camModeSeg button').forEach(b => b.onclick = () => setMode(b.dataset.mode));
  $('#recBtn').onclick = () => startCapture('teach');
  $('#signNowBtn').onclick = () => startCapture('recognize');
  $('#autoSeg').onchange = e => { auto.on = e.target.checked; auto.recording = false; auto.raisedN = 0; recEl.classList.add('hidden'); };
  sel.onchange = () => { $('#customLabel').value = ''; setMode(mode); };
  $('#customLabel').oninput = () => setMode(mode);
  $('#undoBtn').onclick = () => { const l = currentTeachLabel(); if (samples[l]?.length) { samples[l].pop(); if (!samples[l].length) delete samples[l]; stats = null; save(); renderSamples(); teachMsg(`Removed the last sample of “${l}”.`); } };
  $('#clearBtn').onclick = () => { if (!$('#clearBtn').dataset.armed) { $('#clearBtn').dataset.armed = 1; $('#clearBtn').textContent = 'Click again to delete all'; setTimeout(() => { delete $('#clearBtn').dataset.armed; $('#clearBtn').textContent = 'Delete all'; }, 3000); return; } samples = {}; stats = null; save(); renderSamples(); $('#evalOut').textContent = ''; delete $('#clearBtn').dataset.armed; $('#clearBtn').textContent = 'Delete all'; };
  $('#evalBtn').onclick = () => {
    const ready = activeLabels(); if (ready.length < 2) { $('#evalOut').textContent = 'Teach at least 2 signs with 3+ samples each first.'; return; }
    stats = null; const r = looWith(samples, getStats());
    const conf = Object.entries(r.confusions).sort((a, b) => b[1] - a[1]).slice(0, 4).map(([k, v]) => `${k} (${v})`).join(', ');
    $('#evalOut').innerHTML = `<b>${Math.round(r.right / r.total * 100)}%</b> of your ${r.total} samples were recognised correctly when each was held out (${ready.length} signs). ` + (conf ? 'Most confused: ' + conf + '. Add more varied samples for these.' : 'No confusions.') + '<br><span class="sub">This only measures your own signing; other signers will need their own samples or a trained model.</span>';
  };
  $('#exportBtn').onclick = () => { const blob = new Blob([JSON.stringify({ format: 'isl-samples-v1', n: N, samples })], { type: 'application/json' }); const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'isl_taught_signs.json'; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 5000); };
  $('#importIn').onchange = async e => { const f = e.target.files[0]; if (!f) return; try { const j = JSON.parse(await f.text()); if (j.format !== 'isl-samples-v1') throw new Error('not a samples file'); for (const [l, arr] of Object.entries(j.samples)) samples[l] = (samples[l] || []).concat(arr); stats = null; save(); renderSamples(); teachMsg('Imported.'); } catch (err) { teachMsg('Could not import: ' + err.message); } e.target.value = ''; };
  $('#confirmBtn').onclick = () => pending?.cands && confirm(pending.cands[0].label, false);
  $('#retryBtn').onclick = () => { pending = null; $('#resultCard').classList.add('hidden'); $('#camResult').classList.add('hidden'); startCapture('recognize'); };
  $('#speakAll').onclick = () => speak(sentence.join(' '));
  $('#clearSentence').onclick = () => { sentence = []; renderSentence(); };
  addEventListener('keydown', e => {
    if (!document.body.classList.contains('camera-on')) return;
    if (/INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName)) return;
    if (e.code === 'Space') { e.preventDefault(); startCapture(mode === 'teach' ? 'teach' : 'recognize'); }
    else if (e.key === 'Enter' && pending?.cands && mode === 'recognize') { e.preventDefault(); $('#confirmBtn').click(); }
  });
  renderSentence(); setMode('recognize'); load(); loadModel();
  if ('speechSynthesis' in window) speechSynthesis.getVoices();
}
export function leaveCamera(){ if (running) stopCamera(); }

// test hooks (used for automated checks; harmless in normal use)
window.__cam = {
  get samples(){ return samples; }, set samples(v){ samples = v; stats = null; renderSamples(); },
  toSample, packSeq, tracker,
  classify: seq => classifyWith(samples, getStats(), seq, { strict: +$('#strict').value }),
  leaveOneOut: () => looWith(samples, getStats()),
  async initTasks(order){ return createTasks(order); },
  detect(src, W, H, ts){ lastHands = hands.detectForVideo(src, ts); lastPose = pose.detectForVideo(src, ts); return { f: tracker.frame(lastHands, lastPose?.landmarks?.[0], W, H, ts), nHands: lastHands.landmarks.length, pose: !!lastPose.landmarks?.[0] }; },
  rawDetect(src, W, H, ts){ const h = hands.detectForVideo(src, ts), po = pose.detectForVideo(src, ts); const r3 = x => Math.round(x * 1e4) / 1e4;
    return { t: ts, W, H, hands: h.landmarks.map((lm, i) => ({ label: h.handedness?.[i]?.[0]?.categoryName, score: h.handedness?.[i]?.[0]?.score, lm: lm.map(q => [r3(q.x), r3(q.y), r3(q.z)]) })),
             pose: po.landmarks?.[0] ? [0, 11, 12].map(i => { const q = po.landmarks[0][i]; return [r3(q.x), r3(q.y), r3(q.visibility ?? 1)]; }) : null }; },
  finish: (frames, purpose) => finish(frames, purpose), setMode: m => setMode(m), modelPredict, loadModel, get model(){ return model; },
  get running(){ return running; }
};
