// Live news / video → captions → continuous avatar signing.
// Sources: a video file on this laptop, another browser tab (YouTube / news sites), the microphone,
// or a subtitle file (.srt/.vtt) played with a video (exact captions, no speech recognition needed).
const $ = s => document.querySelector(s);
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;

let P = null;                 // avatar player API (from index.html)
let active = false, kind = null, rec = null, track = null, shareStream = null;
let cues = null, cueIdx = 0, pausedByUs = false, lagTimer = null, rafId = 0;
let lastResultAt = 0, startedAt = 0, hintShown = false, restartFails = 0;
let skippedTotal = 0;
const video = () => $('#liveVideo');

/* ---------------- captions + transcript ---------------- */
function setCaption(finalText, interim = ''){
  const el = $('#liveCap');
  el.innerHTML = '';
  if (finalText) { const a = document.createElement('span'); a.textContent = finalText + ' '; el.appendChild(a); }
  if (interim) { const b = document.createElement('span'); b.className = 'interim'; b.textContent = interim; el.appendChild(b); }
  el.classList.toggle('hidden', !finalText && !interim);
}
let lastFinal = '';
function addTranscript(text, low){
  const box = $('#liveTranscript');
  if (box.querySelector('.sub')) box.innerHTML = '';
  const line = document.createElement('div');
  const t = kind === 'subs' || kind === 'file' ? fmtTime(video().currentTime) : new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  line.innerHTML = `<span class="ts">${t}</span> `;
  const s = document.createElement('span'); s.textContent = text; if (low) { s.className = 'low'; s.title = 'Low recognition confidence — shown but not signed'; }
  line.appendChild(s); box.appendChild(line);
  while (box.children.length > 200) box.firstChild.remove();
  box.scrollTop = 1e9;
}
const fmtTime = x => { x = Math.max(0, x || 0); const m = Math.floor(x / 60), s = Math.floor(x % 60); return `${m}:${String(s).padStart(2, '0')}`; };

// a finished sentence/segment from any source
function onFinal(text, confidence){
  text = (text || '').replace(/\s+/g, ' ').trim();
  if (!text) return;
  lastResultAt = performance.now();
  const low = confidence > 0 && confidence < 0.45;          // Chrome reports 0 when it has no estimate
  lastFinal = text; setCaption(text);
  addTranscript(text, low);
  if (low) return;                                            // captions only: never sign what we're unsure of
  P.enqueue(P.glossify(text), { rest: false });
  manageLag();
}

/* ---------------- lag control ---------------- */
function manageLag(){
  if (!active && !P.backlog()) return;
  const b = P.backlog(), mode = $('#lagMode').value;
  let boostTo = 1;
  if (mode === 'live') {
    boostTo = b > 3 ? Math.min(1.5, 1 + (b - 3) * 0.12) : 1;
    if (b > 10) { skippedTotal += P.dropToNewest(4); }
  } else if (mode === 'pause') {
    boostTo = b > 2 ? 1.15 : 1;
    const v = video();
    if (kind === 'file' || kind === 'subs') {
      if (b > 4 && !v.paused) { v.pause(); pausedByUs = true; }
      else if (pausedByUs && b < 1.5) { pausedByUs = false; v.play().catch(() => {}); }
    }
  } else {
    boostTo = b > 4 ? 1.25 : 1;
  }
  P.setBoost(boostTo);
  const badge = $('#lagBadge'), nb = P.backlog();
  if (!active) { badge.classList.add('hidden'); return; }
  badge.classList.remove('hidden');
  const behind = nb > 1.5 ? `signing ~${Math.round(nb)} s behind` : 'signing in sync';
  badge.textContent = `● LIVE · ${behind}` + (pausedByUs ? ' · video paused to catch up' : '') + (skippedTotal ? ` · ${skippedTotal} words skipped` : '');
  badge.className = nb > 6 ? 'warn' : '';
  // gentle hint if a direct-audio source produces nothing
  if (!hintShown && track && (kind === 'file' || kind === 'tab') && performance.now() - startedAt > 25000 && !lastResultAt) {
    const v = video();
    if (kind === 'tab' || (!v.paused && !v.muted)) {
      hintShown = true;
      help('No speech recognised yet. Check the video has speech and isn\'t muted. If Chrome asked for your <b>microphone</b>, your Chrome is too old to read a video\'s sound directly: update Chrome (⋮ → Help → About Google Chrome), or choose <b>Microphone</b> as the source and play the video out loud.');
    }
  }
}

/* ---------------- speech recognition ---------------- */
function startRec(){
  if (!active) return;
  rec = new SR();
  rec.lang = $('#liveLang').value; rec.continuous = true; rec.interimResults = true; rec.maxAlternatives = 1;
  rec.onresult = e => {
    let interim = '';
    for (let i = e.resultIndex; i < e.results.length; i++) {
      const r = e.results[i];
      if (r.isFinal) onFinal(r[0].transcript, r[0].confidence); else interim += r[0].transcript;
    }
    if (interim) { lastResultAt = performance.now(); setCaption(lastFinal, interim); }
  };
  rec.onerror = e => {
    if (e.error === 'no-speech' || e.error === 'aborted') return;
    if (e.error === 'not-allowed' || e.error === 'service-not-allowed') { help(kind === 'mic' ? 'Microphone access was blocked. Click the mic/lock icon in the address bar → Allow, then press Start again.' : 'Chrome blocked speech recognition. Make sure the app is opened through <b>start_app.bat</b> (http://localhost).'); stopLive(); }
    else if (e.error === 'network') { help('Speech recognition needs an internet connection (Chrome sends the audio to Google\'s speech service). Subtitle files work offline.'); }
    else if (e.error === 'audio-capture') { help('No audio could be captured from this source. For a video tab, make sure you ticked <b>Also share tab audio</b>.'); }
    else if (e.error === 'language-not-supported') { help('Chrome can\'t recognise this language here. Pick another language.'); stopLive(); }
    else console.warn('speech error', e.error);
  };
  rec.onend = () => {                         // Chrome stops after silences / ~1 min: start again while live
    if (!active || kind === 'subs') return;
    if (track && track.readyState !== 'live') { stopLive(); return; }
    setTimeout(() => { if (active) { try { startRec(); restartFails = 0; } catch { if (++restartFails > 5) stopLive(); } } }, 150);
  };
  try { track ? rec.start(track) : rec.start(); }
  catch (err) {
    if (track) { help('This Chrome can\'t read the video\'s sound directly. Update Chrome, or choose <b>Microphone</b> as the source and play the video out loud.'); stopLive(); }
    else throw err;
  }
}

/* ---------------- subtitles ---------------- */
function parseSubs(text){
  const toSec = t => { const m = t.trim().replace(',', '.').match(/(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)/); return m ? (+(m[1] || 0)) * 3600 + (+m[2]) * 60 + (+m[3]) : NaN; };
  const out = [];
  for (const block of text.replace(/\r/g, '').split(/\n\s*\n/)) {
    const lines = block.split('\n'); const i = lines.findIndex(l => l.includes('-->'));
    if (i < 0) continue;
    const [a, b] = lines[i].split('-->'); const start = toSec(a), end = toSec(b.split(/\s+/).filter(Boolean)[0] || '');
    const body = lines.slice(i + 1).join(' ').replace(/<[^>]+>/g, '').replace(/\{[^}]+\}/g, '').trim();
    if (!isNaN(start) && body) out.push({ start, end: isNaN(end) ? start + 3 : end, text: body });
  }
  return out.sort((x, y) => x.start - y.start);
}
function cueLoop(){
  if (!active || kind !== 'subs') return;
  const t = video().currentTime;
  while (cueIdx < cues.length && cues[cueIdx].start <= t) onFinal(cues[cueIdx++].text, 1);
  rafId = requestAnimationFrame(cueLoop);
}
function onSeeked(){
  if (!active) return;
  P.clearAll(); pausedByUs = false;
  if (kind === 'subs') { const t = video().currentTime; cueIdx = cues.findIndex(c => c.end > t); if (cueIdx < 0) cueIdx = cues.length; }
}

/* ---------------- start / stop ---------------- */
function help(html){ const el = $('#liveHelp'); el.innerHTML = html || ''; el.classList.toggle('hidden', !html); }
function setSourceUI(){
  const src = $('#liveSource').value;
  $('#liveFileRow').classList.toggle('hidden', src !== 'file');
  $('#liveLangRow').classList.toggle('hidden', false);
  const opt = $('#lagMode').querySelector('option[value=pause]');
  opt.disabled = src !== 'file';
  if (src !== 'file' && $('#lagMode').value === 'pause') $('#lagMode').value = 'live';
  $('#liveSourceHint').innerHTML = {
    file: 'Pick a news video saved on this laptop (MP4/WebM). Add its subtitle file too if you have one (.srt/.vtt) — that gives exact captions.',
    tab: 'Open the news (e.g. a YouTube live news stream) in <b>another Chrome tab</b> first. After pressing Start, choose that tab and tick <b>Also share tab audio</b>.',
    mic: 'Play the news out loud (TV, radio or laptop speakers) near the microphone. Quiet room = better captions.'
  }[src];
}
async function startLive(){
  if (active) return;
  help(''); hintShown = false; lastResultAt = 0; skippedTotal = 0; pausedByUs = false; lastFinal = '';
  const src = $('#liveSource').value, v = video();
  P.clearAll(); setCaption('');
  $('#liveTranscript').innerHTML = '<span class="sub">Captions will appear here.</span>';
  const subFile = $('#liveSubs').files[0], vidFile = $('#liveFile').files[0];
  if (src === 'file') {
    if (!vidFile) { help('Choose a video file first.'); return; }
    if (v.dataset.file !== vidFile.name + vidFile.size) { if (v.src) URL.revokeObjectURL(v.src); v.srcObject = null; v.src = URL.createObjectURL(vidFile); v.dataset.file = vidFile.name + vidFile.size; }
    v.muted = false; v.controls = true;
    $('#liveVideoBox').classList.remove('hidden');
    if (subFile) {
      cues = parseSubs(await subFile.text());
      if (!cues.length) { help('No captions found in that subtitle file (.srt or .vtt expected).'); return; }
      kind = 'subs'; active = true; cueIdx = cues.findIndex(c => c.end > v.currentTime); if (cueIdx < 0) cueIdx = cues.length;
      await v.play().catch(() => {}); rafId = requestAnimationFrame(cueLoop);
      $('#liveState').textContent = `Playing with subtitles (${cues.length} captions)`;
    } else {
      if (!SR) { help('Speech recognition needs Google Chrome. (Or add a subtitle file — that works in any browser.)'); return; }
      kind = 'file'; active = true;
      await v.play().catch(() => {});
      const stream = v.captureStream ? v.captureStream() : v.mozCaptureStream?.();
      track = stream?.getAudioTracks()[0] || await new Promise(res => { const to = setTimeout(() => res(null), 3000); stream?.addEventListener('addtrack', e => { if (e.track.kind === 'audio') { clearTimeout(to); res(e.track); } }); });
      if (!track) { help('This video has no sound track that the browser can read.'); stopLive(); return; }
      startRec(); $('#liveState').textContent = 'Listening to the video';
    }
  } else if (src === 'tab') {
    if (!SR) { help('Speech recognition needs Google Chrome.'); return; }
    try {
      shareStream = await navigator.mediaDevices.getDisplayMedia({ video: { displaySurface: 'browser' }, audio: true, preferCurrentTab: false, selfBrowserSurface: 'exclude', surfaceSwitching: 'include', systemAudio: 'include' });
    } catch (e) { help(e.name === 'NotAllowedError' ? 'Sharing was cancelled.' : 'Could not share a tab: ' + e.message); return; }
    track = shareStream.getAudioTracks()[0];
    if (!track) { shareStream.getTracks().forEach(t => t.stop()); shareStream = null; help('No sound was shared. Press Start again, pick the <b>Chrome tab</b> with the news, and tick <b>Also share tab audio</b>.'); return; }
    v.srcObject = shareStream; v.muted = true; v.controls = false; v.removeAttribute('src'); v.dataset.file = '';
    v.play().catch(() => {}); $('#liveVideoBox').classList.remove('hidden');
    shareStream.getVideoTracks()[0]?.addEventListener('ended', () => stopLive());
    track.addEventListener('ended', () => stopLive());
    kind = 'tab'; active = true; startRec(); $('#liveState').textContent = 'Listening to the shared tab';
  } else {
    if (!SR) { help('Speech recognition needs Google Chrome.'); return; }
    track = null; kind = 'mic'; active = true; $('#liveVideoBox').classList.add('hidden');
    startRec(); $('#liveState').textContent = 'Listening to the microphone';
  }
  startedAt = performance.now();
  $('#liveStart').textContent = '■ Stop'; $('#liveStart').classList.add('live');
  clearInterval(lagTimer); lagTimer = setInterval(manageLag, 250); manageLag();
}
export function stopLive(){
  const was = active; active = false;
  cancelAnimationFrame(rafId);
  try { rec && rec.abort(); } catch {} rec = null;
  if (shareStream) { shareStream.getTracks().forEach(t => t.stop()); shareStream = null; }
  track = null;
  const v = video();
  if (kind === 'tab') { v.srcObject = null; $('#liveVideoBox').classList.add('hidden'); }
  if (pausedByUs) pausedByUs = false;
  if (was && (kind === 'file' || kind === 'subs') && !v.paused) v.pause();
  kind = null; P && P.setBoost(1);
  $('#liveStart').textContent = '▶ Start'; $('#liveStart').classList.remove('live');
  $('#liveState').textContent = 'Stopped';
  clearInterval(lagTimer); lagTimer = null; $('#lagBadge').classList.add('hidden');
  setCaption('', '');
}
export function leaveLive(){ if (active) stopLive(); $('#liveVideoBox').classList.add('hidden'); setCaption(''); }
export function enterLive(){ if ($('#liveSource').value === 'file' && video().src) $('#liveVideoBox').classList.remove('hidden'); }

export function initLive(player){
  P = player;
  $('#liveSource').onchange = () => { if (active) stopLive(); setSourceUI(); };
  $('#liveStart').onclick = () => active ? stopLive() : startLive();
  $('#liveFile').onchange = () => { const f = $('#liveFile').files[0]; $('#liveFileName').textContent = f ? f.name : ''; const v = video(); if (f) { if (v.src) URL.revokeObjectURL(v.src); v.srcObject = null; v.src = URL.createObjectURL(f); v.dataset.file = f.name + f.size; v.controls = true; $('#liveVideoBox').classList.remove('hidden'); } };
  $('#liveSubs').onchange = () => { const f = $('#liveSubs').files[0]; $('#liveSubsName').textContent = f ? f.name : ''; };
  $('#liveLang').onchange = () => { if (active && rec && kind !== 'subs') { try { rec.abort(); } catch {} } };   // onend restarts with the new language
  video().addEventListener('seeked', onSeeked);
  video().addEventListener('ended', () => { if (active && (kind === 'file' || kind === 'subs')) setTimeout(() => { if (P.backlog() < 0.5) stopLive(); }, 1500); });
  setSourceUI();
  if (!SR) $('#liveSRNote').classList.remove('hidden');
}

// test hooks
window.__live = { parseSubs, onFinal, manageLag, get active(){ return active; }, get kind(){ return kind; }, get track(){ return track; } };
