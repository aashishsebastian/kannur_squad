ISL Avatar prototype - laptop version
=====================================

HOW TO START
  1. Double-click  start_app.bat
  2. A small black window opens (that's the local server - keep it open)
     and the app opens in Chrome (or Edge) at http://localhost:8765
  3. To stop the app, close the black window.

TWO INPUT MODES
  Text   - type a sentence and press "Sign it" (or Ctrl+Enter).
  Speech - press "Start listening" and speak. Allow the microphone when the
           browser asks. Speech recognition needs Chrome or Edge and an
           internet connection.

  Words the avatar knows are signed and highlighted in blue as it signs.
  Words it doesn't know yet are shown in amber and are NOT signed.

LIVE - NEWS / VIDEO TO SIGNING (fourth tab)
  Pick a source, press Start, and the avatar signs what is said while live
  captions show everything:
    Video file  - a news video saved on this laptop. Add its subtitle file
                  (.srt/.vtt) if you have one: exact captions, works offline.
    Chrome tab  - open YouTube live news (or any site) in another tab first;
                  after Start choose that tab and tick "Also share tab audio".
    Microphone  - TV or radio playing in the room.
  "When signing falls behind": keep up (speed up / skip words), pause the
  video until signing catches up (video files), or sign everything.
  Needs Google Chrome + internet for speech recognition (not for subtitles).

CAMERA - SIGN TO TEXT (third tab)
  1. Click "Start camera" and press Allow when Chrome asks for the camera.
     (If you clicked Block by mistake: click the camera icon in the address
     bar, choose Allow, then press Start camera again.)
  2. Sit about an arm's length away, head + chest + both hands in view,
     facing a light. Blue lines = your right hand, amber = left hand.
  3. TEACH first (the app has no built-in ISL knowledge yet):
       "Teach signs" -> pick a sign -> Record sample (or press Space)
       -> sign after the 3-2-1. Do each sign 5 times, moving slightly.
       Teach at least 2 signs. "Check accuracy" tests your recordings.
  4. RECOGNISE: "Sign now" (or Space), sign, and the app shows its best
     guess with the top 3. Press Enter to confirm (it speaks the word), or
     click the right word to correct it (it learns from corrections).
  Tick "Automatic" to skip the button: it starts when your hands rise and
  stops when you lower them.
  Privacy: video never leaves the laptop and is never saved. Only the hand
  positions of signs you teach are stored, in this browser. Export/Import
  lets you back them up or move them to another computer.

TRAINED MODEL (recognise signs for anyone, no teaching)
  See TRAINING.md: run train_isl_model.ipynb on free Google Colab, then unzip
  the result into this folder (model\isl_model.onnx + model\isl_model.json).

OTHER CONTROLS
  Speed slider, Replay, Stop, Front / Angled view, drag on the avatar to
  rotate the camera, scroll to zoom, click any word in the lexicon list to
  see that sign alone.

FILES
  index.html   the app
  avatar.glb   the 3D character (exported from Zephyr_ISL_test.blend, with
               the bone hierarchy and colours repaired for the web)
  camera.js / signmath.js   webcam sign recognition
  live.js      live news / video mode
  train_isl_model.ipynb, TRAINING.md, training/   model training (Colab)
  model/       your trained model goes here (created by you, see TRAINING.md)
  lib/         three.js (MIT), MediaPipe hand/pose tracking (Apache 2.0) and
               ONNX Runtime Web (MIT) - everything works offline except
               speech-to-text
  serve.ps1 / start_app.bat   local server launcher

IMPORTANT LIMITS
  - The 29 signs are approximate placeholders for testing the pipeline. They
    must be checked and corrected by fluent Deaf ISL users before real use.
  - Word order follows English, not ISL grammar. No facial grammar yet.
  - Without a trained model, camera recognition only knows the signs you
    teach it and works best for the person who recorded them.
  - The character is a DreamWorks design: private testing only. Swap in an
    original character before any demo or release.
