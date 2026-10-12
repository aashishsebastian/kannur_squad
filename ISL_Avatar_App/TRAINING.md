# Training the sign-recognition model (no sign language knowledge needed)

The app can recognise ISL words **for anyone** once you give it a trained model. You make the model for free
on Google Colab with the notebook in this folder: `train_isl_model.ipynb`. It learns from the **INCLUDE**
dataset: videos of 263 ISL words signed by Deaf students of St. Louis School for the Deaf, Chennai
(IIT Madras / AI4Bharat).

## Steps

1. Open https://colab.research.google.com and sign in with your Google account.
2. **File → Upload notebook** and choose `train_isl_model.ipynb` from this folder.
3. **Runtime → Change runtime type → T4 GPU → Save** (free; makes training fast).
4. In cell 2, choose **Quick start** (5 categories, ~19 GB download) for a first try, or **Everything** (all 263 signs, ~57 GB).
5. **Runtime → Run all.** When asked, allow access to Google Drive (progress is saved there).
6. Wait. Quick start takes about 30–45 minutes, Everything about 1.5–2.5 hours. Most of that is downloading and reading the videos.
   If Colab disconnects, just do **Run all** again: it skips everything already finished.
7. At the end, `isl_model.zip` downloads (a copy is also in Drive → `ISL_training`).
   **Unzip it into this `ISL_Avatar_App` folder**, so you get `model\isl_model.onnx` and `model\isl_model.json`.
8. Restart the app (`start_app.bat`) → Camera tab. It should say **"Trained model loaded: N signs"**. Click *See signs* for the list.

## Reading the results (cell 6)
* **Correct first guess**: accuracy on videos the model never saw while training.
* **Left-handed (mirrored)**: the same test with videos mirrored, as if signed left-handed.
* INCLUDE doesn't record which student signed each video, so the same students appear in training and test videos.
  Real accuracy for new people, cameras and rooms will be lower. Test it with real signers and use **Teach signs**
  in the app to add your own examples for words that fail.

## In the app
"Recognise using" lets you pick **Trained model + my taught signs** (default), **Trained model only**, or **My taught signs only**.
Your own taught signs win when they are a confident match. Words you correct are learned as taught signs.

## Credit (required by the dataset licence, CC BY 4.0)
INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition. A. Sridhar, R. G. Ganesan, P. Kumar, M. Khapra.
ACM Multimedia 2020. https://zenodo.org/records/4010759
Mention this wherever you show or share the trained model (demo, report, app "about" page).

## Files
* `train_isl_model.ipynb`: the notebook (self-contained).
* `training/islfeat.py`: feature maths; identical to `signmath.js` in the app (checked number-for-number).
* `training/isltrain.py`: video reading, augmentation, model, training, export.
* `training/make_notebook.py`: rebuilds the notebook from the two files above.
