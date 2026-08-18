# Getting the real trained weights into this app

This app's CNN architecture (`app/emotion_model.py: build_model3`) is an exact port of
"Model 3" from `Facial_Emotion_Detection.ipynb`. The architecture code runs today, but
the *trained weights* are wherever that Colab session saved them
(`model3.weights.h5`, per the notebook's `ModelCheckpoint` callback), not in this repo.

To wire up real predictions:

1. Open the original Colab notebook (or re-run training if the runtime was reset).
2. If `model3.weights.h5` already exists in the Colab session, download it directly:
   ```python
   from google.colab import files
   files.download("model3.weights.h5")
   ```
3. If it's gone (Colab runtimes reset), re-run cells 89-97 (data loaders through
   `model3.evaluate(test_set)`) to retrain -- the checkpoint callback saves the best
   epoch automatically to `model3.weights.h5`, then download it as above.
4. Place the downloaded file at `models/model3.weights.h5` in this repo.
5. Restart the app (`uvicorn app.main:app --reload`). `/health` should now report
   `"cnn_weights_loaded": true`, and `/predict` will return real predictions instead of
   near-uniform ~25%-per-class output from random initialization.

Until this is done, `/predict` still runs end-to-end (useful for confirming the API
contract and image preprocessing work), but its output is not a meaningful emotion
prediction -- treat it as a plumbing check, not a demo of model accuracy.
