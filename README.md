# Emotion Insight Assistant

A standalone app pairing a facial-emotion CNN with a RAG/GenAI layer: the model
detects an expression, and a client can ask plain-English questions about the
detections ("was the focus group mostly positive this morning?") or get a grounded
daily summary — separate from the audio-domain [Audio Insight Assistant](../audio-insight-assistant)
project, built the same day using the same pattern, applied to a different sensing
modality (vision instead of audio) and a different, prior model (the MIT capstone CNN
instead of TFLite audio classifiers).

## The model: not invented for this app

`app/emotion_model.py` reconstructs **Model 3** from the completed MIT Professional
Education capstone (`Facial_Emotion_Detection.ipynb`) layer-for-layer: a 5-block CNN
(64-128-512-512-128 filters, batchnorm + LeakyReLU, two dense blocks) trained from
scratch on grayscale 48x48 faces. In the original notebook it was the best of six
architectures tried on the same 4-class dataset (happy / sad / neutral / surprise):

| Model | Test accuracy |
|---|---|
| Model 1 — baseline 3-block CNN | 66.4% |
| Model 2 — 4-block CNN, batchnorm + LeakyReLU | 71.1% |
| VGG16 (transfer learning, frozen) | 50.0% |
| ResNet101 (transfer learning, frozen) | 25.0% (chance level) |
| EfficientNetV2B2 (transfer learning, frozen) | 25.0% (chance level) |
| **Model 3 — 5-block CNN (this app)** | **77.3%** |

Per-class performance on the 128-image held-out test set (32 per class):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| happy | 0.81 | 0.91 | 0.85 |
| sad | 0.70 | 0.59 | 0.64 |
| neutral | 0.64 | 0.72 | 0.68 |
| surprise | 0.97 | 0.88 | 0.92 |

The three transfer-learning models underperforming a from-scratch CNN is a real,
documented finding from the notebook, not a bug here: those backbones are pretrained
at ~224x224; resizing down to this dataset's native 48x48 destroyed most of what
they'd learned. That comparison — and the honesty to report ResNet/EfficientNet
essentially failing rather than cherry-picking the winner — is carried into
`data/class_notes.md` so the GenAI layer can cite it accurately if asked.

**The trained weights are real and included.** `models/model3.weights.h5` is the
actual weights from a full 35-epoch retrain of Model 3 in the original Colab notebook
(the first session's checkpoint had been lost to a Colab runtime reset before it could
be exported — see `scripts/export_weights_from_colab.md` for the full story, including
a checkpoint-filename mixup along the way that made this take a bit of detective work).
Before trusting it, the file's embedded architecture was checked layer-by-layer against
`build_model3()` (identical, including the exact 1,782,340 param count pinned in
`tests/test_emotion_model.py`), and predictions from the original file were confirmed
bit-for-bit identical to predictions after converting it into this app's expected
weights format — also pinned as a regression test. `/predict` now runs on these real
weights: `weights_loaded: true` in every response, same as `/health`'s
`cnn_weights_loaded`. (The one thing not re-verified here: the 77% test accuracy figure
below is from the original notebook's held-out evaluation, not re-run in this sandbox,
since the labeled face dataset lives only in that Google Drive — this app ships the
trained model, not the dataset.)

## Architecture

```
POST /predict -> image in, CNN forward pass, {label, confidence, per_class, weights_loaded}
POST /ingest  -> structured EmotionEvent[] into an in-memory store
POST /ask     -> retrieval (TF-IDF over event text + class-reliability notes)
                  -> top-k grounding chunks -> LLM answers restricted to that context
GET  /report/daily -> same pipeline, fixed prompt, summarizes a session's day
```

Retrieval and LLM fallback (`app/retrieval.py`, `app/llm.py`) are the same
design as the audio-domain project: TF-IDF + cosine similarity for retrieval (no
model download, sub-second cold start), and a three-tier generation path — Azure
OpenAI, then OpenAI, then a deterministic extractive fallback — so the service is
fully demoable with zero external keys and always reports which mode produced an
answer.

One addition specific to this domain: the system prompt (`app/llm.py`) explicitly
instructs the model to frame detections as "the model observed an expression
consistent with X," never as a factual claim about how someone feels — facial
expression and internal emotional state aren't the same thing, and a client-facing
tool built on top of an emotion classifier should not blur that.

## Measured (this machine)

- `/ask` and `/report/daily`, offline fallback mode, 10 sample events: retrieval +
  generation **~4 ms** end-to-end (same order of magnitude as the audio project —
  retrieval cost doesn't depend on the domain).
- `/predict`, CNN forward pass on a 48x48 grayscale image, CPU, **real trained
  weights**, steady-state (average of 20 calls after a warm-up call): **~68 ms** for
  the forward pass alone, **~97 ms** full HTTP round trip including image decode.
  This replaces the earlier ~450 ms figure, which was measured on
  random-initialized weights without a warm-up call, so the two aren't a clean
  apples-to-apples comparison — the number that matters going forward is this one.

## Running it

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload

# in another terminal
python scripts/load_sample.py
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "Was the focus group mostly positive this morning?"}'
curl -X POST localhost:8000/predict -F "file=@some_face.jpg"
```

To use a real LLM instead of the offline fallback, copy `.env.example` to `.env` and
fill in Azure OpenAI (or OpenAI) credentials. The real trained CNN weights already
ship in `models/model3.weights.h5` — no extra setup needed; see
`scripts/export_weights_from_colab.md` only if they ever need to be regenerated.

### Docker

```bash
docker build -t emotion-insight-assistant .
docker run -p 8000:8000 --env-file .env emotion-insight-assistant
```

(Written and structurally checked; not run in this sandbox, which has no Docker
daemon — build locally to confirm before demoing, same caveat as the audio project.)

### Tests

```bash
pytest tests/ -v   # 11 passed
```

`tests/test_emotion_model.py` pins the architecture (exact param count from the
notebook's `model3.summary()`), confirms the preprocessing pipeline produces a valid
softmax distribution for square and non-square input images, and — now that real
weights are shipped — checks they actually load (`weights_loaded is True`), that
predictions are deterministic and meaningfully more confident than chance, and that
converting the original full-model file into this app's weights format didn't change
a single prediction (bit-for-bit regression check). `tests/test_api.py` covers the
full API surface including a real image POSTed through `/predict`.

## What I'd do next with more time

- Get a labeled held-out face set into this repo (or a public equivalent) so accuracy
  can be re-verified here directly, instead of citing the original notebook's
  evaluation — the weights are real and verified to load correctly, but this sandbox
  never had a labeled test image to score them against.
- Quantize Model 3 to TFLite and benchmark it the same way ModelBench benchmarks the
  audio classifiers (p50/p95 latency, size, int8 vs float16) — same methodology,
  new modality, natural next step given the on-device quantization background.
- Replace TF-IDF with Azure OpenAI embeddings once the corpus is bigger than a
  handful of sessions.
- Add per-session confidence-drift alerts (e.g. flag a session where 'sad' detections
  cluster unusually, subject to the ethics note above about not overclaiming).
