# keyboard-acoustic

Educational tool for **keyboard acoustic side-channel recognition**: reconstruct the
text someone typed from a microphone recording of the keystrokes alone. The pipeline
segments individual key presses out of the audio, rejects release/noise sounds, classifies
each press with a model ensemble, and optionally recomposes the stream with a language
model to fix neighbour-key confusions.

> ## ⚠️ Legal / ethical use
> Use this **EXCLUSIVELY on your own equipment and your own recordings, with consent.**
> Recording or reconstructing another person's typing without their explicit consent is
> illegal in most jurisdictions and a serious violation of privacy. This project exists for
> **education and defensive security research** — understanding the side channel so it can be
> mitigated. Do not use it to attack anyone.

---

## What it does

Given a `.wav` of someone typing, the tool answers two questions, measured separately
because they are genuinely different problems:

- **Per-key accuracy** — of the key presses the segmenter *finds*, how many does the
  classifier label correctly? (a pure classification metric)
- **Blind char accuracy** — text reconstructed from audio **alone** (segment → detect
  press → classify → apply backspace/enter), compared to the keylogger ground truth by
  edit distance. This is the realistic end-to-end metric and is always lower, because it
  also pays for missed presses (deletions) and spurious peaks (insertions).

### Results (held-out sessions, same keyboard/user, unseen at training)

| Metric | Baseline (SVM) | Current stack (ensemble + LM decode) |
|---|---|---|
| Per-key accuracy | 87.8% | **90.0%** |
| Blind char accuracy | 79.4% | **81.0%** |

Numbers are a weighted average over three holdout sessions (`mix_01`, `val_01`, `val_02`).
The binary press/non-press **detector** is what makes blind reconstruction viable at all:
it keeps ~98% of real presses while dropping ~95% of release/noise peaks, so the classifier
isn't labelling every door-click as a letter.

> **Cross-session vs within-session.** These numbers are *cross-session*: trained on some
> recordings, tested on different ones from the same keyboard/user. Published "~98%" figures
> are usually *within-session* (train and test drawn from the same recording) and are not
> comparable — cross-session is the harder, more honest setting.

---

## How it works

```
audio.wav
   │  band-pass 100–8000 Hz
   ▼
RMS envelope ──► MAD peak detection          (find candidate keystroke times)
   │
   ▼
25 ms windows (10 ms before the peak)         (short enough to not catch the next key)
   │
   ▼
log-mel spectrogram  (64 bands × 5 frames = 320 dims)
   │  StandardScaler
   ▼
binary detector  ──►  drop release / noise peaks   (optional, --detector)
   │
   ▼
ensemble classifier  (SVM-RBF + MLP + HistGradientBoosting, soft vote)
   │
   ▼
language-model decode  (char-trigram + beam search)   (optional, --decode)
   │
   ▼
reconstructed text
```

A 25 ms window is deliberate: at fast typing (<80 ms between keys) an 80 ms window caught
the *start of the next key* and confused neighbours. 25 ms covers the key's attack transient,
which is where the distinguishing information lives.

The **ensemble** beat a single bigger model — a CNN on the same log-mel features *overfit*
cross-session (0.78). The win is model diversity, not depth. Full experiment log in
[`reports/experiments.md`](reports/experiments.md).

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quick demo (no microphone)

```bash
python scripts/demo.py
```

Generates synthetic audio for 5 keys (a, s, d, f, g), then runs prepare → train → evaluate →
predict on `examples/sample.wav`. Typical accuracy > 0.95 (synthetic is easy).

---

## Full workflow (CLI)

All commands run as `python -m src.cli <command>`.

### 1. Record (needs a microphone + consent)

```bash
python -m src.cli record --output data/raw/session_01 --duration 60
```

Writes `labels.csv` (every press, from a keylogger), `keyup.csv` (every key release), and
`final_text.csv` (the resulting text after backspace/enter). These are **ground truth** —
the model only ever sees `audio.wav`. For older sessions without `final_text.csv`:
`python -m src.cli text --raw data/raw`.

### 2. Prepare (segment + align peaks with labels)

```bash
python -m src.cli prepare --raw data/raw --out data/processed
```

### 3. Train

```bash
python -m src.cli train --data data/processed --model models/baseline.joblib --final
```

Each window is also presented shifted by ±2.5 ms during training (the peak position has a
few-ms imprecision); worth ~1 point. `--no-augment` turns it off and trains ~10× faster.
`--final` retrains the saved model on *all* data after measuring on a 20% validation split —
for an honest number, use a separate session via `evaluate --holdout` or `compare`.

The current best classifier is the ensemble at `models/ensemble_trio.joblib`. `compare`,
`predict`, and `evaluate` load it transparently (via `ensemble.load_any`), so it drops into
the pipeline wherever a plain model would go.

### 4. Train the press/non-press detector

```bash
python -m src.cli detector --raw data/raw --out models/detector.joblib
```

The key classifier receives a peak for every press **and** every release/noise, and will
label them all. This binary detector (gradient boosting on MFCC + relative level + timing to
neighbours) throws out the non-press peaks before classification. Essential on unlabelled audio.

### 5. Evaluate (writes `reports/`)

```bash
python -m src.cli evaluate --model models/ensemble_trio.joblib --data data/holdout/val_01 --holdout
```

### 6. Predict on a new audio file

```bash
python -m src.cli predict --model models/ensemble_trio.joblib \
    --detector models/detector.joblib --decode --audio examples/sample.wav
```

`--decode` recomposes the key stream with a language model (beam search over the acoustic
probabilities), `--lam` weights it (default 0.6).

### 7. Compare: real text vs text reconstructed from audio

```bash
python -m src.cli compare --model models/ensemble_trio.joblib \
    --detector models/detector.joblib --decode --session data/holdout/mix_01
```

Writes `predicted_text.csv`, `keys_vs_pred.csv`, and `text_vs_model.txt` into the session
folder. `--session` accepts a single session or a folder of sessions (e.g. `data/raw`).
Prints per-session per-key, known-times, and blind accuracy.

---

## Tests

```bash
pytest tests/ -v
```

## Project layout

```
src/
  record.py      capture audio + keylogger labels
  segment.py     band-pass → envelope → MAD peak detection
  segment2.py    experimental spectral-flux onset detector (higher recall, see notes)
  features.py    log-mel / MFCC feature extraction
  detector.py    binary press vs non-press classifier
  ensemble.py    soft-vote SVM + MLP + HistGradientBoosting
  decode.py      char-trigram LM + beam-search shallow fusion
  predict.py     inference on a new .wav
  compare.py     real text vs audio-reconstructed text
  train.py / evaluate.py / prepare.py / cli.py
reports/         classification reports, confusion matrices, experiments.md
demo_data/       tiny bundled dataset for the demo
examples/        sample.wav + generator
```

Recordings and trained models (`data/`, `models/`) are gitignored — they are large and
specific to one keyboard/user.

---

## Limitations

- Adapts to a **single keyboard / user**. Accuracy collapses on a different keyboard without
  that keyboard's data.
- Needs roughly **~20 samples per key** for good accuracy. Rare keys with 1–3 samples are
  effectively never recognised.
- Current alphabet covers **~30 classes** (a–z + space / backspace / enter / shift). **Digits
  and punctuation are not yet modelled** — text containing them cannot be fully reconstructed.
- Strong background noise degrades accuracy.
- Not robust to very loud mechanical keyboards (baseline features).

## Roadmap

The current blind-text bottleneck is **detector precision** (release/noise peaks that slip
through become insertions), not the classifier (90%) or segmentation recall (98% available via
`segment2.py`). Next levers, in order of expected payoff:

1. **Better press/non-press detector** — sharper attack-transient features, probability
   calibration, per-session thresholds. Unlocks the higher-recall `segment2` segmenter, which
   currently regresses blind text because its extra peaks become insertions.
2. **Word-level language model** instead of the char-trigram, with a real lexicon.
3. **Digit + punctuation coverage** — requires targeted recordings.

### Prior art / references

- CNN on mel-spectrograms — [`k3rnel-pan1c-ksd/automated-keystroke-acoustic-attack`](https://github.com/k3rnel-pan1c-ksd/automated-keystroke-acoustic-attack) (98.1% top-1, within-session).
- Hybrid CoAtNet — [`MomirMilutinovic/keyboard-acoustic-side-channel-attack-coatnet`](https://github.com/MomirMilutinovic/keyboard-acoustic-side-channel-attack-coatnet).
- SpecAugment + per-user adaptation with ~13 samples/key.
