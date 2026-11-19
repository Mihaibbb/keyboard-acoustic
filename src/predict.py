"""Inferență pe un fișier audio nou."""
import librosa
import numpy as np
from . import config as C
from .features import features_for, predict_keys
from .segment import segment_audio

# Vârfuri cu probabilitate de apăsare sub pragul ăsta (detector binar) sunt
# eliberări de tastă / zgomot / voce și se aruncă. Vezi src/detector.py.
PRESS_THRESHOLD = 0.6

def predict_audio(model, audio_path, sr: int = C.SR, detector=None,
                  threshold: float = PRESS_THRESHOLD, lm=None, lam: float = 0.6):
    audio, _ = librosa.load(audio_path, sr=sr, mono=True)
    windows, sample_peaks, _ = segment_audio(audio, sr=sr)
    if windows.shape[0] == 0:
        return []
    feats = features_for(model, windows, sr=sr)
    preds, confs = predict_keys(model, feats)
    peak_t = sample_peaks / sr
    if detector is not None:
        from .detector import press_probability
        press_p = press_probability(detector, windows, peak_t, sr=sr)
    else:
        press_p = np.ones(len(sample_peaks))
    # Vârfurile păstrate (apăsare detectată), în ordinea timpului.
    kept = [i for i in range(len(sample_peaks))
            if str(preds[i]) != C.NONE_LABEL and press_p[i] >= threshold]
    keys = {i: str(preds[i]) for i in kept}
    # Cu model de limbaj: recombină întreg fluxul de taste prin beam search
    # (enter resetează contextul intern al LM-ului).
    if lm is not None and kept:
        from .decode import beam_decode
        classes = [str(c) for c in model.classes_]
        proba = model.predict_proba(feats)
        dkeys = beam_decode(proba[kept], classes, lm, lam=lam,
                            force_labels=[preds[i] for i in kept])
        keys = {i: dk for i, dk in zip(kept, dkeys)}
    return [
        {"time_sec": float(sample_peaks[i] / sr), "key": keys[i],
         "confidence": float(confs[i]), "press_prob": float(press_p[i])}
        for i in kept
    ]
