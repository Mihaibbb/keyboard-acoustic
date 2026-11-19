"""Detector binar: vârful audio e o apăsare sau altceva (eliberare, zgomot, voce)?

Clasificatorul de taste primește doar ferestrele trecute de detector. Față de
clasa NONE_LABEL din același SVM, detectorul vede și contextul vârfului:
cât de tare e față de restul sesiunii și față de vecini, și la ce distanță
în timp sunt vecinii (eliberarea vine la ~100ms după apăsarea ei și e mai slabă).
"""
from pathlib import Path
import librosa
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from . import config as C
from .features import extract_features_batch
from .prepare import _session_dirs, estimate_offset, match_one_to_one, read_labels
from .segment import segment_audio

MAX_GAP_SEC = 1.0

def context_features(windows: np.ndarray, peak_times: np.ndarray) -> np.ndarray:
    """Per vârf: nivel relativ la sesiune, atac, și nivel/distanță față de 2 vecini pe fiecare parte."""
    n = len(peak_times)
    if n == 0:
        return np.zeros((0, 10), dtype=np.float32)
    w = windows.astype(np.float64)
    level = np.log(np.sqrt(np.mean(w ** 2, axis=1)) + 1e-9)
    level = level - np.median(level)          # independent de gain-ul microfonului
    head = w[:, : w.shape[1] // 4]
    attack = np.log(np.sqrt(np.mean(head ** 2, axis=1)) + 1e-9) - \
        np.log(np.sqrt(np.mean(w ** 2, axis=1)) + 1e-9)
    cols = [level, attack]
    for k in (1, 2):
        for sign in (-1, 1):
            idx = np.arange(n) + sign * k
            ok = (idx >= 0) & (idx < n)
            j = np.clip(idx, 0, n - 1)
            gap = np.where(ok, np.abs(peak_times[j] - peak_times), MAX_GAP_SEC)
            gap = np.minimum(gap, MAX_GAP_SEC)
            # Vecin lipsă sau prea departe: fără informație de nivel.
            dlevel = np.where(ok & (gap < MAX_GAP_SEC), level - level[j], 0.0)
            cols += [gap, dlevel]
    return np.stack(cols, axis=1).astype(np.float32)

def detector_features(windows: np.ndarray, peak_times: np.ndarray,
                      sr: int = C.SR, mfcc: np.ndarray = None) -> np.ndarray:
    if mfcc is None:
        mfcc = extract_features_batch(windows, sr=sr)
    return np.hstack([mfcc, context_features(windows, peak_times)])

def session_examples(session: Path, sr: int = C.SR):
    """(features, is_press) pentru o sesiune. Vârfurile ambigue (aproape de o
    etichetă, dar nepotrivite 1:1) sunt lăsate deoparte."""
    audio, _ = librosa.load(Path(session) / "audio.wav", sr=sr, mono=True)
    windows, peaks, _ = segment_audio(audio, sr=sr)
    peak_t = peaks / sr
    label_t, _ = read_labels(Path(session) / "labels.csv")
    shifted = label_t + estimate_offset(peak_t, label_t)
    matched = match_one_to_one(peak_t, shifted)
    feats = detector_features(windows, peak_t, sr=sr)
    y = np.full(len(peak_t), -1)
    for i, t in enumerate(peak_t):
        if i in matched:
            y[i] = 1
        elif len(shifted) == 0 or np.min(np.abs(shifted - t)) > C.NONE_MIN_DIST_SEC:
            y[i] = 0
    keep = y >= 0
    return feats[keep], y[keep]

def train_detector(raw_dir: Path, sr: int = C.SR, verbose: bool = True):
    X, y = [], []
    for session in _session_dirs(Path(raw_dir)):
        if not (session / "audio.wav").exists() or not (session / "labels.csv").exists():
            continue
        f, t = session_examples(session, sr=sr)
        X.append(f)
        y.append(t)
        if verbose:
            print(f"[{session.name}] apăsări: {int(t.sum())} | alte vârfuri: {int((t == 0).sum())}")
    if not X:
        raise RuntimeError("Nicio sesiune cu audio.wav + labels.csv.")
    X, y = np.vstack(X), np.concatenate(y)
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08,
                                           random_state=C.SEED)
    model.fit(X, y)
    return model, len(y)

def press_probability(detector, windows: np.ndarray, peak_times: np.ndarray,
                      sr: int = C.SR, mfcc: np.ndarray = None) -> np.ndarray:
    if len(peak_times) == 0:
        return np.zeros(0)
    return detector.predict_proba(detector_features(windows, peak_times, sr, mfcc))[:, 1]
