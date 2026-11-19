"""Extragere MFCC + Δ + ΔΔ, rezumate prin medie și deviație standard."""
import numpy as np
import librosa
from . import config as C

FEAT_DIM = C.N_MFCC * 3 * 2  # 120

def _safe_delta(mfcc: np.ndarray):
    """Δ și ΔΔ, cu lățimea limitată la numărul de cadre.

    librosa.feature.delta cere width impar ≤ nr. cadre; fereastra scurtă (25ms)
    dă doar ~5 cadre, iar implicitul width=9 arunca eroare. Sub 3 cadre nu se
    poate estima o pantă, deci Δ devine zero (forma statică rămâne în MFCC)."""
    n = mfcc.shape[1]
    if n < 3:
        z = np.zeros_like(mfcc)
        return z, z
    width = min(9, n if n % 2 else n - 1)
    delta = librosa.feature.delta(mfcc, width=width)
    delta2 = librosa.feature.delta(mfcc, width=width, order=2)
    return delta, delta2

def extract_features(window: np.ndarray, sr: int = C.SR) -> np.ndarray:
    if window.size == 0:
        return np.zeros(FEAT_DIM, dtype=np.float32)
    # Normalizare per apăsare: sesiunile au niveluri diferite (gain microfon,
    # distanță), iar MFCC depind de nivel. Contează forma spectrului, nu volumul.
    window = window.astype(np.float32)
    window = window / (np.max(np.abs(window)) + 1e-9)
    mfcc = librosa.feature.mfcc(y=window, sr=sr,
                                 n_mfcc=C.N_MFCC, n_fft=C.N_FFT,
                                 hop_length=C.HOP_LENGTH)
    delta, delta2 = _safe_delta(mfcc)
    feats = np.vstack([mfcc, delta, delta2])
    return np.concatenate([feats.mean(axis=1), feats.std(axis=1)]).astype(np.float32)

def extract_features_batch(windows: np.ndarray, sr: int = C.SR) -> np.ndarray:
    if windows.shape[0] == 0:
        return np.zeros((0, FEAT_DIM), dtype=np.float32)
    return np.stack([extract_features(w, sr) for w in windows])

# --- Spectrogramă log-mel întreagă -------------------------------------------
# Rezumatul de mai sus (medie + deviație în timp) șterge forma în timp a
# apăsării; tastele vecine au spectru mediu aproape identic și diferă mai ales
# prin cum evoluează sunetul în cele 80ms. Aici se păstrează toate cadrele.
N_MELS = 64
MEL_FMIN = 100.0
MEL_FMAX = 10000.0
MEL_FRAMES = 1 + C.WINDOW_SAMPLES // C.HOP_LENGTH
MEL_DIM = N_MELS * MEL_FRAMES

def extract_mel_features(window: np.ndarray, sr: int = C.SR) -> np.ndarray:
    if window.size == 0:
        return np.zeros(MEL_DIM, dtype=np.float32)
    window = window.astype(np.float32)
    window = window / (np.max(np.abs(window)) + 1e-9)
    if len(window) != C.WINDOW_SAMPLES:      # dimensiune fixă a vectorului
        window = librosa.util.fix_length(window, size=C.WINDOW_SAMPLES)
    mel = librosa.feature.melspectrogram(y=window, sr=sr, n_fft=C.N_FFT,
                                         hop_length=C.HOP_LENGTH, n_mels=N_MELS,
                                         fmin=MEL_FMIN, fmax=MEL_FMAX)
    return librosa.power_to_db(mel, ref=np.max, top_db=80.0).ravel().astype(np.float32)

def extract_mel_features_batch(windows: np.ndarray, sr: int = C.SR) -> np.ndarray:
    if windows.shape[0] == 0:
        return np.zeros((0, MEL_DIM), dtype=np.float32)
    return np.stack([extract_mel_features(w, sr) for w in windows])

FEATURE_KINDS = {"summary": extract_features_batch, "mel": extract_mel_features_batch}

def features_for(model, windows: np.ndarray, sr: int = C.SR) -> np.ndarray:
    """Features în formatul cu care a fost antrenat modelul (cele vechi: rezumat MFCC)."""
    return FEATURE_KINDS[getattr(model, "feature_kind_", "summary")](windows, sr)

def predict_keys(model, feats: np.ndarray):
    """(taste, încredere). Eticheta vine din decizia SVM-ului; probabilitățile
    (Platt) pot da alt argmax și sunt puțin mai slabe, deci rămân doar ca încredere."""
    probs = model.predict_proba(feats)
    keys = np.asarray(model.predict(feats)).astype(str)
    col = {str(c): i for i, c in enumerate(model.classes_)}
    conf = probs[np.arange(len(keys)), [col[k] for k in keys]]
    return keys, conf
