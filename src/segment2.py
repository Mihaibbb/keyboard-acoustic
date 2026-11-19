"""Detector de onset îmbunătățit (flux spectral) pentru tastare rapidă.

Motivație
---------
Detectorul din ``segment.py`` folosește anvelopa RMS + prag median+k·MAD.
Când două apăsări sunt mai apropiate de ~80 ms, lobii lor de energie se
contopesc și ``find_peaks`` vede un singur vârf — a doua apăsare se pierde.

``segment2.py`` înlocuiește pasul de detectare cu **flux spectral**
(``librosa.onset.onset_strength``): acesta răspunde la *schimbarea* spectrului
(atacul tastei), nu la energia netedă, deci rezolvă două atacuri apropiate chiar
dacă energia lor se suprapune. Un hop fin (256 ≈ 5.8 ms) permite vârfuri la
~25 ms distanță. Peste asta punem o **poartă de energie blândă**: un vârf de flux
e păstrat doar dacă RMS local (max pe o mică fereastră în jurul lui) depășește
median+egate·MAD. Asta elimină micile vârfuri de flux din liniște/zgomot fără a
atinge apăsările reale (care sunt mult mai puternice).

API identic cu ``segment.py`` (``segment_audio`` → windows, sample_peaks,
filtered), deci e un înlocuitor direct pentru ``prepare.py``, ``predict.py``,
``compare.py``, ``detector.py``. Nu modifică ``segment.py`` sau ``config.py``.

Parametri acordați pe val_02 + session_06 (cazurile cu cea mai mare pondere de
pauze <80 ms). Vezi scriptul de evaluare din scratchpad pentru tabelul complet.
"""
import warnings
import numpy as np
import librosa
from scipy.signal import find_peaks

from . import config as C
# Reutilizăm band-pass-ul și extragerea ferestrelor din detectorul original,
# ca pre/post-procesarea să fie identică — diferă doar pasul de detectare.
from .segment import bandpass, extract_windows

# ---- Parametri acordabili -------------------------------------------------
# Pas fin pentru anvelopa de onset: 256 eșantioane ≈ 5.8 ms la 44.1 kHz.
ONSET_HOP = 256
# Fereastra STFT pentru fluxul spectral.
ONSET_NFFT = 1024
# Pragul pe anvelopa de onset: median + k·MAD (robust, ca în segment.py).
ONSET_MAD_K = 3.5
# Distanța minimă între vârfuri. 25 ms < pauza minimă reală dintre taste, deci
# apăsările rapide (40-80 ms) nu mai fuzionează (segment.py folosea 50 ms).
ONSET_MIN_DIST_MS = 25.0
# Poarta de energie: păstrează vârful doar dacă RMS local max ≥ median+egate·MAD.
# Blândă (0.6) — taie doar fluxul din liniște, nu apăsările slabe. 0 = dezactivat.
ENERGY_GATE_K = 0.6
# Fereastra (ms) în care căutăm maximul RMS în jurul vârfului de flux, pentru
# poartă. Atacul are energia maximă la câteva ms după debutul fluxului.
ENERGY_GATE_POST_MS = 12.0


def onset_envelope(filtered: np.ndarray, sr: int = C.SR,
                   hop: int = ONSET_HOP, n_fft: int = ONSET_NFFT) -> np.ndarray:
    """Anvelopa de flux spectral (tărie de onset) pe audio-ul band-pass."""
    y = np.ascontiguousarray(filtered, dtype=np.float32)
    with warnings.catch_warnings():
        # n_fft mare + sr=44.1k produce filtre mel înalte goale (semnalul e deja
        # filtrat <8 kHz, deci contribuie ~0) — avertisment inofensiv.
        warnings.simplefilter("ignore")
        return librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop,
                                            n_fft=n_fft)


def detect_onset_samples(filtered: np.ndarray, sr: int = C.SR,
                         hop: int = ONSET_HOP, n_fft: int = ONSET_NFFT,
                         k: float = ONSET_MAD_K,
                         min_dist_ms: float = ONSET_MIN_DIST_MS,
                         energy_gate_k: float = ENERGY_GATE_K,
                         gate_post_ms: float = ENERGY_GATE_POST_MS) -> np.ndarray:
    """Vârfuri de onset, returnate ca indici de **eșantion** în ``filtered``.

    1. flux spectral → anvelopă de onset (hop fin);
    2. ``find_peaks`` peste median + k·MAD, cu distanță minimă mică;
    3. poartă de energie blândă: elimină vârfurile de flux din liniște.
    """
    filtered = np.asarray(filtered)
    if filtered.size == 0:
        return np.array([], dtype=np.int64)

    oenv = onset_envelope(filtered, sr=sr, hop=hop, n_fft=n_fft)
    if oenv.size == 0:
        return np.array([], dtype=np.int64)

    med = float(np.median(oenv))
    mad = float(np.median(np.abs(oenv - med))) + 1e-9
    thr = med + k * mad
    dist_bins = max(1, int(round(min_dist_ms / 1000.0 * sr / hop)))
    peaks, _ = find_peaks(oenv, height=thr, distance=dist_bins)

    if energy_gate_k > 0 and len(peaks):
        rms = librosa.feature.rms(y=np.ascontiguousarray(filtered, dtype=np.float32),
                                  frame_length=2 * hop, hop_length=hop,
                                  center=True)[0]
        rmed = float(np.median(rms))
        rmad = float(np.median(np.abs(rms - rmed))) + 1e-12
        rthr = rmed + energy_gate_k * rmad
        w = max(1, int(round(gate_post_ms / 1000.0 * sr / hop)))
        lo = np.clip(peaks - 1, 0, len(rms) - 1)
        hi = np.clip(peaks + w + 1, 1, len(rms))
        keep = np.array([rms[a:b].max() >= rthr for a, b in zip(lo, hi)])
        peaks = peaks[keep]

    # Frame de onset → eșantion. STFT centrat: frame t ≈ eșantion t·hop.
    return (peaks.astype(np.int64) * hop)


def segment_audio(audio: np.ndarray, sr: int = C.SR):
    """Înlocuitor direct pentru ``segment.segment_audio``.

    Returnează ``(windows, sample_peaks, filtered)`` cu aceeași semnificație:
    ferestre fixe în jurul fiecărui vârf detectat, pozițiile vârfurilor în
    eșantioane, și semnalul band-pass.
    """
    filtered = bandpass(audio, sr=sr)
    sample_peaks = detect_onset_samples(filtered, sr=sr)
    windows = extract_windows(filtered, sample_peaks)
    return windows, sample_peaks, filtered
