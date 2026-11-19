"""Filtrare band-pass, anvelopă RMS, detectare vârfuri, extragere ferestre."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, find_peaks
from . import config as C

def bandpass(audio: np.ndarray, sr: int = C.SR,
             low: float = C.LOWCUT, high: float = C.HIGHCUT) -> np.ndarray:
    nyq = sr / 2.0
    low = max(low, 1.0)
    high = min(high, nyq - 1.0)
    sos = butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")
    return sosfiltfilt(sos, audio.astype(np.float64))

def rms_envelope(audio: np.ndarray, win: int = C.RMS_WIN,
                 hop: int = C.RMS_HOP) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float64)
    if len(audio) < win:
        return np.array([np.sqrt(np.mean(audio ** 2))])
    n = 1 + (len(audio) - win) // hop
    csum = np.concatenate([[0.0], np.cumsum(audio ** 2)])
    starts = np.arange(n) * hop
    energy = (csum[starts + win] - csum[starts]) / win
    return np.sqrt(np.maximum(energy, 0.0))

def detect_peaks(envelope: np.ndarray, k: float = C.PEAK_MAD_K,
                 min_dist: int = C.PEAK_MIN_DIST,
                 hop: int = C.RMS_HOP) -> np.ndarray:
    """Vârfuri peste prag median + k·MAD.

    Pragul e robust (nu depinde de vârful maxim), deci un singur click
    foarte puternic / clipping nu face tastele mai slabe să dispară.
    """
    if len(envelope) == 0:
        return np.array([], dtype=np.int64)
    med = float(np.median(envelope))
    mad = float(np.median(np.abs(envelope - med))) + 1e-6
    thr = med + k * mad
    dist_bins = max(1, min_dist // hop)
    peaks, _ = find_peaks(envelope, height=thr, distance=dist_bins)
    return peaks.astype(np.int64)

def envelope_peak_to_sample(peaks: np.ndarray, hop: int = C.RMS_HOP,
                            win: int = C.RMS_WIN) -> np.ndarray:
    return (peaks * hop + win // 2).astype(np.int64)

def extract_windows(audio: np.ndarray, sample_peaks: np.ndarray,
                    window_samples: int = C.WINDOW_SAMPLES,
                    pre_samples: int = C.PRE_PEAK_SAMPLES) -> np.ndarray:
    """Fereastră [vârf - pre, vârf - pre + window); zero-padding la margini."""
    if len(sample_peaks) == 0:
        return np.zeros((0, window_samples), dtype=np.float32)
    padded = np.pad(audio, (window_samples, window_samples), mode="constant")
    out = np.zeros((len(sample_peaks), window_samples), dtype=np.float32)
    for i, p in enumerate(sample_peaks):
        start = int(p) + window_samples - pre_samples
        out[i] = padded[start: start + window_samples].astype(np.float32)
    return out

def segment_audio(audio: np.ndarray, sr: int = C.SR):
    filtered = bandpass(audio, sr=sr)
    env = rms_envelope(filtered)
    peak_idx = detect_peaks(env)
    sample_peaks = envelope_peak_to_sample(peak_idx)
    windows = extract_windows(filtered, sample_peaks)
    return windows, sample_peaks, filtered
