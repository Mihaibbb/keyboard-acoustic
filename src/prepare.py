"""Aliniază ferestrele detectate cu etichetele ground-truth."""
import csv
from pathlib import Path
import librosa
import numpy as np
from . import config as C
from .segment import segment_audio
from .train import save_processed

# Caracter cu Shift → tasta fizică (layout US). "!" și "1" sunt aceeași tastă,
# deci trebuie să fie aceeași clasă.
_SHIFTED = dict(zip('~!@#$%^&*()_+{}|:"<>?', "`1234567890-=[]\\;',./"))

def normalize_key(key: str) -> str:
    if len(key) == 1:
        key = _SHIFTED.get(key, key)
        return key.lower()
    return key

def read_labels(labels_path: Path):
    times, keys = [], []
    with open(labels_path) as f:
        for row in csv.DictReader(f):
            times.append(float(row["timestamp_sec"]))
            keys.append(normalize_key(row["key"]))
    return np.array(times, dtype=np.float64), keys

def estimate_offset(peak_times: np.ndarray, label_times: np.ndarray,
                    search: float = C.OFFSET_SEARCH_SEC,
                    step: float = C.OFFSET_STEP_SEC,
                    tol: float = C.TOLERANCE_SEC) -> float:
    """Decalajul o pentru care peak ≈ label + o se potrivesc cel mai bine.

    Ceasul etichetelor și ceasul audio nu pornesc simultan (latența de
    pornire a stream-ului audio, ~60-100ms pe macOS), iar decalajul diferă
    de la sesiune la sesiune. La egalitate se preferă decalajul cel mai mic.
    """
    if len(peak_times) == 0 or len(label_times) == 0:
        return 0.0
    pk = np.sort(np.asarray(peak_times, dtype=np.float64))

    def nearest_dist(shifted):
        if len(pk) == 1:
            return np.abs(pk[0] - shifted)
        idx = np.clip(np.searchsorted(pk, shifted), 1, len(pk) - 1)
        return np.minimum(np.abs(pk[idx] - shifted), np.abs(pk[idx - 1] - shifted))

    n = int(round(search / step))
    grid = np.arange(-n, n + 1) * step          # conține 0.0 exact
    grid = grid[np.argsort(np.abs(grid), kind="stable")]
    best_off, best_score = 0.0, -1.0
    for off in grid:
        # Scor „soft”: o potrivire perfectă valorează 1, una la limita tol ~0.
        # Un prag hard ar avea un platou lat de 2·tol și ar alege marginea lui.
        d = nearest_dist(label_times + off)
        score = float(np.sum(np.clip(1.0 - d / tol, 0.0, None)))
        if score > best_score + 1e-9:
            best_off, best_score = float(off), score
    # Rafinare sub pasul grilei: mediana reziduurilor perechilor potrivite.
    shifted = label_times + best_off
    idx = np.clip(np.searchsorted(pk, shifted), 0, len(pk) - 1)
    cand = np.stack([pk[idx], pk[np.maximum(idx - 1, 0)]])
    nearest = cand[np.argmin(np.abs(cand - shifted), axis=0), np.arange(len(shifted))]
    resid = nearest - shifted
    resid = resid[np.abs(resid) <= tol]
    if len(resid):
        best_off += float(np.median(resid))
    return best_off

def match_one_to_one(peak_times: np.ndarray, label_times: np.ndarray,
                     tol: float = C.TOLERANCE_SEC):
    """Potrivire greedy 1:1 (cele mai apropiate perechi întâi).

    Returnează {index_vârf: index_etichetă}. Fiecare etichetă și fiecare
    vârf sunt folosite cel mult o dată.
    """
    if len(peak_times) == 0 or len(label_times) == 0:
        return {}
    d = np.abs(peak_times[:, None] - label_times[None, :])
    pi, li = np.nonzero(d <= tol)
    order = np.argsort(d[pi, li], kind="stable")
    used_p, used_l, out = set(), set(), {}
    for o in order:
        p, l = int(pi[o]), int(li[o])
        if p in used_p or l in used_l:
            continue
        used_p.add(p)
        used_l.add(l)
        out[p] = l
    return out

def _session_dirs(raw_dir: Path):
    if (raw_dir / "audio.wav").exists():
        return [raw_dir]
    return [s for s in sorted(raw_dir.iterdir()) if s.is_dir()]

def prepare_sessions(raw_dir: Path, out_dir: Path, sr: int = C.SR,
                     include_none: bool = False, verbose: bool = True):
    """include_none: adaugă vârfurile fără apăsare (release/zgomot) ca clasa
    NONE_LABEL, ca predict să le poată filtra. Experimental: cu puține date
    nu generalizează între sesiuni și „înghite” apăsări reale."""
    raw_dir = Path(raw_dir)
    X, y = [], []
    for session in _session_dirs(raw_dir):
        audio_path = session / "audio.wav"
        labels_path = session / "labels.csv"
        if not audio_path.exists() or not labels_path.exists():
            continue
        audio, _ = librosa.load(audio_path, sr=sr, mono=True)
        windows, peaks, _ = segment_audio(audio, sr=sr)
        label_times, label_keys = read_labels(labels_path)
        peak_times = peaks / sr

        offset = estimate_offset(peak_times, label_times)
        shifted = label_times + offset
        matches = match_one_to_one(peak_times, shifted)
        n_none = 0
        for i, pt in enumerate(peak_times):
            if i in matches:
                X.append(windows[i])
                y.append(label_keys[matches[i]])
            elif include_none and (len(shifted) == 0 or
                                   np.min(np.abs(shifted - pt)) > C.NONE_MIN_DIST_SEC):
                # Vârf fără apăsare: eliberarea tastei sau zgomot.
                X.append(windows[i])
                y.append(C.NONE_LABEL)
                n_none += 1
        if verbose:
            peak_amp = float(np.max(np.abs(audio))) if len(audio) else 0.0
            clip = "  ⚠ CLIPPING: scade gain-ul microfonului" if peak_amp >= 0.99 else ""
            print(f"[{session.name}] decalaj etichete→audio: {offset*1000:+.0f} ms | "
                  f"apăsări potrivite: {len(matches)}/{len(label_times)} | "
                  f"vârfuri {C.NONE_LABEL}: {n_none}{clip}")
    if not X:
        raise RuntimeError("Nu s-au găsit ferestre aliniate cu etichete.")
    X = np.stack(X).astype(np.float32)
    y = np.array(y)
    save_processed(out_dir, X, y)
    return X.shape, len(set(y))
