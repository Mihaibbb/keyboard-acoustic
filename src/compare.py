"""Textul real (keylogger) vs textul reconstruit din audio de model.

Două reconstrucții:
- „cu momente cunoscute”: pentru fiecare apăsare din labels.csv se ia
  vârful audio potrivit și predicția lui (· = apăsare fără vârf detectat).
  Arată cât de bine recunoaște modelul tastele când știe unde sunt.
- „orb”: toate vârfurile detectate în audio, fără etichete — ce ar obține
  cineva care are doar înregistrarea (include eliberări de taste, zgomot, voce).
"""
import csv
from pathlib import Path
import librosa
import numpy as np
from . import config as C
from .decode import beam_decode
from .features import features_for, predict_keys
from .prepare import estimate_offset, match_one_to_one, normalize_key
from .segment import segment_audio
from .text import final_chars, read_events

MISSED = "·"

def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def _text(keys):
    """Ca text.final_chars, dar acceptă și MISSED; fără timpi."""
    chars, _ = final_chars([(0.0, k) for k in keys])
    return "".join(c for _, c in chars)

def _segments(keys):
    """Împarte indicii apăsărilor în linii (la enter). Enter-ul rămâne în linia pe care o închide."""
    segs, cur = [], []
    for i, k in enumerate(keys):
        cur.append(i)
        if k == "enter":
            segs.append(cur)
            cur = []
    if cur:
        segs.append(cur)
    return segs

_SYMBOLS = {"space": "_", "backspace": "<", "enter": "|", "shift": "^",
            "shift_r": "^", MISSED: MISSED}

def _symbol(key: str) -> str:
    """Un singur caracter per tastă, ca rândurile să rămână aliniate."""
    if key in _SYMBOLS:
        return _SYMBOLS[key]
    return key if len(key) == 1 else "?"

def _wrap(text: str, width: int):
    return [text[i: i + width] for i in range(0, len(text), width)] or [""]

def _write_readable(path: Path, name: str, rows, real, oracle, key_acc,
                    acc_known, acc_blind, has_detector, width: int = 90):
    """Fișier text: textul din keylogger și textul dedus de model, unul sub altul."""
    out = [f"Sesiune: {name}",
           f"Taste ghicite corect (pe apăsările găsite în audio): {key_acc:.1%}",
           f"Litere corecte, model cu momentele apăsărilor cunoscute: {acc_known:.1%}",
           f"Litere corecte, model doar din audio "
           f"({'cu' if has_detector else 'fără'} detector): {acc_blind:.1%}",
           "", "=" * width, "1. TEXTE (după aplicarea backspace-ului)", "=" * width]
    for r in rows:
        out += ["", f"--- linia {r[0]} ({r[1]}–{r[2]} s)", "",
                "KEYLOGGER (ce ai tastat):", *_wrap(r[5], width), "",
                f"MODEL, doar din audio ({float(r[4]):.1%} litere corecte):",
                *_wrap(r[7], width), "",
                f"MODEL, cu momentele apăsărilor cunoscute ({float(r[3]):.1%}):",
                *_wrap(r[6], width)]
    out += ["", "=" * width, "2. TASTĂ CU TASTĂ (fiecare apăsare, în ordine)",
            "   _ spațiu   < backspace   | enter   ^ shift   "
            f"{MISSED} apăsare negăsită în audio   x = greșit", "=" * width]
    rs = "".join(_symbol(k) for k in real)
    ps = "".join(_symbol(k) for k in oracle)
    marks = "".join(" " if a == b else "x" for a, b in zip(rs, ps))
    for i in range(0, len(rs), width):
        out += ["", f"real : {rs[i: i + width]}", f"model: {ps[i: i + width]}",
                f"       {marks[i: i + width]}".rstrip()]
    path.write_text("\n".join(out) + "\n")

def compare_session(model, session: Path, out_dir: Path, sr: int = C.SR,
                    detector=None, threshold: float = 0.6, lm=None, lam: float = 0.6):
    session, out_dir = Path(session), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    audio, _ = librosa.load(session / "audio.wav", sr=sr, mono=True)
    windows, peaks, _ = segment_audio(audio, sr=sr)
    peak_t = peaks / sr
    feats = features_for(model, windows, sr=sr)
    pred, conf = predict_keys(model, feats)
    # Cu model de limbaj (lm): calea oarbă recombină tastele pe linie prin beam
    # search peste probabilitățile acustice (vezi src/decode.py).
    proba = model.predict_proba(feats) if lm is not None else None
    classes = [str(c) for c in model.classes_]
    if detector is not None:
        from .detector import press_probability
        is_press = press_probability(detector, windows, peak_t, sr=sr) >= threshold
    else:
        is_press = np.ones(len(peak_t), dtype=bool)

    events = read_events(session / "labels.csv")
    label_t = np.array([t for t, _ in events])
    real = [normalize_key(k) for _, k in events]
    shifted = label_t + estimate_offset(peak_t, label_t)
    by_label = {l: p for p, l in match_one_to_one(peak_t, shifted).items()}

    known = set(model.classes_)
    with open(out_dir / "keys_vs_pred.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_sec", "real_key", "pred_key", "confidence", "status"])
        for i, (t, k) in enumerate(zip(label_t, real)):
            p = by_label.get(i)
            if p is None:
                status, pk, c = "nedetectat", "", ""
            else:
                pk, c = pred[p], f"{conf[p]:.2f}"
                status = ("corect" if pk == k else
                          "greșit" if k in known else "clasă_necunoscută")
            w.writerow([f"{t:.4f}", k, pk, c, status])

    oracle = [pred[by_label[i]] if i in by_label else MISSED for i in range(len(real))]
    rows, totals = [], np.zeros(4)
    for n, seg in enumerate(_segments(real), 1):
        real_txt = _text([real[i] for i in seg]).rstrip("\n")
        if not real_txt.strip():
            continue
        oracle_txt = _text([oracle[i] for i in seg]).rstrip("\n")
        lo, hi = shifted[seg[0]] - 0.05, shifted[seg[-1]] + 0.05
        bidx = [j for j in range(len(peak_t))
                if lo <= peak_t[j] <= hi and pred[j] != C.NONE_LABEL and is_press[j]]
        if lm is not None and bidx:
            blind_keys = beam_decode(proba[bidx], classes, lm, lam=lam,
                                     force_labels=[pred[j] for j in bidx])
        else:
            blind_keys = [pred[j] for j in bidx]
        blind_txt = _text(blind_keys).rstrip("\n")
        e_o, e_b = edit_distance(real_txt, oracle_txt), edit_distance(real_txt, blind_txt)
        totals += [len(real_txt), e_o, e_b, 0]
        rows.append([n, f"{label_t[seg[0]]:.2f}", f"{label_t[seg[-1]]:.2f}",
                     f"{1 - e_o / len(real_txt):.3f}", f"{1 - e_b / len(real_txt):.3f}",
                     real_txt, oracle_txt, blind_txt])
    with open(out_dir / "predicted_text.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["line", "start_sec", "end_sec", "char_acc_known_times",
                    "char_acc_blind", "real_text", "pred_known_times", "pred_blind"])
        w.writerows(rows)
    n_chars, e_o, e_b, _ = totals
    key_acc = float(np.mean([pred[p] == real[l] for l, p in by_label.items()
                             if real[l] in known]))
    _write_readable(out_dir / "text_vs_model.txt", session.name, rows, real, oracle,
                    key_acc, 1 - e_o / n_chars, 1 - e_b / n_chars,
                    detector is not None)
    return {
        "labels": len(real), "peaks": len(peak_t), "detected": len(by_label),
        "key_acc": key_acc,
        "char_acc_known_times": 1 - e_o / n_chars,
        "char_acc_blind": 1 - e_b / n_chars,
        "rows": rows,
    }
