import csv
import numpy as np
from src import config as C
from src.prepare import (estimate_offset, match_one_to_one, normalize_key,
                         prepare_sessions)
from src.synth import synth_session
from src.train import load_processed

def test_normalize_key_maps_shifted_chars_to_physical_key():
    assert normalize_key("!") == "1"
    assert normalize_key("#") == "3"
    assert normalize_key("<") == ","
    assert normalize_key("A") == "a"
    assert normalize_key("backspace") == "backspace"

def test_estimate_offset_recovers_late_labels():
    rng = np.random.default_rng(1)
    peaks = np.cumsum(rng.uniform(0.06, 0.2, 200))
    labels = peaks + 0.085  # etichetele întârzie 85ms față de audio
    off = estimate_offset(peaks, labels)
    assert abs(off - (-0.085)) <= C.OFFSET_STEP_SEC

def test_estimate_offset_prefers_zero_on_periodic_data():
    peaks = np.arange(0.3, 6.0, 0.3)
    assert abs(estimate_offset(peaks, peaks.copy())) < 1e-6

def test_match_one_to_one_never_reuses_label():
    peaks = np.array([1.000, 1.010, 1.200])
    labels = np.array([1.005, 1.195])
    m = match_one_to_one(peaks, labels)
    assert sorted(m.values()) == [0, 1]
    assert m[2] == 1

def test_prepare_corrects_label_lag_and_adds_none(tmp_path):
    # Apăsări rapide (90ms) + etichete întârziate 80ms: potrivirea veche
    # „cea mai apropiată etichetă în 100ms” ar da tasta vecină.
    raw = tmp_path / "raw"
    keys = "asdfg" * 8
    presses = [(k, round(0.3 + i * 0.09, 3)) for i, k in enumerate(keys)]
    # un click „release” fără etichetă, departe de orice apăsare
    synth_session(raw / "s0", presses + [("d", 4.5)], total_dur=5.0)
    with open(raw / "s0" / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_sec", "key"])
        for k, t in presses:
            w.writerow([f"{t + 0.08:.4f}", k])
    prepare_sessions(raw, tmp_path / "proc", verbose=False)
    _, y = load_processed(tmp_path / "proc")
    assert list(y) == list(keys)   # implicit fără NONE_LABEL

    prepare_sessions(raw, tmp_path / "proc2", include_none=True, verbose=False)
    _, y = load_processed(tmp_path / "proc2")
    assert [lab for lab in y if lab != C.NONE_LABEL] == list(keys)
    assert list(y).count(C.NONE_LABEL) == 1
