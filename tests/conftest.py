import numpy as np
import pytest
from pathlib import Path
from src.synth import synth_session

@pytest.fixture
def synthetic_session(tmp_path):
    path = tmp_path / "session_01"
    keys = [("a", 0.3), ("s", 0.6), ("d", 0.9), ("f", 1.2), ("g", 1.5)]
    synth_session(path, keys, total_dur=2.0)
    return path, keys

@pytest.fixture
def synthetic_dataset(tmp_path):
    raw = tmp_path / "raw"
    rng = np.random.default_rng(42)
    keys = ["a", "s", "d"]
    for sess_i in range(3):
        path = raw / f"session_{sess_i:02d}"
        presses = []
        t = 0.3
        for _ in range(20):
            presses.append((keys[rng.integers(len(keys))], round(t, 3)))
            t += 0.3
        synth_session(path, presses, total_dur=t + 0.3)
    return raw
