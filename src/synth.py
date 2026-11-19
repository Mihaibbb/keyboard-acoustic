"""Sintetizator de click-uri pentru demo și teste."""
import csv
from pathlib import Path
import numpy as np
import soundfile as sf

DEFAULT_FREQ = {"a": 500, "s": 1000, "d": 1500, "f": 2000, "g": 2500}

def synth_click(freq: float = 1000.0, dur_ms: float = 80.0,
                amp: float = 0.5, sr: int = 44100) -> np.ndarray:
    n = int(sr * dur_ms / 1000)
    t = np.arange(n) / sr
    return (amp * np.hanning(n) * np.sin(2 * np.pi * freq * t)).astype(np.float32)

def synth_session(path: Path, keys_with_times, freq_map=None,
                  sr: int = 44100, total_dur: float = 3.0,
                  noise: float = 0.001) -> None:
    freq_map = freq_map or DEFAULT_FREQ
    audio = np.zeros(int(sr * total_dur), dtype=np.float32)
    for key, t_sec in keys_with_times:
        click = synth_click(freq_map.get(key, 1000.0), sr=sr)
        start = int(t_sec * sr) - len(click) // 2
        if start < 0:
            start = 0
        end = start + len(click)
        if end > len(audio):
            audio = np.pad(audio, (0, end - len(audio)))
        audio[start:end] += click
    if noise > 0:
        audio += noise * np.random.randn(len(audio)).astype(np.float32)
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    sf.write(path / "audio.wav", audio, sr)
    with open(path / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_sec", "key"])
        for key, t_sec in keys_with_times:
            w.writerow([f"{t_sec:.4f}", key])