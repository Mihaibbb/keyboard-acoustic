"""Generează examples/sample.wav pentru testarea comenzii predict."""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.synth import synth_click, DEFAULT_FREQ  # noqa: E402

SR = 44100

def main():
    out = Path(__file__).parent / "sample.wav"
    presses = [("a", 0.3), ("s", 0.7), ("d", 1.1), ("f", 1.5), ("g", 1.9)]
    total = 2.5
    audio = np.zeros(int(SR * total), dtype=np.float32)
    for key, t_sec in presses:
        click = synth_click(DEFAULT_FREQ[key])
        start = int(t_sec * SR) - len(click) // 2
        audio[start:start + len(click)] += click
    audio += 0.001 * np.random.randn(len(audio)).astype(np.float32)
    sf.write(out, audio, SR)
    print(f"Generated {out}")

if __name__ == "__main__":
    main()
