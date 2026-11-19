"""Demo end-to-end fără microfon: date sintetice → prepare → train → evaluate → predict."""
import subprocess
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.synth import synth_session  # noqa: E402

def main():
    raw = ROOT / "demo_data" / "raw"
    proc = ROOT / "demo_data" / "processed"
    model = ROOT / "demo_data" / "model.joblib"
    sample = ROOT / "examples" / "sample.wav"

    keys = ["a", "s", "d", "f", "g"]
    rng = np.random.default_rng(42)
    for i in range(3):
        path = raw / f"session_{i:02d}"
        presses, t = [], 0.3
        for _ in range(20):
            presses.append((keys[rng.integers(len(keys))], round(t, 3)))
            t += 0.3
        synth_session(path, presses, total_dur=t + 0.3)
    print(f"Generat 3 sesiuni × 20 apăsări în {raw}")

    subprocess.run([sys.executable, "-m", "src.cli", "prepare",
                    "--raw", str(raw), "--out", str(proc)], check=True, cwd=ROOT)
    subprocess.run([sys.executable, "-m", "src.cli", "train",
                    "--data", str(proc), "--model", str(model)], check=True, cwd=ROOT)
    subprocess.run([sys.executable, "-m", "src.cli", "evaluate",
                    "--model", str(model), "--data", str(proc)], check=True, cwd=ROOT)
    subprocess.run([sys.executable, "examples/make_sample.py"], check=True, cwd=ROOT)
    subprocess.run([sys.executable, "-m", "src.cli", "predict",
                    "--model", str(model), "--audio", str(sample)],
                   check=True, cwd=ROOT)

if __name__ == "__main__":
    main()