"""Înregistrare audio sincronizată cu apăsări de taste.

Timestamp-urile tastelor sunt măsurate pe ceasul stream-ului audio
(PortAudio), relativ la momentul primului eșantion înregistrat. Varianta
veche folosea time.time() pornit înainte de deschiderea microfonului, iar
latența de pornire (~60-100ms pe macOS) decala toate etichetele.
"""
import csv
import threading
from pathlib import Path
import numpy as np
import sounddevice as sd
import soundfile as sf
from pynput import keyboard
from . import config as C
from .text import write_final_text

def _key_name(key) -> str:
    char = getattr(key, "char", None)
    if char:
        return char
    return str(key).replace("Key.", "")

def _key_id(key):
    """Identificator de tastă fizică, stabil între press și release."""
    vk = getattr(key, "vk", None)
    if vk is None and hasattr(key, "value"):
        vk = getattr(key.value, "vk", None)
    return vk if vk is not None else str(key)

class KeyTracker:
    """Apăsări (keydown) și eliberări (keyup) per tastă fizică.

    - O singură apăsare per tastă ținută: autorepeat-ul OS nu face sunet nou.
    - Eliberarea primește numele de la apăsare, nu pe cel de la eliberare: cu
      Shift lăsat primul, „A” apăsat se eliberează ca „a”, dar e aceeași tastă
      și același eveniment de apăsare, deci rândul de keyup trebuie să-l numească
      la fel.
    - Eliberarea unei taste apăsate înainte de pornirea înregistrării se ignoră.
    """

    def __init__(self):
        self.presses = []    # (t_down, nume)
        self.releases = []   # (t_up, nume, t_down)
        self._down = {}      # id tastă → (t_down, nume)

    def press(self, kid, name: str, t: float) -> bool:
        if kid in self._down:
            return False
        self._down[kid] = (t, name)
        self.presses.append((t, name))
        return True

    def release(self, kid, t: float) -> bool:
        info = self._down.pop(kid, None)
        if info is None:
            return False
        t_down, name = info
        self.releases.append((t, name, t_down))
        return True

def write_keyup_csv(path: Path, releases):
    """keyup.csv: timestamp_sec (eliberare), key, press_sec (apăsarea ei)."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_sec", "key", "press_sec"])
        for t_up, k, t_down in sorted(releases):
            w.writerow([f"{t_up:.4f}", k, f"{t_down:.4f}"])

def record_session(out_dir: Path, duration: float, sr: int = C.SR):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    keys = KeyTracker()
    chunks = []
    t0 = [None]       # timpul (ceas stream) al primului eșantion înregistrat
    overflows = [0]
    stop = threading.Event()

    def audio_cb(indata, frames, time_info, status):
        if status.input_overflow:
            overflows[0] += 1
        if t0[0] is None:
            adc = time_info.inputBufferAdcTime
            if adc > 0:
                t0[0] = adc
            else:  # backend fără timestamp ADC: aproximare
                t0[0] = stream.time - frames / sr - stream.latency
        chunks.append(indata[:, 0].copy())

    def on_press(key):
        if key == keyboard.Key.esc:   # Esc oprește înregistrarea (nu e etichetat)
            stop.set()
            return
        if t0[0] is None:
            return
        keys.press(_key_id(key), _key_name(key), stream.time - t0[0])

    def on_release(key):
        if t0[0] is None:
            return
        keys.release(_key_id(key), stream.time - t0[0])

    stream = sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                            callback=audio_cb)
    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()
    stream.start()
    try:
        stop.wait(timeout=duration)   # durata e maximul; Esc oprește mai devreme
    finally:
        stream.stop()
        stream.close()
        listener.stop()

    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    audio = audio[: int(duration * sr)]
    sf.write(out_dir / "audio.wav", audio, sr)
    with open(out_dir / "labels.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp_sec", "key"])
        for t, k in keys.presses:
            w.writerow([f"{t:.4f}", k])
    # Separat de labels.csv, ca prepare/text să rămână neschimbate.
    write_keyup_csv(out_dir / "keyup.csv", keys.releases)
    write_final_text(out_dir / "labels.csv", out_dir / "final_text.csv")
    n_shift = sum(1 for _, k in keys.presses if k.startswith("shift"))
    print(f"Eliberări înregistrate: {len(keys.releases)} | apăsări Shift: {n_shift}")
    if overflows[0]:
        print(f"Atenție: {overflows[0]} overflow-uri audio (eșantioane pierdute).")
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    if peak >= 0.99:
        print("Atenție: semnal clipped (vârf 1.0). Scade gain-ul microfonului.")
    return len(keys.presses), len(audio) / sr
