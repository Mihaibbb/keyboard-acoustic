import numpy as np
import soundfile as sf
from src import config as C
from src.segment import (bandpass, segment_audio, rms_envelope, detect_peaks,
                         extract_windows)
from src.synth import synth_session, synth_click

def test_bandpass_shapes():
    audio = np.random.randn(44100).astype(np.float32)
    assert bandpass(audio).shape == audio.shape

def test_rms_envelope_length():
    env = rms_envelope(np.random.randn(44100).astype(np.float32))
    assert env.ndim == 1 and len(env) > 10

def test_detect_peaks_empty():
    assert len(detect_peaks(np.array([]))) == 0

def test_segment_detects_three_peaks(tmp_path):
    path = tmp_path / "s"
    synth_session(path, [("a", 0.2), ("s", 0.5), ("d", 0.8)], total_dur=1.2)
    audio, sr = sf.read(path / "audio.wav", dtype="float32")
    windows, peaks, _ = segment_audio(audio, sr=sr)
    assert windows.shape[0] == len(peaks) == 3
    times = peaks / sr
    for expected in [0.2, 0.5, 0.8]:
        assert np.min(np.abs(times - expected)) < 0.03
    assert windows.shape[1] == C.WINDOW_SAMPLES

def test_fast_typing_keeps_every_press(tmp_path):
    # Tastare rapidă: 70ms între taste (sub vechiul PEAK_MIN_DIST de 150ms).
    path = tmp_path / "fast"
    presses = [(k, round(0.2 + i * 0.07, 3))
               for i, k in enumerate("asdfgasdfg")]
    synth_session(path, presses, total_dur=1.2)
    audio, sr = sf.read(path / "audio.wav", dtype="float32")
    _, peaks, _ = segment_audio(audio, sr=sr)
    assert len(peaks) == len(presses)

def test_loud_click_does_not_hide_quiet_ones():
    # Un click clipped nu trebuie să ridice pragul peste tastele normale.
    sr = C.SR
    audio = 0.001 * np.random.default_rng(0).standard_normal(sr * 2)
    audio = audio.astype(np.float32)
    for t, amp in [(0.3, 1.0), (0.8, 0.05), (1.3, 0.05), (1.7, 0.05)]:
        click = synth_click(1000.0, amp=amp, sr=sr)
        s = int(t * sr) - len(click) // 2
        audio[s:s + len(click)] += click
    _, peaks, _ = segment_audio(np.clip(audio, -1, 1), sr=sr)
    assert len(peaks) == 4

def test_window_starts_before_peak():
    audio = np.zeros(C.SR, dtype=np.float32)
    audio[C.SR // 2] = 1.0
    w = extract_windows(audio, np.array([C.SR // 2]))
    assert w.shape == (1, C.WINDOW_SAMPLES)
    assert np.argmax(w[0]) == C.PRE_PEAK_SAMPLES
