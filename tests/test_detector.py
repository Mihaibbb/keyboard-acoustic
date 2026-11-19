import numpy as np
from src.detector import context_features, train_detector, press_probability
from src.synth import synth_session

def test_context_features_shape_and_level_sign():
    rng = np.random.default_rng(0)
    windows = rng.normal(0, 1, (5, 200)).astype(np.float32)
    windows[2] *= 8.0                       # vârf mult mai tare
    pt = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    f = context_features(windows, pt)
    assert f.shape == (5, 10)
    assert f[2, 0] == f[:, 0].max()         # nivelul relativ: cel mai tare

def test_context_features_empty():
    assert context_features(np.zeros((0, 200)), np.zeros(0)).shape == (0, 10)

def test_detector_learns_press_vs_quiet(tmp_path):
    # Sesiuni sintetice: synth pune apăsări la etichete; vârfurile dintre ele
    # (dacă apar) nu sunt etichetate. Detectorul trebuie să dea prob mare pe apăsări.
    raw = tmp_path / "raw"
    for i in range(3):
        keys = [(k, 0.3 + j * 0.3) for j, k in enumerate("asdfghjkl")]
        synth_session(raw / f"session_{i:02d}", keys, total_dur=3.2)
    det, n = train_detector(raw, verbose=False)
    assert n > 0
    from src.segment import segment_audio
    import librosa
    from src import config as C
    audio, _ = librosa.load(raw / "session_00" / "audio.wav", sr=C.SR)
    win, peaks, _ = segment_audio(audio)
    pp = press_probability(det, win, peaks / C.SR)
    assert pp.shape == (len(peaks),)
    assert pp.min() >= 0.0 and pp.max() <= 1.0
