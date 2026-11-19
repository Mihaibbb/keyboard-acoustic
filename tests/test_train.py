import numpy as np
from sklearn.model_selection import train_test_split
from src.train import save_processed, load_processed, train_model
from src.features import extract_features_batch
from src import config as C

def test_train_synthetic_two_classes(tmp_path):
    rng = np.random.default_rng(0)
    n = 20
    sr = C.SR
    t = np.arange(4410) / sr
    Xa = (rng.normal(0, 0.1, (n, 4410)) +
          0.5 * np.sin(2 * np.pi * 500 * t)).astype(np.float32)
    Xb = (rng.normal(0, 0.1, (n, 4410)) +
          0.5 * np.sin(2 * np.pi * 2000 * t)).astype(np.float32)
    X = np.vstack([Xa, Xb])
    y = np.array(["a"] * n + ["b"] * n)

    path = tmp_path / "proc"
    save_processed(path, X, y)
    X2, y2 = load_processed(path)
    assert np.array_equal(X, X2) and np.array_equal(y, y2)

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model = train_model(extract_features_batch(Xtr), ytr)
    assert model.score(extract_features_batch(Xte), yte) >= 0.9

def test_shift_windows_and_augmentation():
    from src.train import shift_windows, augment_time_shifts
    X = np.arange(12, dtype=np.float32).reshape(2, 6)
    assert np.array_equal(shift_windows(X, 2)[0], [0, 0, 0, 1, 2, 3])
    assert np.array_equal(shift_windows(X, -2)[0], [2, 3, 4, 5, 0, 0])
    assert shift_windows(X, 0) is X
    y = np.array(["a", "b"])
    Xa, ya = augment_time_shifts(X, y, shift=1)
    assert Xa.shape == (6, 6) and list(ya) == ["a", "b"] * 3
    assert np.array_equal(Xa[2:4], X)          # originalul rămâne în set
    Xn, yn = augment_time_shifts(X, y, shift=0)
    assert Xn is X and yn is y
