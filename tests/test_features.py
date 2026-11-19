import numpy as np
from src.features import extract_features, extract_features_batch, FEAT_DIM
from src import config as C

def test_feature_shape_and_finite():
    w = np.random.randn(C.WINDOW_SAMPLES).astype(np.float32)
    f = extract_features(w)
    assert f.shape == (FEAT_DIM,)
    assert np.all(np.isfinite(f))

def test_empty_window_returns_zeros():
    f = extract_features(np.zeros(0, dtype=np.float32))
    assert f.shape == (FEAT_DIM,)
    assert np.all(f == 0)

def test_batch_shape():
    ws = np.random.randn(4, C.WINDOW_SAMPLES).astype(np.float32)
    feats = extract_features_batch(ws)
    assert feats.shape == (4, FEAT_DIM)

def test_mel_features_shape_and_short_window():
    from src.features import extract_mel_features, extract_mel_features_batch, MEL_DIM
    w = np.random.randn(C.WINDOW_SAMPLES).astype(np.float32)
    f = extract_mel_features(w)
    assert f.shape == (MEL_DIM,) and np.all(np.isfinite(f))
    # Fereastră mai scurtă / goală: aceeași dimensiune a vectorului.
    assert extract_mel_features(w[:1000]).shape == (MEL_DIM,)
    assert np.all(extract_mel_features(np.zeros(0, dtype=np.float32)) == 0)
    assert extract_mel_features_batch(np.stack([w, w])).shape == (2, MEL_DIM)

def test_features_for_follows_model_kind_and_predict_keys():
    from src.features import (features_for, predict_keys, FEAT_DIM, MEL_DIM,
                              extract_mel_features_batch)
    from src.train import train_model
    rng = np.random.default_rng(0)
    t = np.arange(C.WINDOW_SAMPLES) / C.SR
    X = np.vstack([rng.normal(0, 0.1, (12, C.WINDOW_SAMPLES)) + np.sin(2 * np.pi * f * t)
                   for f in (500, 3000)]).astype(np.float32)
    y = np.array(["a"] * 12 + ["b"] * 12)
    model = train_model(extract_mel_features_batch(X), y, feature_kind="mel")
    assert features_for(model, X).shape == (24, MEL_DIM)
    keys, conf = predict_keys(model, features_for(model, X))
    assert (keys == y).mean() >= 0.9
    assert np.all((conf >= 0) & (conf <= 1))

    class Old:            # model vechi, fără feature_kind_ → rezumat MFCC
        pass
    assert features_for(Old(), X).shape == (24, FEAT_DIM)
