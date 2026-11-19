import csv
import shutil
from pathlib import Path
import joblib
from src.prepare import prepare_sessions
from src.train import load_processed, train_model, split
from src.features import extract_features_batch
from src.predict import predict_audio

def test_full_pipeline(synthetic_dataset, tmp_path):
    sessions = sorted(synthetic_dataset.iterdir())
    train_raw = tmp_path / "train_raw"
    train_raw.mkdir()
    for s in sessions[1:]:
        shutil.copytree(s, train_raw / s.name)
    test_session = sessions[0]

    proc = tmp_path / "proc"
    prepare_sessions(train_raw, proc)
    X, y = load_processed(proc)
    assert X.shape[0] > 10 and len(set(y)) >= 2

    Xtr, Xte, ytr, yte = split(X, y)
    model = train_model(extract_features_batch(Xtr), ytr)
    acc = model.score(extract_features_batch(Xte), yte)
    assert acc >= 0.7

    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    loaded = joblib.load(model_path)

    results = predict_audio(loaded, test_session / "audio.wav")
    assert 15 <= len(results) <= 25
    assert all(r["key"] in {"a", "s", "d"} for r in results)
    assert all(0.0 <= r["confidence"] <= 1.0 for r in results)

    truth = {}
    with open(test_session / "labels.csv") as f:
        for row in csv.DictReader(f):
            truth[round(float(row["timestamp_sec"]), 2)] = row["key"]

    matched = 0
    for r in results:
        nearest_t = min(truth, key=lambda t: abs(t - r["time_sec"]))
        if abs(nearest_t - r["time_sec"]) < 0.05 and truth[nearest_t] == r["key"]:
            matched += 1
    assert matched / len(results) >= 0.7
