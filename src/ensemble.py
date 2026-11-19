"""Soft-voting ensemble classifier: SVM(rbf, proba) + MLP + HistGradientBoosting
on the mel320 log-mel features (same `extract_mel_features` the SVM baseline uses).

Drop-in for the single SVC: exposes `classes_`, `predict`, `predict_proba` and
`feature_kind_ = "mel"`, so it works unchanged with `features.features_for`,
`features.predict_keys`, `compare.py`, `evaluate.py`, `predict.py`.

Model file is a plain dict {"svm","mlp","hgb","classes"} (no custom-class pickle),
so loading only needs this module on the path.
"""
from pathlib import Path
import joblib
import numpy as np
from . import config as C

DEFAULT_PATH = C.MODELS / "ensemble_trio.joblib"


class EnsembleClassifier:
    """Equal-weight soft-vote of svm+mlp+hgb over aligned class-probability columns."""
    feature_kind_ = "mel"

    def __init__(self, path=DEFAULT_PATH, weights=(1.0, 1.0, 1.0)):
        d = joblib.load(Path(path))
        self.members = [("svm", d["svm"], weights[0]),
                        ("mlp", d["mlp"], weights[1]),
                        ("hgb", d["hgb"], weights[2])]
        self.classes_ = np.asarray(d["classes"]).astype(str)
        self._idx = {nm: [list(np.asarray(m.classes_).astype(str)).index(c)
                          for c in self.classes_] for nm, m, _ in self.members}

    def predict_proba(self, feats):
        P = sum(w * m.predict_proba(feats)[:, self._idx[nm]]
                for nm, m, w in self.members)
        return P / P.sum(1, keepdims=True)

    def predict(self, feats):
        return self.classes_[self.predict_proba(feats).argmax(1)].astype(str)


def train_ensemble(X_feats: np.ndarray, y: np.ndarray, feature_kind: str = "mel"):
    """Fit svm(proba)+mlp+hgb on standardized mel features; return a dict to joblib.dump.
    Hyperparameters match the tuned configuration from the classifier sweep."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier

    svm = Pipeline([("s", StandardScaler()),
                    ("m", SVC(kernel="rbf", C=10.0, gamma="scale",
                              probability=True, random_state=C.SEED))])
    mlp = Pipeline([("s", StandardScaler()),
                    ("m", MLPClassifier((256, 128), alpha=1e-3, max_iter=200,
                                        batch_size=256, early_stopping=True,
                                        n_iter_no_change=10, random_state=C.SEED))])
    hgb = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.1,
                                         l2_regularization=1.0, random_state=C.SEED)
    svm.fit(X_feats, y)
    mlp.fit(X_feats, y)
    hgb.fit(X_feats, y)
    classes = np.asarray(sorted(set(y))).astype(str)
    return {"svm": svm, "mlp": mlp, "hgb": hgb, "classes": classes,
            "feature_kind": feature_kind}


def load_ensemble(path=DEFAULT_PATH) -> EnsembleClassifier:
    return EnsembleClassifier(path=path)


# Membrele cheie ale dict-ului salvat de train_ensemble; servesc la detecția
# fișierelor de ensemble vs modelele sklearn obișnuite (un singur SVC etc.).
_ENSEMBLE_KEYS = {"svm", "mlp", "hgb", "classes"}


def load_any(path):
    """Încarcă un model: dacă fișierul e dict-ul ensemble-ului, îl împachetează
    în EnsembleClassifier; altfel întoarce obiectul joblib ca atare (SVC vechi)."""
    obj = joblib.load(Path(path))
    if isinstance(obj, dict) and _ENSEMBLE_KEYS <= set(obj):
        return EnsembleClassifier(path=path)
    return obj
