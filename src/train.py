"""Antrenare SVM + StandardScaler, salvare și încărcare."""
from pathlib import Path
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from . import config as C

def save_processed(path: Path, X: np.ndarray, y: np.ndarray) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "windows.npz", X=X, y=y)

def load_processed(path: Path):
    data = np.load(Path(path) / "windows.npz")
    return data["X"], data["y"]

def split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2):
    return train_test_split(X, y, test_size=test_size,
                            random_state=C.SEED, stratify=y)
    
def filter_classes(X, y, min_samples: int = 15):
    from collections import Counter
    counts = Counter(y)
    keep = {k for k, v in counts.items() if v >= min_samples}
    mask = np.array([label in keep for label in y])
    return X[mask], y[mask], counts

def shift_windows(X: np.ndarray, k: int) -> np.ndarray:
    """Decalează ferestrele cu k eșantioane (k>0: mai târziu), completând cu zero."""
    if k == 0:
        return X
    out = np.zeros_like(X)
    if k > 0:
        out[:, k:] = X[:, :-k]
    else:
        out[:, :k] = X[:, -k:]
    return out

def augment_time_shifts(X: np.ndarray, y: np.ndarray,
                        shift: int = C.AUG_SHIFT_SAMPLES):
    """Triplează setul de antrenare: original + decalat cu ±shift eșantioane.
    Doar pentru antrenare; validarea și predicția folosesc ferestrele ca atare."""
    if shift <= 0 or len(X) == 0:
        return X, y
    return (np.concatenate([shift_windows(X, -shift), X, shift_windows(X, shift)]),
            np.tile(y, 3))

def train_model(X_feats: np.ndarray, y: np.ndarray, feature_kind: str = "summary",
                probability: bool = True):
    """SVM RBF pe features standardizate.

    Fără CalibratedClassifierCV: cu cv=3 antrena 3 modele pe câte 2/3 din date
    și alegea eticheta din probabilitățile mediate, ceea ce pierdea câteva
    puncte de acuratețe. probability=True păstrează predict_proba, iar
    eticheta vine din predict (decizia SVM-ului antrenat pe toate datele).
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10.0, gamma="scale", probability=probability,
                    random_state=C.SEED)),
    ])
    model.fit(X_feats, y)
    model.feature_kind_ = feature_kind   # vezi features.features_for
    return model
