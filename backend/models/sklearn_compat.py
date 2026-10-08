"""
Lightweight NumPy-based compatibility layer for scikit-learn metrics, model selection, and encoders.
Replaces heavy scikit-learn/scipy dependencies to optimize deployment size.
"""
import numpy as np

def mean_absolute_error(y_true, y_pred):
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))

def root_mean_squared_error(y_true, y_pred):
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))

def r2_score(y_true, y_pred):
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    return float(1.0 - (ss_res / (ss_tot + 1e-10)))

def f1_score(y_true, y_pred, average="macro"):
    yt = np.array(y_true, dtype=int)
    yp = np.array(y_pred, dtype=int)
    tp = np.sum((yt == 1) & (yp == 1))
    fp = np.sum((yt == 0) & (yp == 1))
    fn = np.sum((yt == 1) & (yp == 0))
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    return float(2 * (precision * recall) / (precision + recall + 1e-10))

class KFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def split(self, X):
        n = len(X)
        indices = np.arange(n)
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            rng.shuffle(indices)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            val_idx = indices[start:stop]
            train_idx = np.concatenate([indices[:start], indices[stop:]])
            yield train_idx, val_idx
            current = stop

class LabelEncoder:
    def __init__(self):
        self.classes_ = np.array([])
        self._mapping = {}
        self._inverse = {}

    def fit(self, y):
        self.classes_ = np.unique(y)
        self._mapping = {val: idx for idx, val in enumerate(self.classes_)}
        self._inverse = {idx: val for idx, val in enumerate(self.classes_)}
        return self

    def transform(self, y):
        return np.array([self._mapping.get(val, 0) for val in y])

    def fit_transform(self, y):
        self.fit(y)
        return self.transform(y)

    def inverse_transform(self, y):
        return np.array([self._inverse.get(idx, self.classes_[0] if len(self.classes_) > 0 else None) for idx in y])
