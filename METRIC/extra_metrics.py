"""Additional frozen-feature transferability scores; no model training or I/O.

PARC/SFDA use PCA(32) fit only on Probe-Train. SFDA follows the original
confidence-mix and shrinkage FDA scoring, applied in that PCA space.
GBC uses per-class Ledoit-Wolf covariances in the same PCA space.
"""
import itertools
import warnings
import numpy as np
from scipy.linalg import solve
from scipy.special import softmax
from scipy.stats import spearmanr
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler

EXTRA = ('parc_pca32', 'transrate', 'gbc_pca32', 'sfda_pca32', 'knn_probe', 'label_cka')


def label_cka(x, y, task):
    x = np.asarray(x, float); x = x - x.mean(axis=0)
    if task == 'regression':
        y = np.asarray(y, float).reshape(-1, 1)
    elif task == 'classification':
        _, index = np.unique(y, return_inverse=True)
        y = np.eye(index.max() + 1)[index]
    else:
        y = np.asarray(y, float)
    y = y - y.mean(axis=0)
    if len(x) > x.shape[1]:
        # Linear CKA in feature space avoids allocating a samples-by-samples matrix.
        numerator = np.square(x.T @ y).sum()
        denom = np.linalg.norm(x.T @ x) * np.linalg.norm(y.T @ y)
    else:
        gram_x = x @ x.T; gram_y = y @ y.T
        numerator = np.sum(gram_x * gram_y)
        denom = np.linalg.norm(gram_x) * np.linalg.norm(gram_y)
    return float(numerator / denom) if denom > 1e-20 else np.nan


def parc(x, y):
    _, labels = np.unique(y, return_inverse=True)
    y_onehot = np.eye(labels.max() + 1)[labels]
    x = StandardScaler().fit_transform(x)
    with np.errstate(invalid='ignore', divide='ignore'):
        a, b = 1 - np.corrcoef(x), 1 - np.corrcoef(y_onehot)
    ix = np.triu_indices(len(x), 1)
    if not np.isfinite(a[ix]).all() or not np.isfinite(b[ix]).all():
        return np.nan
    return float(spearmanr(a[ix], b[ix]).statistic * 100)


def transrate(x, y, eps=1e-4):
    def rate(z):
        gram = (z @ z.T if len(z) <= z.shape[1] else z.T @ z) / (len(z) * eps)
        sign, value = np.linalg.slogdet(np.eye(len(gram)) + gram)
        if sign <= 0:
            raise ValueError('non-positive coding-rate determinant')
        return 0.5 * value
    return float(rate(x) - sum(len(z) / len(x) * rate(z)
                               for c in np.unique(y) if len(z := x[y == c])))


def gbc(x, y):
    classes = np.unique(y)
    if len(classes) < 2 or min(np.sum(y == c) for c in classes) < 2:
        return np.nan
    models = [(x[y == c].mean(0), LedoitWolf().fit(x[y == c]).covariance_) for c in classes]
    scale = max(float(np.var(x)), 1e-12)
    models = [(mu, cov + np.eye(x.shape[1]) * scale * 1e-6) for mu, cov in models]
    overlap = 0.
    for (m1, c1), (m2, c2) in itertools.combinations(models, 2):
        pooled = (c1 + c2) / 2
        diff = m1 - m2
        d = 0.125 * diff @ solve(pooled, diff, assume_a='pos')
        d += 0.5 * (np.linalg.slogdet(pooled)[1] - 0.5 * (np.linalg.slogdet(c1)[1] + np.linalg.slogdet(c2)[1]))
        overlap += np.exp(-max(float(d), 0.))
    return float(-2 * overlap)


def sfda(x, y):
    _, y = np.unique(y, return_inverse=True)
    counts = np.bincount(y); classes = len(counts)
    if classes < 2:
        return np.nan
    priors = counts / len(y)
    def fit(z, shrink=None):
        means = np.stack([z[y == c].mean(0) for c in range(classes)])
        within = np.zeros((z.shape[1], z.shape[1]))
        for c in range(classes):
            v = z[y == c] - means[c]
            within += priors[c] * (v.T @ v / len(v))
        if shrink is None:
            v = within.sum(axis=1)
            for _ in range(3):
                w = within @ v; w /= max(np.linalg.norm(w), 1e-20)
                v = within @ w; v /= max(np.linalg.norm(v), 1e-20)
            largest = float(v @ within @ w)
            shrink = max(np.exp(-5 * largest), 1e-10)
        mu = np.trace(within) / within.shape[0]
        regular = (1 - shrink) * within + (shrink * mu + 1e-12) * np.eye(within.shape[0])
        coefs = solve(regular, means.T, assume_a='pos').T
        intercept = -0.5 * np.sum(means * coefs, axis=1) + np.log(priors)
        return means, shrink, softmax(z @ coefs.T + intercept, axis=1)
    means, shrink, probabilities = fit(x)
    # Original SFDA applies a second softmax to the first-stage probabilities.
    confidence = softmax(probabilities, axis=1)[np.arange(len(y)), y, None]
    other_means = (means.sum(0) - means[y]) / (classes - 1)
    mixed = confidence * x + (1 - confidence) * other_means
    _, _, probabilities = fit(mixed, shrink)
    return float(probabilities[np.arange(len(y)), y].mean())


def knn(train_x, train_y, val_x, val_y, task):
    scaler = StandardScaler().fit(train_x)
    train = scaler.transform(train_x); val = scaler.transform(val_x)
    k = min(5, len(train))
    if task == 'regression':
        model = KNeighborsRegressor(n_neighbors=k, metric='cosine').fit(train, train_y)
        pred = model.predict(val); mae = float(mean_absolute_error(val_y, pred))
        return {'knn_probe': -mae, 'knn_mae': mae,
                'knn_rmse': float(np.sqrt(mean_squared_error(val_y, pred))),
                'knn_r2': float(r2_score(val_y, pred)) if len(val_y)>1 else np.nan}
    model = KNeighborsClassifier(n_neighbors=k, metric='cosine').fit(train, train_y)
    pred = model.predict(val); prob = model.predict_proba(val)
    auc = [roc_auc_score(np.asarray(val_y) == c, prob[:, i]) for i, c in enumerate(model.classes_)
           if len(np.unique(np.asarray(val_y) == c)) == 2]
    f1 = float(f1_score(val_y, pred, average='macro', zero_division=0))
    return {'knn_probe': f1, 'knn_f1_macro': f1, 'knn_auroc': float(np.mean(auc)) if auc else np.nan}


def evaluate_extra(train_x, train_y, val_x, val_y, task):
    x = np.asarray(train_x, dtype=np.float64); y = np.asarray(train_y)
    result = {name: np.nan for name in EXTRA}
    if task not in ('classification', 'regression'):
        return result
    # No transformation is fitted on Probe-Val.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        result['label_cka'] = label_cka(x, y, task)
        result.update(knn(x, y, val_x, val_y, task))
        if task == 'classification' and len(np.unique(y)) >= 2:
            n = min(32, len(x) - 1, x.shape[1])
            reduced = PCA(n_components=n, svd_solver='full').fit_transform(x)
            result['parc_pca32'] = parc(reduced, y)
            result['transrate'] = transrate(x, y)
            result['gbc_pca32'] = gbc(reduced, y)
            result['sfda_pca32'] = sfda(reduced, y)
    return result
