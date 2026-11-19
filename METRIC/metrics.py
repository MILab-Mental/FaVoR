"""Small, deterministic representation and transferability calculations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.covariance import ledoit_wolf_shrinkage
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeClassifier
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler


METRICS = ('rankme', 'feature_variance', 'top1_singular_energy', 'logme', 'hscore',
           'ridge_probe', 'linear_probe')
DIRECTION = {name: (-1 if name == 'top1_singular_energy' else 1) for name in METRICS}


def strata(y, task, bins):
    """Classification one-hot, multilabel indicators or regression quantile bins."""
    y = np.asarray(y)
    if task == 'multi_label_classification':
        return y.astype(bool)
    if task == 'regression':
        edges = np.unique(np.quantile(y.astype(float), np.linspace(0, 1, bins + 1)))
        labels = np.searchsorted(edges[1:-1], y, side='right')
    else:
        _, labels = np.unique(y, return_inverse=True)
    return np.eye(int(labels.max()) + 1, dtype=bool)[labels]


def sample_split(y, groups, task, per_class, val_fraction, bins, seed, max_samples=None):
    """Group-disjoint split first, then sample each stratum within each split."""
    if per_class < 2 or not 0 < val_fraction < 1 or bins < 2:
        raise ValueError('sampling requires per_class >= 2, 0 < val_fraction < 1, bins >= 2')
    if max_samples is not None and max_samples < 2:
        raise ValueError('max_samples must be at least 2')
    y, groups = np.asarray(y), np.asarray(groups)
    if len(y) != len(groups) or len(set(groups)) < 2:
        raise ValueError('at least two distinct subjects are required')
    indicators = strata(y, task, bins)
    total = indicators.sum(0)
    present = total > 0
    # Optimize only the group assignment and sample quotas; no probe is fitted here.
    splitter = GroupShuffleSplit(n_splits=128, test_size=val_fraction, random_state=seed)
    best = None
    for train, val in splitter.split(np.zeros(len(y)), groups=groups):
        counts = indicators[val].sum(0)
        missing = ((counts == 0) | (counts == total)) & present
        imbalance = np.mean(np.abs(counts[present] / total[present] - val_fraction))
        score = (int(missing.sum()), float(imbalance), abs(len(val) / len(y) - val_fraction))
        if best is None or score < best[0]:
            best = (score, train, val)
    rng = np.random.default_rng(seed)
    n_val = max(1, min(per_class - 1, round(per_class * val_fraction)))
    quotas = (per_class - n_val, n_val)
    if max_samples is None:
        limits = (len(y), len(y))
    else:
        val_limit = max(1, min(max_samples - 1, round(max_samples * val_fraction)))
        limits = (max_samples - val_limit, val_limit)
    selected = []
    for pool, quota, limit in zip(best[1:], quotas, limits):
        # Iterative deficits also handle overlapping multi-label classes.
        pool = rng.permutation(pool)
        rarity = (1 / np.maximum(indicators[pool].sum(0), 1)
                  if task == 'multi_label_classification' and max_samples is not None else 1)
        counts = np.zeros(indicators.shape[1], dtype=int)
        chosen = []
        while len(pool) and len(chosen) < limit:
            deficits = np.maximum(quota - counts, 0)
            scores = indicators[pool] @ (deficits * rarity)
            possible = np.flatnonzero(scores > 0)
            if not len(possible):
                break
            # Avoid exceeding any positive-label quota in multilabel sampling.
            if task == 'multi_label_classification':
                possible = possible[np.all(counts + indicators[pool[possible]] <= quota, axis=1)]
                if not len(possible):
                    break
            pick = possible[np.argmax(scores[possible])]
            idx = int(pool[pick])
            chosen.append(idx)
            counts += indicators[idx]
            pool = np.delete(pool, pick)
        selected.append(np.asarray(sorted(chosen), dtype=int))
    train, val = selected
    if not len(train) or not len(val) or set(groups[train]) & set(groups[val]):
        raise ValueError('could not form nonempty subject-disjoint probe splits')
    if task == 'classification' and len(np.unique(y[train])) < 2:
        raise ValueError('probe train has fewer than two observed classes')
    return train, val


def spectrum(x):
    x = np.asarray(x, dtype=np.float64)
    raw = np.linalg.svd(x, compute_uv=False)
    centered = x - x.mean(0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if raw.sum() <= 1e-14 or np.square(singular).sum() <= 1e-28:
        return {'rankme': np.nan, 'feature_variance': float(np.var(x, axis=0).mean()),
                'top1_singular_energy': np.nan}
    probabilities = raw / raw.sum()
    probabilities = probabilities[probabilities > 0]
    energy = np.square(singular)
    return {'rankme': float(np.exp(-np.sum(probabilities * np.log(probabilities)))),
            'feature_variance': float(np.var(x, axis=0).mean()),
            'top1_singular_energy': float(energy[0] / energy.sum())}


def logme(x, y, task):
    """Bayesian linear-regression log evidence, averaged over target dimensions.

    SVD fixed point follows the LogME method; the determinant includes the
    D-r zero eigenvalues when feature dimension exceeds sample count.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y)
    if task == 'classification':
        targets = np.stack([y == c for c in np.unique(y)], axis=1).astype(float)
    elif task == 'multi_label_classification':
        targets = y.astype(float)
        targets = targets[:, (targets.min(0) != targets.max(0))]
    else:
        targets = y.astype(float).reshape(-1, 1)
    if targets.shape[1] == 0:
        return np.nan
    n, d = x.shape
    u, s, _ = np.linalg.svd(x, full_matrices=False)
    eig = s * s
    result = []
    for target in targets.T:
        projection = u.T @ target
        projection2 = projection * projection
        residual = max(0., float(target @ target - projection2.sum()))
        alpha = beta = 1.0
        for _ in range(100):
            ratio = alpha / beta
            gamma = float(np.sum(eig / (eig + ratio)))
            m2 = float(np.sum(eig * projection2 / (eig + ratio) ** 2))
            residual2 = float(np.sum(projection2 * (ratio / (eig + ratio)) ** 2) + residual)
            new_alpha = max(gamma / (m2 + 1e-12), 1e-12)
            new_beta = max((n - gamma) / (residual2 + 1e-12), 1e-12)
            if abs(new_alpha / new_beta - ratio) / max(ratio, 1e-12) < 1e-4:
                alpha, beta = new_alpha, new_beta
                break
            alpha, beta = new_alpha, new_beta
        ratio = alpha / beta
        m2 = float(np.sum(eig * projection2 / (eig + ratio) ** 2))
        residual2 = float(np.sum(projection2 * (ratio / (eig + ratio)) ** 2) + residual)
        logdet = np.log(alpha + beta * eig).sum() + (d - len(eig)) * np.log(alpha)
        value = (d * np.log(alpha) + n * np.log(beta) - logdet
                 - beta * residual2 - alpha * m2 - n * np.log(2 * np.pi)) / (2 * n)
        result.append(value)
    return float(np.mean(result))


def hscore(x, y, task, bins):
    """Shrinkage H-score in the sample-space SVD; regression uses quantile bins."""
    x = np.asarray(x, dtype=np.float64)
    centered = x - x.mean(0)
    _, s, vh = np.linalg.svd(centered, full_matrices=False)
    d, n = x.shape[1], len(x)
    shrink = float(ledoit_wolf_shrinkage(x))
    variance = float(np.square(centered).sum() / (n * d))
    if variance <= 1e-20:
        return np.nan
    denominator = (1 - shrink) * s * s / n + shrink * variance
    denominator = np.maximum(denominator, 1e-20)
    groups = strata(y, task, bins)
    # Multilabel indicators are separate binary tasks; average their H-scores.
    label_sets = [np.asarray(y)[:, j] for j in range(groups.shape[1])] if task == 'multi_label_classification' else [np.argmax(groups, axis=1)]
    results = []
    for label in label_sets:
        values = np.unique(label)
        if len(values) < 2:
            continue
        value = 0.
        for c in values:
            subset = centered[label == c]
            projected = subset.mean(0) @ vh.T
            value += len(subset) / n * np.sum(projected * projected / denominator)
        results.append((1 - shrink) * value)
    return float(np.mean(results)) if results else np.nan


def _auc(y, scores, task, classes=None):
    y, scores = np.asarray(y), np.asarray(scores)
    if task == 'classification':
        if scores.ndim == 1:
            return float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else np.nan
        columns = [(y == c, scores[:, i]) for i, c in enumerate(classes)]
    else:
        if scores.ndim == 1:
            scores = scores[:, None]
        columns = [(y[:, i], scores[:, i]) for i in range(y.shape[1])]
    aucs = [roc_auc_score(t, s) for t, s in columns if len(np.unique(t)) == 2]
    return float(np.mean(aucs)) if aucs else np.nan


def probes(train_x, train_y, val_x, val_y, task, seed):
    scaler = StandardScaler().fit(train_x)
    x = scaler.transform(train_x)
    v = scaler.transform(val_x)
    train_y, val_y = np.asarray(train_y), np.asarray(val_y)
    values = {}
    for name in ('ridge', 'linear'):
        if task == 'regression':
            model = Ridge(alpha=1.0) if name == 'ridge' else LinearRegression()
            model.fit(x, train_y)
            prediction = model.predict(v)
            mae = float(mean_absolute_error(val_y, prediction))
            values.update({name + '_mae': mae,
                           name + '_rmse': float(np.sqrt(mean_squared_error(val_y, prediction))),
                           name + '_r2': float(r2_score(val_y, prediction)) if len(val_y) > 1 else np.nan,
                           name + '_probe': -mae})
        else:
            model = (RidgeClassifier(alpha=1.0) if name == 'ridge' else
                     LogisticRegression(C=1.0, max_iter=1000, random_state=seed))
            if task == 'multi_label_classification':
                model = OneVsRestClassifier(model)
            model.fit(x, train_y)
            prediction = model.predict(v)
            score = model.decision_function(v)
            if task == 'multi_label_classification' and np.asarray(score).ndim == 1:
                score = np.asarray(score)[:, None]
            f1 = float(f1_score(val_y, prediction, average='macro', zero_division=0))
            values.update({name + '_f1_macro': f1, name + '_probe': f1,
                           name + '_auroc': _auc(val_y, score, task, getattr(model, 'classes_', None))})
    return values


def evaluate(train_x, train_y, val_x, val_y, task, bins, seed):
    values = spectrum(train_x)
    values['logme'] = logme(train_x, train_y, task)
    values['hscore'] = hscore(train_x, train_y, task, bins)
    values.update(probes(train_x, train_y, val_x, val_y, task, seed))
    return values


def transfer_scores(raw):
    """Per task and trajectory, percentile-normalize only metrics common to all checkpoints."""
    out = raw.copy()
    out['transfer_score'] = np.nan
    out['transfer_metrics'] = ''
    keys = ['trajectory', 'modality', 'task_id']
    for _, idx in out.groupby(keys, dropna=False).groups.items():
        idx = list(idx)
        if len(idx) < 2:
            continue
        common = [m for m in METRICS if m in out and np.isfinite(out.loc[idx, m].to_numpy(float)).all()]
        if not common:
            continue
        normalized = []
        for metric in common:
            vals = out.loc[idx, metric].astype(float) * DIRECTION[metric]
            ranks = vals.rank(method='average')
            normalized.append((ranks - 1) / (len(idx) - 1))
        out.loc[idx, 'transfer_score'] = pd.concat(normalized, axis=1).mean(axis=1)
        out.loc[idx, 'transfer_metrics'] = ','.join(common)
    return out


def correlations(raw, finetune):
    columns = ['trajectory', 'modality', 'task_id', 'scope', 'metric', 'finetune_metric',
               'finetune_split', 'protocol_consistency', 'pairs', 'spearman', 'kendall', 'selected_checkpoint',
               'best_checkpoint', 'selection_regret', 'normalized_regret', 'metric_tie', 'best_tie']
    if finetune.empty:
        return pd.DataFrame(columns=columns)
    keys = ['trajectory', 'modality', 'task_id', 'checkpoint']
    if finetune.duplicated(keys).any():
        raise ValueError('fine-tuning results must be unique per trajectory/modality/task/checkpoint')
    merged = raw.merge(finetune, on=keys, how='inner', validate='one_to_one')
    result = []
    for key, part in merged.groupby(['trajectory', 'modality', 'task_id', 'scope'], dropna=False):
        if part['finetune_metric'].nunique() != 1 or part['finetune_mode'].nunique() != 1 or part['finetune_split'].nunique() != 1:
            raise ValueError(f'inconsistent fine-tuning metric/mode/split for {key}')
        sign = 1 if part['finetune_mode'].iloc[0] == 'max' else -1
        for metric in (*METRICS, 'transfer_score'):
            if metric not in part:
                continue
            valid = part[np.isfinite(part[metric]) & np.isfinite(part['finetune_value'])].copy()
            if len(valid) < 2:
                continue
            preference = DIRECTION.get(metric, 1)
            x = valid[metric].to_numpy(float) * preference
            y = valid['finetune_value'].to_numpy(float) * sign
            rho = float(spearmanr(x, y).statistic) if len(valid) >= 3 and np.ptp(x) and np.ptp(y) else np.nan
            tau = float(kendalltau(x, y).statistic) if len(valid) >= 3 and np.ptp(x) and np.ptp(y) else np.nan
            # Stable tie breaking by checkpoint ID, independent of FT outcomes.
            valid = valid.assign(_preference=x, _performance=y).sort_values('checkpoint')
            selected = valid.loc[valid['_preference'].idxmax()]
            best = valid.loc[valid['_performance'].idxmax()]
            regret = float(best['_performance'] - selected['_performance'])
            span = float(valid['_performance'].max() - valid['_performance'].min())
            result.append(dict(zip(['trajectory', 'modality', 'task_id', 'scope'], key), metric=metric,
                finetune_metric=part['finetune_metric'].iloc[0], finetune_split=part['finetune_split'].iloc[0],
                protocol_consistency=('unverified' if 'finetune_protocol' not in valid or valid['finetune_protocol'].isna().any()
                    or (valid['finetune_protocol'] == 'unverified').any() else
                    ('consistent' if valid['finetune_protocol'].nunique() == 1 else 'mixed')),
                pairs=len(valid), spearman=rho, kendall=tau,
                selected_checkpoint=selected['checkpoint'], best_checkpoint=best['checkpoint'],
                selection_regret=regret, normalized_regret=regret / span if span > 0 else np.nan,
                metric_tie=int(np.isclose(x, np.max(x), rtol=1e-12, atol=1e-12).sum()) > 1,
                best_tie=int(np.isclose(y, np.max(y), rtol=1e-12, atol=1e-12).sum()) > 1))
    return pd.DataFrame(result, columns=columns)


def leave_one_task_out(corr, selection='rank'):
    """Select a metric on other tasks, then measure its held-out task regret."""
    if selection not in ('rank', 'regret'):
        raise ValueError('selection must be rank or regret')
    keys = ['trajectory', 'modality', 'finetune_metric', 'finetune_split', 'protocol_consistency']
    columns = keys + ['heldout_task', 'training_tasks', 'selected_metric', 'selection_criterion', 'training_score',
                      'heldout_spearman', 'heldout_checkpoint', 'heldout_best_checkpoint',
                      'heldout_regret', 'heldout_normalized_regret']
    if corr.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for key, group in corr.groupby(keys, dropna=False):
        tasks = sorted(group['task_id'].unique())
        if len(tasks) < 3:
            continue
        for heldout in tasks:
            train = group[group['task_id'] != heldout].copy()
            test = group[(group['task_id'] == heldout) & np.isfinite(group['selection_regret'])]
            n_train = len(tasks) - 1
            if selection == 'regret':
                criterion = 'mean_regret'
                train = train[(~train['metric_tie']) & (~train['best_tie']) & np.isfinite(train['selection_regret'])]
                test = test[(~test['metric_tie']) & (~test['best_tie'])]
                train['_score'] = -train['selection_regret']
            elif train['pairs'].ge(3).all():
                criterion = 'mean_spearman'
                train = train[np.isfinite(train['spearman'])]
                train['_score'] = train['spearman']
            elif train['pairs'].eq(2).all():
                criterion = 'pairwise_accuracy'
                train = train[(~train['metric_tie']) & (~train['best_tie']) & np.isfinite(train['normalized_regret'])]
                test = test[(~test['metric_tie']) & (~test['best_tie'])]
                train['_score'] = 1 - train['normalized_regret']
            else:
                continue
            counts = train.groupby('metric')['task_id'].nunique()
            eligible = sorted(set(counts[counts == n_train].index) & set(test['metric']))
            if not eligible:
                continue
            means = train[train['metric'].isin(eligible)].groupby('metric')['_score'].mean()
            chosen = sorted(eligible, key=lambda metric: (-means[metric], metric))[0]
            result = test[test['metric'] == chosen].iloc[0]
            rows.append(dict(zip(keys, key), heldout_task=heldout, training_tasks=n_train,
                selected_metric=chosen, selection_criterion=criterion,
                training_score=float(-means[chosen] if criterion == 'mean_regret' else means[chosen]),
                heldout_spearman=result['spearman'], heldout_checkpoint=result['selected_checkpoint'],
                heldout_best_checkpoint=result['best_checkpoint'], heldout_regret=result['selection_regret'],
                heldout_normalized_regret=result['normalized_regret']))
    return pd.DataFrame(rows, columns=columns)
