"""Shared fine-tuning early-stopping defaults and legacy resume support."""
import math


def early_stopping_patience(args):
    # Keep historical, self-contained autoresearch configs readable.
    legacy = args.get('finetune', {}).get('early_stopping_patience', 10)
    patience = int(args.get('optimization', {}).get('early_stopping_patience', legacy))
    if patience < 0:
        raise ValueError('optimization.early_stopping_patience must be non-negative')
    return patience


def stale_epochs_from_history(history, metric, maximize=True):
    """Recover consecutive non-improvements from old checkpoint histories.

    Video histories contain only evaluated epochs, so skipped evaluations do
    not consume patience. Ties and non-finite scores are not improvements.
    """
    best = -math.inf if maximize else math.inf
    stale = 0
    for row in history:
        if metric not in row:
            continue
        score = float(row[metric])
        improved = math.isfinite(score) and (score > best if maximize else score < best)
        if improved:
            best, stale = score, 0
        else:
            stale += 1
    return stale
