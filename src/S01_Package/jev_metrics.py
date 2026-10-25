"""P1 record/unit weights and scalar scores without model or runtime dependencies.

The numerical operations and output keys are preserved from the fitting module.
Saved-output reports retain their own calibration/ECE validation and formatting.
"""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import f1_score


def record_unit_weights(unit, record):
    unit, record = np.asarray(unit).astype(str), np.asarray(record).astype(str)
    if unit.ndim != 1 or record.shape != unit.shape or not len(unit):
        raise ValueError("Need aligned nonempty unit/record arrays")
    out = np.zeros(len(unit), dtype=np.float64)
    for b in np.unique(unit):
        records = np.unique(record[unit == b])
        for r in records:
            mask = record == r
            if np.any(unit[mask] != b):
                raise ValueError("One recording cannot belong to different physical units")
            out[mask] = 1.0 / (len(np.unique(unit)) * len(records) * mask.sum())
    return out


def score_logits(logits, y, weights):
    a = np.asarray(logits, dtype=np.float64)
    if a.ndim != 2 or a.shape != (len(y), 3) or not np.isfinite(a).all():
        raise ValueError("Invalid class logits")
    logp = a - logsumexp(a, axis=1, keepdims=True)
    p, predicted = np.exp(logp), np.argmax(a, axis=1)
    truth = np.eye(3)[y]
    return {"nll": float(weights @ -logp[np.arange(len(y)), y]),
            "error": float(weights @ (predicted != y)),
            "macro_f1": float(f1_score(y, predicted, labels=[0,1,2], average="macro", sample_weight=weights, zero_division=0)),
            "brier": float(weights @ np.square(p-truth).sum(1))}


def score_probabilities(probabilities, y, weights, *, floor=1e-12):
    """Score the actual probabilities; a log floor does not renormalize them."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.shape != (len(y), 3) or not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1):
        raise ValueError("Invalid class probability distribution")
    predicted = p.argmax(1)
    truth = np.eye(3)[y]
    py = p[np.arange(len(y)), y]
    return {"nll": float(weights @ -np.log(np.maximum(py, floor))),
            "error": float(weights @ (predicted != y)),
            "macro_f1": float(f1_score(y, predicted, labels=[0,1,2], average="macro", sample_weight=weights, zero_division=0)),
            "brier": float(weights @ np.square(p-truth).sum(1)),
            "log_floor_weight": float(weights @ (py < floor))}
