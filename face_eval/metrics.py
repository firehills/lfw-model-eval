"""Verification scores, confusion counts and descriptive ROC metrics."""

import numpy as np


METRICS = ("cosine", "dot", "euclidean", "manhattan")


def unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("Cannot normalize a zero or nonfinite vector")
    return vector / norm


def score(probe: np.ndarray, templates: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        return np.clip(np.stack([unit(t) for t in templates]) @ unit(probe), -1, 1)
    if metric == "dot":
        return templates @ probe
    if metric == "euclidean":
        return np.linalg.norm(templates - probe, axis=1)
    if metric == "manhattan":
        return np.abs(templates - probe).sum(axis=1)
    raise ValueError(f"Unknown metric: {metric}")


def higher_is_better(metric: str) -> bool:
    if metric not in METRICS:
        raise ValueError(f"Unknown metric: {metric}")
    return metric in {"cosine", "dot"}


def verification_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float, metric: str) -> dict:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.ndim != 1 or labels.shape != scores.shape or not len(labels):
        raise ValueError("Expected equally sized nonempty label/score vectors")
    if not np.isfinite(scores).all() or not np.isfinite(threshold):
        raise ValueError("Scores and threshold must be finite")
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative:
        raise ValueError("Both genuine and impostor comparisons are required (at least two identities)")
    higher = higher_is_better(metric)
    accepted = scores >= threshold if higher else scores <= threshold
    ta = int((accepted & labels).sum())
    fa = int((accepted & ~labels).sum())
    fr, tr = positive - ta, negative - fa
    far, frr = fa / negative, fr / positive
    precision = ta / (ta + fa) if ta + fa else 0.0
    recall = ta / positive

    # Process tied scores together; accepting only part of a tie would inflate AUC.
    oriented = scores if higher else -scores
    order = np.argsort(-oriented, kind="stable")
    ordered_labels, ordered_scores = labels[order], oriented[order]
    endpoints = np.r_[np.flatnonzero(np.diff(ordered_scores)), len(labels) - 1]
    tpr = np.r_[0.0, np.cumsum(ordered_labels)[endpoints] / positive]
    fpr = np.r_[0.0, np.cumsum(~ordered_labels)[endpoints] / negative]
    fnr = 1 - tpr
    crossing = int(np.flatnonzero(fpr >= fnr)[0])
    previous = crossing - 1
    difference = fpr - fnr
    weight = -difference[previous] / (difference[crossing] - difference[previous])
    eer = fpr[previous] + weight * (fpr[crossing] - fpr[previous])
    return {
        "true_accepted": ta, "false_accepted": fa,
        "false_rejected": fr, "true_rejected": tr,
        "genuine_comparisons": positive, "impostor_comparisons": negative,
        "accuracy": (ta + tr) / len(labels),
        "balanced_accuracy": (recall + tr / negative) / 2,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_accept_rate": far, "false_reject_rate": frr,
        "true_accept_rate": recall, "true_reject_rate": tr / negative,
        "roc_auc": float(np.trapz(tpr, fpr)),
        "eer_interpolated": float(eer),
        "roc": {"false_accept_rate": fpr.tolist(), "true_accept_rate": tpr.tolist()},
    }
