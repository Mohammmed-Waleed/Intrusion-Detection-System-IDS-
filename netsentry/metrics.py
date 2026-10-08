"""IDS-oriented evaluation: detection rate, false-alarm rate, per-family and per-attack results."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)

from .datasets import NORMAL, Records


def calibrate_threshold(normal_scores, target_far: float) -> float:
    """Smallest threshold whose false-alarm rate on `normal_scores` is <= target_far.

    Alerts fire when ``score >= threshold``. Ties are handled conservatively: the
    returned value never lets more than ``floor(target_far * n)`` normal rows through.
    """
    s = np.sort(np.asarray(normal_scores, dtype=float))[::-1]
    if len(s) == 0:
        return 0.5
    k = int(np.floor(target_far * len(s)))
    if k >= len(s):
        return 0.0
    return float(np.nextafter(s[k], np.inf))


def binary_metrics(y_attack, score, threshold: float) -> dict:
    y = np.asarray(y_attack, dtype=bool)
    s = np.asarray(score, dtype=float)
    pred = s >= threshold
    tp = int(np.sum(pred & y))
    fp = int(np.sum(pred & ~y))
    tn = int(np.sum(~pred & ~y))
    fn = int(np.sum(~pred & y))
    div = lambda a, b: float(a / b) if b else 0.0  # noqa: E731
    precision, recall = div(tp, tp + fp), div(tp, tp + fn)
    both = y.any() and (~y).any()
    return {
        "threshold": float(threshold),
        "accuracy": div(tp + tn, len(y)),
        "detection_rate": recall,             # a.k.a. recall / TPR on attacks
        "false_alarm_rate": div(fp, fp + tn),  # FPR on normal traffic
        "precision": precision,
        "f1": div(2 * precision * recall, precision + recall),
        "roc_auc": float(roc_auc_score(y, s)) if both else None,
        "pr_auc": float(average_precision_score(y, s)) if both else None,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def roc_points(y_attack, score, max_points: int = 200) -> dict:
    y = np.asarray(y_attack, dtype=bool)
    if not (y.any() and (~y).any()):
        return {"fpr": [], "tpr": []}
    fpr, tpr, _ = roc_curve(y, score)
    if len(fpr) > max_points:
        keep = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
        fpr, tpr = fpr[keep], tpr[keep]
    return {"fpr": [round(float(v), 5) for v in fpr], "tpr": [round(float(v), 5) for v in tpr]}


def evaluate(detector, rec: Records, seen_attacks: set[str] | None = None,
             threshold: float | None = None) -> dict:
    """Score labelled records and compute everything the report needs.

    `seen_attacks` (labels present in training data) splits detection rate into attacks
    the model has seen before vs novel ones, the real test for an IDS.
    """
    if rec.category is None:
        raise ValueError("evaluation needs labelled records")
    t = detector.threshold if threshold is None else threshold
    pred = detector.predict(rec.X, threshold=t)
    y_attack = rec.is_attack
    score = pred["attack_score"].to_numpy()

    labels = [NORMAL] + [c for c in detector.classes if c != NORMAL]
    labels += sorted(set(rec.category) - set(labels))
    y_true, y_pred = rec.category.to_numpy(), pred["category"].to_numpy()
    p, r, f, sup = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    present = [i for i, n in enumerate(sup) if n > 0]

    attack_rows = []
    hit = pred["is_attack"].to_numpy()
    fam_ok = y_pred == y_true
    for name, idx in rec.attack.groupby(rec.attack).groups.items():
        idx = np.asarray(idx)
        if rec.category.iloc[idx[0]] == NORMAL:
            continue
        attack_rows.append({
            "attack": name,
            "category": rec.category.iloc[idx[0]],
            "n": int(len(idx)),
            "detection_rate": float(hit[idx].mean()),
            "family_accuracy": float(fam_ok[idx].mean()),
            "seen": None if seen_attacks is None else name in seen_attacks,
        })
    attack_rows.sort(key=lambda row: row["n"], reverse=True)

    out = {
        "n": int(len(rec)),
        "n_attack": int(y_attack.sum()),
        "binary": binary_metrics(y_attack, score, t),
        "binary_at_0.5": binary_metrics(y_attack, score, 0.5),
        "multiclass": {
            "accuracy": float(np.mean(fam_ok)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=[labels[i] for i in present],
                                       average="macro", zero_division=0)),
            "labels": labels,
            "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "per_class": {labels[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                      "f1": float(f[i]), "support": int(sup[i])} for i in range(len(labels))},
        },
        "per_attack": attack_rows,
        "roc": roc_points(y_attack, score),
    }
    if seen_attacks is not None:
        split = {}
        for key, want in (("seen", True), ("novel", False)):
            rows = [a for a in attack_rows if a["seen"] is want]
            n = sum(a["n"] for a in rows)
            split[key] = {"attack_types": len(rows), "n": n,
                          "detection_rate": (sum(a["detection_rate"] * a["n"] for a in rows) / n) if n else None}
        out["seen_vs_novel"] = split
    return out
