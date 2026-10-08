"""A trained model + its alert threshold, bundled so it can be saved and served."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .datasets import NORMAL


class Detector:
    """Wraps a fitted sklearn pipeline that classifies traffic into families.

    The pipeline is multiclass (``normal``, ``DoS``, ``Probe``...). The binary alert
    decision comes from ``attack_score = 1 - P(normal)`` compared against ``threshold``,
    which is calibrated to a target false-alarm rate at training time. If a row is an
    alert, its family is the most likely attack class.
    """

    FORMAT = 1

    def __init__(self, pipeline, threshold: float = 0.5, meta: dict | None = None):
        self.pipeline = pipeline
        self.threshold = float(threshold)
        self.meta = meta or {}

    @property
    def classes(self) -> list[str]:
        return [str(c) for c in self.pipeline.classes_]

    @property
    def features(self) -> list[str]:
        return list(self.meta["features"])

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.features if c not in X.columns]
        if missing:
            more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
            raise ValueError(f"input is missing {len(missing)} feature column(s) the model was trained on: "
                             f"{missing[:8]}{more}")
        return X[self.features]

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        p = self.pipeline.predict_proba(self._align(X))
        return pd.DataFrame(p, columns=self.classes, index=X.index)

    def predict(self, X: pd.DataFrame, threshold: float | None = None) -> pd.DataFrame:
        """One row per input: attack_score, is_attack, category, family_score."""
        proba = self.predict_proba(X)
        t = self.threshold if threshold is None else threshold
        score = 1.0 - proba[NORMAL] if NORMAL in proba else pd.Series(1.0, index=proba.index)
        attack_cols = [c for c in proba.columns if c != NORMAL]
        family = proba[attack_cols].idxmax(axis=1)
        is_attack = score >= t
        return pd.DataFrame({
            "attack_score": score.round(6),
            "is_attack": is_attack,
            "category": family.where(is_attack, NORMAL),
            "family_score": proba[attack_cols].max(axis=1).round(6),
        }, index=X.index)

    # ------------------------------------------------------------------ persistence

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"format": self.FORMAT, "pipeline": self.pipeline,
                     "threshold": self.threshold, "meta": self.meta}, path, compress=3)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Detector":
        """Only load bundles you trust: joblib files can execute code when unpickled."""
        blob = joblib.load(path)
        if not isinstance(blob, dict) or blob.get("format") != cls.FORMAT:
            raise ValueError(f"{path} is not a NetSentry model bundle")
        return cls(blob["pipeline"], blob["threshold"], blob["meta"])

    def describe(self) -> str:
        m = self.meta
        return (f"{m.get('model', '?')} on {m.get('dataset', '?')} | classes: {', '.join(self.classes)} | "
                f"threshold {self.threshold:.4g} (target FAR {m.get('target_far')})")


def feature_importance(detector: Detector, top: int | None = None) -> list[tuple[str, float]]:
    """Importance per *input* column (one-hot columns summed back to their source).

    Uses impurity importance for tree ensembles and mean |coef| for linear models.
    Returns [] for models without either (gradient boosting, MLP).
    """
    from .features import source_column

    prep, clf = detector.pipeline.named_steps["prep"], detector.pipeline.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        imp = np.asarray(clf.feature_importances_, dtype=float)
    elif hasattr(clf, "coef_"):
        imp = np.abs(np.asarray(clf.coef_, dtype=float)).mean(axis=0)
    else:
        return []
    cats = list(detector.meta.get("categorical", []))
    agg: dict[str, float] = {}
    for name, v in zip(prep.get_feature_names_out(), imp):
        col = source_column(str(name), cats)
        agg[col] = agg.get(col, 0.0) + float(v)
    total = sum(agg.values()) or 1.0
    ranked = sorted(((k, v / total) for k, v in agg.items()), key=lambda kv: kv[1], reverse=True)
    return ranked[:top] if top else ranked
