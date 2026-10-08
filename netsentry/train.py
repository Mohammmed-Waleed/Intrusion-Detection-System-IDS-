"""Train a detector, calibrate its alert threshold, and evaluate it on the held-out test set."""

from __future__ import annotations

import platform
import time
from datetime import datetime, timezone

import sklearn
from sklearn.pipeline import Pipeline

from . import __version__
from .datasets import DATASETS, load_dataset, stratified_split
from .detector import Detector, feature_importance
from .features import build_preprocessor
from .metrics import binary_metrics, calibrate_threshold, evaluate
from .models import make_model


def train(dataset: str = "nsl-kdd", data_path: str | None = None, model: str = "rf", *,
          target_far: float = 0.01, threshold: float | None = None, val_size: float = 0.1,
          test_size: float = 0.2, max_rows: int | None = None, balanced: bool = True,
          seed: int = 0, log=print) -> tuple[Detector, dict]:
    """Returns the fitted detector and a metrics dict (test + validation)."""
    t0 = time.perf_counter()
    train_rec, test_rec = load_dataset(dataset, data_path, seed=seed, test_size=test_size, max_rows=max_rows)
    log(f"loaded {dataset}: {len(train_rec):,} train / {len(test_rec):,} test rows, "
        f"{train_rec.X.shape[1]} features")

    # Hold out part of the training data to pick the alert threshold. The test set is
    # never used for any decision, so its numbers stay honest.
    fit_rec, val_rec = stratified_split(train_rec, val_size, seed + 1)
    categorical = [c for c in DATASETS[dataset]["categorical"] if c in train_rec.X.columns]
    numeric = [c for c in train_rec.X.columns if c not in categorical]
    pipe = Pipeline([("prep", build_preprocessor(numeric, categorical)),
                     ("clf", make_model(model, seed, balanced))])

    log(f"fitting {model} on {len(fit_rec):,} rows "
        f"({fit_rec.category.value_counts().to_dict()})")
    t1 = time.perf_counter()
    pipe.fit(fit_rec.X, fit_rec.category)
    fit_seconds = time.perf_counter() - t1

    meta = {
        "netsentry_version": __version__,
        "sklearn_version": sklearn.__version__,
        "python": platform.python_version(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": dataset,
        "data_path": str(data_path or DATASETS[dataset]["default_path"]),
        "model": model,
        "balanced": balanced,
        "seed": seed,
        "test_size": test_size,
        "val_size": val_size,
        "max_rows": max_rows,
        "features": list(train_rec.X.columns),
        "categorical": categorical,
        "target_far": None if threshold is not None else target_far,
        "train_attacks": sorted(set(train_rec.attack)),
        "n_fit": len(fit_rec),
        "n_val": len(val_rec),
        "n_test": len(test_rec),
        "fit_seconds": round(fit_seconds, 2),
    }
    det = Detector(pipe, 0.5, meta)

    val_scores = det.predict(val_rec.X)["attack_score"].to_numpy()
    det.threshold = threshold if threshold is not None else calibrate_threshold(
        val_scores[~val_rec.is_attack], target_far)
    log(f"alert threshold: {det.threshold:.4g}"
        + ("" if threshold is not None else f" (calibrated for {target_far:.2%} false alarms on validation)"))

    t2 = time.perf_counter()
    metrics = evaluate(det, test_rec, seen_attacks=set(train_rec.attack))
    metrics["predict_rows_per_second"] = round(len(test_rec) / max(time.perf_counter() - t2, 1e-9))
    metrics["validation"] = binary_metrics(val_rec.is_attack, val_scores, det.threshold)
    metrics["feature_importance"] = feature_importance(det, top=20)
    metrics["meta"] = meta
    metrics["total_seconds"] = round(time.perf_counter() - t0, 2)

    b = metrics["binary"]
    det.meta["test_summary"] = {k: b[k] for k in ("detection_rate", "false_alarm_rate", "roc_auc")} | {
        "family_accuracy": metrics["multiclass"]["accuracy"]}
    return det, metrics
