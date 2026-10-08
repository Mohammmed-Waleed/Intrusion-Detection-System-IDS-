"""Model zoo. All are plain scikit-learn estimators, so no GPU or extra installs are needed."""

from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

MODELS = {
    "rf": "Random forest (default; strong, fast, gives feature importances)",
    "et": "Extra trees",
    "hgb": "Histogram gradient boosting",
    "logreg": "Logistic regression (linear baseline)",
    "mlp": "Multi-layer perceptron (128-64)",
}


def make_model(name: str, seed: int = 0, balanced: bool = False):
    """`balanced` reweights classes inversely to frequency (helps rare U2R/R2L recall,
    at the cost of more false alarms)."""
    cw = "balanced" if balanced else None
    if name == "rf":
        return RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=seed,
                                      class_weight="balanced_subsample" if balanced else None)
    if name == "et":
        return ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=seed, class_weight=cw)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, early_stopping=True,
                                              random_state=seed, class_weight=cw)
    if name == "logreg":
        return LogisticRegression(max_iter=3000, class_weight=cw)
    if name == "mlp":
        # early_stopping=True crashes on string class labels in scikit-learn 1.7, so stop
        # on training-loss plateau instead.
        return MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=60, random_state=seed)
    raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
