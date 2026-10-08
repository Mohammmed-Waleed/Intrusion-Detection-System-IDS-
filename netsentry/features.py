"""Feature preprocessing shared by every model."""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def signed_log1p(x):
    """Compress heavy-tailed counters (bytes, packets/s) while keeping their sign.

    CICIDS2017 has a few negative durations / window sizes, so a plain log1p won't do.
    """
    x = np.asarray(x, dtype="float64")
    return np.sign(x) * np.log1p(np.abs(x))


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    num = Pipeline([
        ("log", FunctionTransformer(signed_log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    parts = [("num", num, numeric)]
    if categorical:
        # Unseen protocol/service values at inference time become an all-zero vector.
        parts.append(("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical))
    return ColumnTransformer(parts, verbose_feature_names_out=False)


def source_column(transformed_name: str, categorical: list[str]) -> str:
    """Map a transformed feature (``service_http``) back to its input column (``service``)."""
    for c in categorical:
        if transformed_name.startswith(c + "_"):
            return c
    return transformed_name
