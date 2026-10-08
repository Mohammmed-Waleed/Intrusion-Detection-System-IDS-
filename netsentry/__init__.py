"""NetSentry: machine-learning intrusion detection trained on NSL-KDD or CICIDS2017."""

__version__ = "0.1.0"

from .detector import Detector, feature_importance  # noqa: E402
from .metrics import evaluate  # noqa: E402

__all__ = ["Detector", "evaluate", "feature_importance", "__version__"]
