"""
ML model wrapper for credit default risk scoring.

The wrapper is the only place that knows how the model was trained: which
columns it expects, in which order, and what its output means. The API layer
above it deals in business objects, never in feature vectors.
"""

import logging
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from app.config import DECLINE_THRESHOLD, MODEL_PATH, MODEL_VERSION, REVIEW_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# pay_status[0..5] in the API -> these columns. There is no PAY_1: the original
# UCI file skips it, and the model was trained on that naming.
PAY_STATUS_COLUMNS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]

# Column order the model was fitted on. Changing this list without retraining
# is the classic training-serving skew bug (Lesson 04).
FEATURE_COLUMNS = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + PAY_STATUS_COLUMNS
    + [f"BILL_AMT{i}" for i in range(1, 7)]
    + [f"PAY_AMT{i}" for i in range(1, 7)]
)


class CreditRiskModel:
    """Loads the trained pipeline and turns applications into decisions."""

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model = None
        self.metadata: Dict[str, Any] = {}
        self._load_model()

    # =========================================================================
    # Loading
    # =========================================================================
    # scripts/train_model.py saves a dict with two keys:
    #     {"pipeline": <sklearn Pipeline>, "metadata": {...}}

    def _load_model(self) -> None:
        """Load the trained pipeline from disk.

        Raises FileNotFoundError if the artifact is missing, and ValueError if
        it was trained on a different feature list than FEATURE_COLUMNS.
        """
        try:
            bundle = joblib.load(self.model_path)
        except FileNotFoundError:
            logger.error(
                "Model file not found at %s. Train it first with "
                "'python scripts/train_model.py', or point MODEL_PATH at the artifact.",
                self.model_path,
            )
            raise

        metadata = bundle.get("metadata", {})
        trained_on = metadata.get("features")
        if trained_on is not None and list(trained_on) != FEATURE_COLUMNS:
            # Refuse to serve rather than score with columns in the wrong order.
            raise ValueError(
                f"Model at {self.model_path} was trained on {trained_on}, "
                f"but the API sends {FEATURE_COLUMNS}"
            )

        self.model = bundle["pipeline"]
        self.metadata = metadata
        logger.info(
            "Model loaded from %s (type=%s, trained_at=%s)",
            self.model_path,
            metadata.get("model_type", "unknown"),
            metadata.get("trained_at", "unknown"),
        )

    # =========================================================================
    # Feature frame
    # =========================================================================
    # The API speaks in grouped lists (pay_status, bill_amt, pay_amt); the model
    # was trained on 23 flat columns. This method bridges the two.

    @staticmethod
    def to_frame(applications: List[Dict[str, Any]]) -> pd.DataFrame:
        """Flatten API payloads into the wide frame the model was trained on."""
        rows = []
        for a in applications:
            row: Dict[str, Any] = {
                "LIMIT_BAL": a["limit_bal"],
                "SEX": a["sex"],
                "EDUCATION": a["education"],
                "MARRIAGE": a["marriage"],
                "AGE": a["age"],
            }
            for i, name in enumerate(PAY_STATUS_COLUMNS):
                row[name] = a["pay_status"][i]
            for i in range(6):
                row[f"BILL_AMT{i + 1}"] = a["bill_amt"][i]
                row[f"PAY_AMT{i + 1}"] = a["pay_amt"][i]
            rows.append(row)
        # columns= fixes the order, whatever order the dict keys were built in.
        return pd.DataFrame(rows, columns=FEATURE_COLUMNS)

    # -------------------------------------------------------------------------
    def predict_proba(self, applications: List[Dict[str, Any]]) -> np.ndarray:
        """Return the probability of default for each application. (PROVIDED)"""
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        frame = self.to_frame(applications)
        return self.model.predict_proba(frame)[:, 1]

    # =========================================================================
    # Decision rule
    # =========================================================================
    #   probability >= DECLINE_THRESHOLD  ->  HIGH   / DECLINE
    #   probability >= REVIEW_THRESHOLD   ->  MEDIUM / REVIEW
    #   otherwise                         ->  LOW    / APPROVE
    # The highest threshold is checked first: checked the other way round,
    # every score above 0.30 would stop at REVIEW and nothing would be declined.

    @staticmethod
    def decide(probability: float) -> Dict[str, str]:
        """Turn a probability into a risk band and an underwriting decision."""
        if probability >= DECLINE_THRESHOLD:
            return {"risk_band": "HIGH", "decision": "DECLINE"}
        if probability >= REVIEW_THRESHOLD:
            return {"risk_band": "MEDIUM", "decision": "REVIEW"}
        return {"risk_band": "LOW", "decision": "APPROVE"}

    # =========================================================================
    # Scoring
    # =========================================================================
    def score(self, application: Dict[str, Any]) -> Dict[str, Any]:
        """Score one application and return the full response payload."""
        probability = round(float(self.predict_proba([application])[0]), 4)
        result: Dict[str, Any] = {
            "default_probability": probability,
            "review_threshold": REVIEW_THRESHOLD,
            "decline_threshold": DECLINE_THRESHOLD,
            "model_version": MODEL_VERSION,
        }
        # Decide on the rounded value the caller sees, so the response can never
        # show 0.3 next to APPROVE.
        result.update(self.decide(probability))
        return result

    # -------------------------------------------------------------------------
    def score_batch(self, applications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Score many applications in a single vectorised pass. (PROVIDED)

        Note this calls predict_proba ONCE for the whole batch rather than
        looping over self.score. One call to the model for 500 applicants is
        far cheaper than 500 calls — the same reasoning behind batch serving
        in Lesson 08.
        """
        probabilities = self.predict_proba(applications)
        results = []
        for probability in probabilities:
            # Rounded before deciding, exactly as in score().
            probability = round(float(probability), 4)
            result = {
                "default_probability": probability,
                "review_threshold": REVIEW_THRESHOLD,
                "decline_threshold": DECLINE_THRESHOLD,
                "model_version": MODEL_VERSION,
            }
            result.update(self.decide(probability))
            results.append(result)
        return results

    # -------------------------------------------------------------------------
    def is_loaded(self) -> bool:
        """Check whether the model is ready to serve. (PROVIDED)"""
        return self.model is not None


# =============================================================================
# Singleton accessor
# =============================================================================
_model_instance: Optional[CreditRiskModel] = None


def get_model() -> CreditRiskModel:
    """Get or create the model singleton."""
    global _model_instance
    if _model_instance is None:
        _model_instance = CreditRiskModel()
    return _model_instance
