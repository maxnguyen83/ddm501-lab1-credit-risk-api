"""
ML model wrapper for credit default risk scoring.

The wrapper is the only place that knows how the model was trained: which
columns it expects, in which order, and what its output means. The API layer
above it deals in business objects, never in feature vectors.

TODO: Complete the four methods marked below.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from app.config import DECLINE_THRESHOLD, MODEL_PATH, MODEL_VERSION, REVIEW_THRESHOLD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Column order the model was fitted on. Changing this list without retraining
# is the classic training-serving skew bug (Lesson 04).
FEATURE_COLUMNS = (
    ["LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE"]
    + ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
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
    # TODO 1: Implement _load_model
    # =========================================================================
    # scripts/train_model.py saves a dict with two keys:
    #     {"pipeline": <sklearn Pipeline>, "metadata": {...}}
    #
    # Requirements:
    #   - load the file at self.model_path with joblib.load
    #   - put the pipeline in self.model and the metadata in self.metadata
    #   - log success
    #   - on FileNotFoundError, log an error that tells the reader how to fix it
    #     (they need to run scripts/train_model.py), then re-raise
    #

    def _load_model(self) -> None:
        """Load the trained pipeline from disk."""
        # TODO: implement
        #
        # try:
        #     bundle = joblib.load(???)
        #     self.model = bundle[???]
        #     self.metadata = bundle.get("metadata", {})
        #     logger.info("Model loaded from %s", self.model_path)
        # except FileNotFoundError:
        #     logger.error(...)
        #     raise
        pass

    # =========================================================================
    # TODO 2: Implement to_frame
    # =========================================================================
    # The API speaks in grouped lists (pay_status, bill_amt, pay_amt); the model
    # was trained on 23 flat columns. This method bridges the two.
    #
    # Requirements:
    #   - accept a list of application dicts (the output of .model_dump())
    #   - return a DataFrame whose columns are exactly FEATURE_COLUMNS, in order
    #   - map pay_status[0..5]  -> PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6
    #     (note: PAY_1 does not exist — that is an artifact of the original
    #      dataset, and reproducing it faithfully is part of the job)
    #   - map bill_amt[i] -> BILL_AMT{i+1}, pay_amt[i] -> PAY_AMT{i+1}
    #

    @staticmethod
    def to_frame(applications: List[Dict[str, Any]]) -> pd.DataFrame:
        """Flatten API payloads into the wide frame the model was trained on."""
        # TODO: implement
        #
        # rows = []
        # for a in applications:
        #     row = {"LIMIT_BAL": a["limit_bal"], ...}
        #     for i, name in enumerate(["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]):
        #         row[name] = a["pay_status"][i]
        #     ...
        #     rows.append(row)
        # return pd.DataFrame(rows, columns=FEATURE_COLUMNS)
        pass

    # -------------------------------------------------------------------------
    def predict_proba(self, applications: List[Dict[str, Any]]) -> np.ndarray:
        """Return the probability of default for each application. (PROVIDED)"""
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        frame = self.to_frame(applications)
        return self.model.predict_proba(frame)[:, 1]

    # =========================================================================
    # TODO 3: Implement decide
    # =========================================================================
    # Requirements:
    #   probability >= DECLINE_THRESHOLD  ->  {"risk_band": "HIGH",   "decision": "DECLINE"}
    #   probability >= REVIEW_THRESHOLD   ->  {"risk_band": "MEDIUM", "decision": "REVIEW"}
    #   otherwise                         ->  {"risk_band": "LOW",    "decision": "APPROVE"}

    @staticmethod
    def decide(probability: float) -> Dict[str, str]:
        """Turn a probability into a risk band and an underwriting decision."""
        # TODO: implement
        pass

    # =========================================================================
    # TODO 4: Implement score
    # =========================================================================
    # Requirements:
    #   - call self.predict_proba([application]) and take the first element
    #   - round the probability to 4 decimal places
    #   - return a dict with all six response fields:
    #     default_probability, risk_band, decision,
    #     review_threshold, decline_threshold, model_version

    def score(self, application: Dict[str, Any]) -> Dict[str, Any]:
        """Score one application and return the full response payload."""
        # TODO: implement
        pass

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
            probability = float(probability)
            result = {
                "default_probability": round(probability, 4),
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
