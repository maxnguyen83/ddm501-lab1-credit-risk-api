"""
Unit tests for the model wrapper in app/model.py.

Covers loading (and refusing to load), the API-to-feature mapping and the
response payload; the decision rule itself is in test_decision.py. Needs the
trained artifact: run `python scripts/train_model.py` first (CI does).
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

import app.config as config
import app.model as model_module
from app.model import FEATURE_COLUMNS, PAY_STATUS_COLUMNS, CreditRiskModel

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "credit_default.csv"

APPLICATION = {
    "limit_bal": 300000.0,
    "sex": 2,
    "education": 1,
    "marriage": 2,
    "age": 38,
    "pay_status": [-1, 0, 1, 2, 3, 4],
    "bill_amt": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
    "pay_amt": [21.0, 22.0, 23.0, 24.0, 25.0, 26.0],
}

RESPONSE_FIELDS = {
    "default_probability",
    "risk_band",
    "decision",
    "review_threshold",
    "decline_threshold",
    "model_version",
}


@pytest.fixture(scope="module")
def scorer():
    """The real trained model, loaded once for the module."""
    return CreditRiskModel()


class _ConstantPipeline:
    """Stand-in pipeline that always returns the same probability of default."""

    def __init__(self, probability):
        self.probability = probability

    def predict_proba(self, frame):
        n = len(frame)
        return np.column_stack([np.full(n, 1 - self.probability), np.full(n, self.probability)])


def constant_model(tmp_path, probability, metadata=None):
    """A CreditRiskModel backed by a saved _ConstantPipeline bundle."""
    path = tmp_path / "constant.joblib"
    bundle = {"pipeline": _ConstantPipeline(probability)}
    if metadata is not None:
        bundle["metadata"] = metadata
    joblib.dump(bundle, path)
    return CreditRiskModel(model_path=str(path))


# =============================================================================
# Loading
# =============================================================================
class TestLoadModel:
    def test_loads_pipeline_and_metadata(self, scorer):
        assert scorer.is_loaded()
        assert scorer.metadata["model_type"] == "HistGradientBoostingClassifier"
        assert scorer.metadata["features"] == FEATURE_COLUMNS

    def test_missing_file_is_logged_and_reraised(self, tmp_path, caplog):
        missing = tmp_path / "nope.joblib"
        with caplog.at_level(logging.ERROR, logger="app.model"):
            with pytest.raises(FileNotFoundError):
                CreditRiskModel(model_path=str(missing))
        # The log tells the reader how to fix it, not just that it broke.
        assert "scripts/train_model.py" in caplog.text
        assert str(missing) in caplog.text

    def test_refuses_a_model_trained_on_other_columns(self, tmp_path):
        reordered = list(reversed(FEATURE_COLUMNS))
        with pytest.raises(ValueError, match="trained on"):
            constant_model(tmp_path, 0.1, metadata={"features": reordered})

    def test_bundle_without_metadata_still_loads(self, tmp_path):
        loaded = constant_model(tmp_path, 0.1)
        assert loaded.is_loaded()
        assert loaded.metadata == {}

    def test_predict_proba_needs_a_loaded_model(self, tmp_path):
        loaded = constant_model(tmp_path, 0.1)
        loaded.model = None
        with pytest.raises(RuntimeError, match="not loaded"):
            loaded.predict_proba([APPLICATION])

    def test_get_model_is_a_singleton(self, monkeypatch):
        monkeypatch.setattr(model_module, "_model_instance", None)
        first = model_module.get_model()
        assert model_module.get_model() is first


# =============================================================================
# to_frame: API payload -> the 23 training columns
# =============================================================================
class TestToFrame:
    def test_columns_are_exactly_feature_columns_in_order(self):
        frame = CreditRiskModel.to_frame([APPLICATION])
        assert list(frame.columns) == FEATURE_COLUMNS
        assert len(FEATURE_COLUMNS) == 23
        assert "PAY_1" not in frame.columns

    def test_grouped_lists_map_to_the_right_months(self):
        row = CreditRiskModel.to_frame([APPLICATION]).iloc[0]
        # pay_status[0] is PAY_0, pay_status[1] is PAY_2: the dataset skips PAY_1.
        assert [row[c] for c in ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]] == [
            -1, 0, 1, 2, 3, 4,
        ]
        assert [row[f"BILL_AMT{i}"] for i in range(1, 7)] == APPLICATION["bill_amt"]
        assert [row[f"PAY_AMT{i}"] for i in range(1, 7)] == APPLICATION["pay_amt"]
        assert (row["LIMIT_BAL"], row["SEX"], row["EDUCATION"], row["MARRIAGE"], row["AGE"]) == (
            300000.0, 2, 1, 2, 38,
        )

    def test_one_row_per_application_in_order(self):
        second = dict(APPLICATION, age=55)
        frame = CreditRiskModel.to_frame([APPLICATION, second, APPLICATION])
        assert frame["AGE"].tolist() == [38, 55, 38]

    def test_round_trips_a_training_row(self):
        """Training-serving parity: a CSV row rebuilt from its API form is identical."""
        training = pd.read_csv(DATA_PATH, nrows=5).drop(columns=["default_payment_next_month"])
        applications = [
            {
                "limit_bal": r["LIMIT_BAL"],
                "sex": r["SEX"],
                "education": r["EDUCATION"],
                "marriage": r["MARRIAGE"],
                "age": r["AGE"],
                "pay_status": [r[c] for c in PAY_STATUS_COLUMNS],
                "bill_amt": [r[f"BILL_AMT{i}"] for i in range(1, 7)],
                "pay_amt": [r[f"PAY_AMT{i}"] for i in range(1, 7)],
            }
            for _, r in training.iterrows()
        ]
        rebuilt = CreditRiskModel.to_frame(applications)
        pd.testing.assert_frame_equal(rebuilt, training[FEATURE_COLUMNS], check_dtype=False)


# =============================================================================
# score / score_batch: the response payload
# =============================================================================
class TestScore:
    def test_returns_the_six_response_fields(self, scorer):
        result = scorer.score(APPLICATION)
        assert set(result) == RESPONSE_FIELDS
        assert result["review_threshold"] == config.REVIEW_THRESHOLD
        assert result["decline_threshold"] == config.DECLINE_THRESHOLD
        assert result["model_version"] == config.MODEL_VERSION

    def test_probability_is_a_plain_float_rounded_to_4dp(self, scorer):
        probability = scorer.score(APPLICATION)["default_probability"]
        assert type(probability) is float
        assert 0.0 <= probability <= 1.0
        assert probability == round(probability, 4)

    def test_score_matches_score_batch(self, scorer):
        assert scorer.score_batch([APPLICATION]) == [scorer.score(APPLICATION)]

    @pytest.mark.parametrize(
        "raw,shown,decision", [(0.29996, 0.3, "REVIEW"), (0.59996, 0.6, "DECLINE")]
    )
    def test_decision_uses_the_probability_the_caller_sees(self, tmp_path, raw, shown, decision):
        """0.29996 is shown as 0.3, so it must be decided as 0.3 — in both paths."""
        loaded = constant_model(tmp_path, raw)
        single = loaded.score(APPLICATION)
        assert (single["default_probability"], single["decision"]) == (shown, decision)
        assert loaded.score_batch([APPLICATION]) == [single]

    def test_batch_keeps_request_order(self, scorer):
        risky = dict(APPLICATION, pay_status=[4, 3, 3, 2, 2, 2], pay_amt=[0.0] * 6)
        good, bad = scorer.score_batch([APPLICATION, risky])
        assert bad["default_probability"] > good["default_probability"]
