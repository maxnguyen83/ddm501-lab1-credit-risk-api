"""
Tests for the Credit Default Risk Scoring API.

Run with:
    pytest tests/ -v
    pytest tests/ -v --cov=app --cov-report=term-missing

TODO: Complete the test cases marked below.
"""

import copy

import pytest
from fastapi.testclient import TestClient

from app.main import app

# A well-behaved applicant: always paid on time, low utilisation.
GOOD_APPLICANT = {
    "limit_bal": 300000,
    "sex": 2,
    "education": 1,
    "marriage": 2,
    "age": 38,
    "pay_status": [-1, -1, -1, -1, -1, -1],
    "bill_amt": [12000, 11500, 11000, 10500, 10000, 9500],
    "pay_amt": [12000, 11500, 11000, 10500, 10000, 9500],
}

# An applicant in trouble: months behind, near the limit, paying almost nothing.
RISKY_APPLICANT = {
    "limit_bal": 20000,
    "sex": 1,
    "education": 3,
    "marriage": 1,
    "age": 24,
    "pay_status": [4, 3, 3, 2, 2, 2],
    "bill_amt": [19800, 19500, 19000, 18500, 18000, 17500],
    "pay_amt": [0, 0, 200, 0, 300, 0],
}


@pytest.fixture(scope="module")
def client():
    """Client bound to the app lifespan, so the model is actually loaded.

    Using `with TestClient(app)` rather than a bare TestClient(app) matters:
    only the context manager runs the lifespan hook. Without it the model is
    never loaded and every test sees a 503.
    """
    with TestClient(app) as c:
        yield c


# =============================================================================
# Health and info (PROVIDED — use these as your template)
# =============================================================================
class TestHealthEndpoint:
    """The /health endpoint."""

    def test_returns_200(self, client):
        assert client.get("/health").status_code == 200

    def test_response_shape(self, client):
        data = client.get("/health").json()
        assert set(data) == {"status", "model_loaded", "model_version"}
        assert isinstance(data["model_loaded"], bool)

    def test_model_is_loaded(self, client):
        data = client.get("/health").json()
        assert data["model_loaded"] is True
        assert data["status"] == "healthy"


class TestInfoEndpoints:
    """The / and /model/info endpoints."""

    def test_root_returns_api_info(self, client):
        data = client.get("/").json()
        assert {"name", "version", "docs", "health"} <= set(data)

    def test_model_info_reports_metrics(self, client):
        data = client.get("/model/info").json()
        assert data["is_loaded"] is True
        assert 0.5 < data["metrics"]["roc_auc"] <= 1.0


# =============================================================================
# TODO 1: Happy-path tests for /predict
# =============================================================================
class TestPredictEndpoint:
    """The /predict endpoint — happy path."""

    def test_valid_application_returns_200(self, client):
        """POST GOOD_APPLICANT to /predict and assert the status code."""
        # TODO: implement
        pass

    def test_probability_is_a_valid_probability(self, client):
        """Assert default_probability is between 0.0 and 1.0 inclusive."""
        # TODO: implement
        pass

    def test_response_contains_every_field(self, client):
        """Assert the response has exactly the six documented fields.

        Hint: assert set(data) == {...}. Use == rather than <=, so that an
        accidental extra field — say, one carrying applicant data — fails the
        test instead of slipping into production.
        """
        # TODO: implement
        pass

    def test_decision_is_consistent_with_thresholds(self, client):
        """The decision must follow from the score — no third source of truth.

        For both GOOD_APPLICANT and RISKY_APPLICANT, read back
        default_probability, review_threshold and decline_threshold from the
        response and assert that decision and risk_band are what those three
        numbers imply.
        """
        # TODO: implement
        pass

    def test_deterministic(self, client):
        """The same request twice must give the same score.

        Obvious? It stops being obvious the moment someone adds a timestamp
        feature, a random seed, or a cache.
        """
        # TODO: implement
        pass


# =============================================================================
# TODO 2: Behavioural tests
# =============================================================================
class TestModelBehaviour:
    """Properties the model must satisfy."""

    def test_risky_scores_higher_than_good(self, client):
        """RISKY_APPLICANT must get a higher probability than GOOD_APPLICANT."""
        # TODO: implement
        pass

    def test_more_delay_never_lowers_risk(self, client):
        """Monotonicity: worse repayment history must not reduce the score.

        Take GOOD_APPLICANT, copy it twice (use copy.deepcopy), set
        pay_status to [1, 0, 0, 0, 0, 0] on one and [4, 3, 3, 2, 2, 2] on the
        other, and assert the second scores at least as high as the first.
        """
        # TODO: implement
        pass


# =============================================================================
# TODO 3: Validation tests
# =============================================================================
# Every one of these must come back 422 — rejected by the schema, never
# reaching the model.
class TestValidation:
    """Bad input must fail at the edge."""

    @pytest.mark.parametrize(
        "field,value",
        [
            ("sex", 3),
            ("education", 9),
            ("marriage", 0),
            ("age", 12),
            ("age", 150),
            ("limit_bal", 0),
            ("limit_bal", -5000),
        ],
    )
    def test_out_of_range_value_is_rejected(self, client, field, value):
        """Copy GOOD_APPLICANT, overwrite one field, assert 422.

        Hint: parametrize runs this once per (field, value) pair, so seven
        tests come out of one function body.
        """
        # TODO: implement
        pass

    def test_missing_field_is_rejected(self, client):
        """Delete a required field and assert 422."""
        # TODO: implement
        pass

    def test_wrong_list_length_is_rejected(self, client):
        """Send pay_status with 3 entries instead of 6 and assert 422."""
        # TODO: implement
        pass

    def test_pay_status_out_of_domain_is_rejected(self, client):
        """Send pay_status = [99, 0, 0, 0, 0, 0] and assert 422.

        This one only passes if you wrote the custom validator in schemas.py.
        Field bounds alone will not catch it.
        """
        # TODO: implement
        pass

    def test_negative_payment_is_rejected(self, client):
        """Send a negative value in pay_amt and assert 422."""
        # TODO: implement
        pass

    def test_empty_body_is_rejected(self, client):
        """POST {} and assert 422."""
        # TODO: implement
        pass


# =============================================================================
# TODO 4: Batch tests
# =============================================================================
class TestBatchEndpoint:
    """The /predict/batch endpoint."""

    def test_returns_one_result_per_application(self, client):
        """Send three applications, assert total_count and list length."""
        # TODO: implement
        pass

    def test_order_is_preserved(self, client):
        """Send [GOOD, RISKY] and assert the second result scores higher."""
        # TODO: implement
        pass

    def test_matches_single_prediction(self, client):
        """A batch of one must give exactly what /predict gives.

        This is the test that catches a batch path which quietly reorders
        columns or skips a preprocessing step.
        """
        # TODO: implement
        pass

    def test_empty_batch_is_rejected(self, client):
        """POST {"applications": []} and assert 422."""
        # TODO: implement
        pass
