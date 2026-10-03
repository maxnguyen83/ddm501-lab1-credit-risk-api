"""
Contract tests at the HTTP edge: what /predict and /predict/batch accept and
reject. The instructor's core validation cases live in test_api.py; these add
the cases a real caller is likely to get wrong, and check that the schema does
not reject values that are legitimate.
"""

import copy

import pytest
from fastapi.testclient import TestClient

from app.main import app

APPLICANT = {
    "limit_bal": 300000,
    "sex": 2,
    "education": 1,
    "marriage": 2,
    "age": 38,
    "pay_status": [-1, -1, -1, -1, -1, -1],
    "bill_amt": [12000, 11500, 11000, 10500, 10000, 9500],
    "pay_amt": [12000, 11500, 11000, 10500, 10000, 9500],
}


@pytest.fixture(scope="module")
def client():
    """Client bound to the app lifespan, so the model is actually loaded."""
    with TestClient(app) as c:
        yield c


def with_field(field, value):
    """Deep copy of the applicant with one field overwritten."""
    application = copy.deepcopy(APPLICANT)
    application[field] = value
    return application


class TestEdgeAccepts:
    """Legitimate values must not be rejected — over-strict is also a bug."""

    @pytest.mark.parametrize(
        "field,value",
        [
            ("age", 18),
            ("age", 100),
            ("limit_bal", 2_000_000),
            ("marriage", 3),
            ("pay_status", [-2, -2, -2, -2, -2, -2]),
            ("pay_status", [8, 8, 8, 8, 8, 8]),
            ("pay_amt", [0, 0, 0, 0, 0, 0]),
            ("bill_amt", [-1500, 0, 0, 0, 0, 0]),
        ],
    )
    def test_boundary_value_is_scored(self, client, field, value):
        assert client.post("/predict", json=with_field(field, value)).status_code == 200

    def test_integer_valued_floats_are_accepted(self, client):
        """Core banking may send 300000.0 for a limit; that is the same number."""
        response = client.post("/predict", json=with_field("limit_bal", 300000.0))
        assert response.status_code == 200


class TestEdgeRejects:
    """Malformed requests a real integration might send. All must be 422."""

    @pytest.mark.parametrize(
        "field,value",
        [
            ("limit_bal", "a lot"),
            ("sex", "female"),
            ("age", "38 years"),
            ("age", 38.5),
            ("pay_status", "000000"),
            ("pay_status", [0, 0, 0, 0, 0, "late"]),
            ("bill_amt", {"t-1": 12000}),
            ("pay_amt", None),
        ],
    )
    def test_wrong_type_is_rejected(self, client, field, value):
        response = client.post("/predict", json=with_field(field, value))
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"][1] == field

    @pytest.mark.parametrize("field", ["pay_status", "bill_amt", "pay_amt"])
    def test_seven_months_is_rejected(self, client, field):
        response = client.post("/predict", json=with_field(field, [0] * 7))
        assert response.status_code == 422
        assert response.json()["detail"][0]["type"] == "too_long"

    def test_both_custom_validators_report_in_one_response(self, client):
        """The caller sees every problem in one round trip, not one per retry."""
        bad = with_field("pay_status", [9, 0, 0, 0, 0, 0])
        bad["pay_amt"] = [100, -1, 100, 100, 100, 100]
        response = client.post("/predict", json=bad)
        assert response.status_code == 422
        fields = {error["loc"][-1] for error in response.json()["detail"]}
        assert fields == {"pay_status", "pay_amt"}

    def test_oversized_batch_is_rejected(self, client):
        """501 applications is one more than the documented limit."""
        response = client.post("/predict/batch", json={"applications": [APPLICANT] * 501})
        assert response.status_code == 422

    def test_one_bad_item_rejects_the_whole_batch(self, client):
        """The contract applies item by item; the error points at the bad one."""
        response = client.post(
            "/predict/batch", json={"applications": [APPLICANT, with_field("age", 12)]}
        )
        assert response.status_code == 422
        assert response.json()["detail"][0]["loc"] == ["body", "applications", 1, "age"]

    def test_batch_without_wrapper_is_rejected(self, client):
        """A bare list instead of {"applications": [...]} is a contract error."""
        response = client.post("/predict/batch", json=[APPLICANT])
        assert response.status_code == 422
