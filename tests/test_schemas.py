"""
Unit tests for the API contract in app/schemas.py.

These run without the model or the HTTP layer: they check the schema on its
own, so a broken bound or validator is reported here and not as a confusing
endpoint failure.
"""

import copy

import pytest
from pydantic import ValidationError

from app.schemas import (
    BatchPredictionRequest,
    CreditApplication,
    HealthResponse,
    PredictionResponse,
)

VALID = {
    "limit_bal": 300000,
    "sex": 2,
    "education": 1,
    "marriage": 2,
    "age": 38,
    "pay_status": [-1, -1, -1, -1, -1, -1],
    "bill_amt": [12000, 11500, 11000, 10500, 10000, 9500],
    "pay_amt": [12000, 11500, 11000, 10500, 10000, 9500],
}

VALID_RESPONSE = {
    "default_probability": 0.1234,
    "risk_band": "LOW",
    "decision": "APPROVE",
    "review_threshold": 0.30,
    "decline_threshold": 0.60,
    "model_version": "1.0.0",
}


def with_field(field, value, base=VALID):
    """Deep copy of a payload with one field overwritten."""
    payload = copy.deepcopy(base)
    payload[field] = value
    return payload


def error_fields(exc: ValidationError) -> set:
    """Top-level field names that failed validation."""
    return {error["loc"][0] for error in exc.errors()}


# =============================================================================
# CreditApplication: accepted input
# =============================================================================
class TestCreditApplicationAccepts:
    def test_valid_payload(self):
        app = CreditApplication(**VALID)
        dumped = app.model_dump()
        assert set(dumped) == set(VALID)
        assert dumped["bill_amt"] == [float(v) for v in VALID["bill_amt"]]

    @pytest.mark.parametrize(
        "field,value",
        [
            ("age", 18),
            ("age", 100),
            ("limit_bal", 2_000_000),
            ("limit_bal", 0.01),
            ("marriage", 1),
            ("marriage", 3),
            ("pay_status", [-2, -2, -2, -2, -2, -2]),
            ("pay_status", [8, 8, 8, 8, 8, 8]),
            ("pay_amt", [0, 0, 0, 0, 0, 0]),
        ],
    )
    def test_boundary_values_are_inclusive(self, field, value):
        assert getattr(CreditApplication(**with_field(field, value)), field) == value

    def test_negative_bill_is_allowed(self):
        """A negative statement means the customer is in credit; it is real data."""
        app = CreditApplication(**with_field("bill_amt", [-500, 0, 100, 200, 300, 400]))
        assert app.bill_amt[0] == -500.0

    def test_documented_example_is_valid(self):
        """The example shown in Swagger must itself pass the contract."""
        example = CreditApplication.model_json_schema()["examples"][0]
        assert CreditApplication(**example).model_dump() == CreditApplication(**VALID).model_dump()


# =============================================================================
# CreditApplication: rejected input
# =============================================================================
class TestCreditApplicationRejects:
    @pytest.mark.parametrize(
        "field,value",
        [
            ("sex", 0),
            ("sex", 3),
            ("education", 0),
            ("education", 5),
            ("marriage", 0),
            ("marriage", 4),
            ("age", 17),
            ("age", 101),
            ("age", 30.5),
            ("limit_bal", 0),
            ("limit_bal", -1),
            ("limit_bal", 2_000_001),
            ("limit_bal", "a lot"),
        ],
    )
    def test_scalar_out_of_domain(self, field, value):
        with pytest.raises(ValidationError) as exc:
            CreditApplication(**with_field(field, value))
        assert error_fields(exc.value) == {field}

    @pytest.mark.parametrize("field", ["pay_status", "bill_amt", "pay_amt"])
    @pytest.mark.parametrize("length", [0, 5, 7])
    def test_monthly_lists_need_exactly_six_values(self, field, length):
        with pytest.raises(ValidationError) as exc:
            CreditApplication(**with_field(field, [0] * length))
        assert error_fields(exc.value) == {field}

    @pytest.mark.parametrize("field", sorted(VALID))
    def test_every_field_is_required(self, field):
        payload = copy.deepcopy(VALID)
        del payload[field]
        with pytest.raises(ValidationError) as exc:
            CreditApplication(**payload)
        assert exc.value.errors()[0]["type"] == "missing"

    @pytest.mark.parametrize("field", ["bill_amt", "pay_amt"])
    @pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_amounts(self, field, bad):
        """NaN would otherwise be scored silently as a missing value."""
        values = [1000.0] * 6
        values[2] = bad
        with pytest.raises(ValidationError) as exc:
            CreditApplication(**with_field(field, values))
        assert exc.value.errors()[0]["type"] == "finite_number"


# =============================================================================
# The two custom validators
# =============================================================================
class TestCustomValidators:
    @pytest.mark.parametrize("bad_code", [-3, 9, 99, -100])
    def test_pay_status_outside_minus2_to_8(self, bad_code):
        statuses = [0, 0, 0, 0, 0, 0]
        statuses[3] = bad_code
        with pytest.raises(ValidationError) as exc:
            CreditApplication(**with_field("pay_status", statuses))
        error = exc.value.errors()[0]
        assert error["loc"] == ("pay_status",)
        # The message points at the month (index 3 is month t-4).
        assert "t-4" in error["msg"]
        assert "between -2 and 8" in error["msg"]

    def test_field_bounds_alone_would_not_catch_it(self):
        """Every code is a valid int; only the validator knows 99 is meaningless."""
        with pytest.raises(ValidationError):
            CreditApplication(**with_field("pay_status", [99, 0, 0, 0, 0, 0]))

    @pytest.mark.parametrize("bad_amount", [-0.01, -1, -10_000])
    def test_negative_payment(self, bad_amount):
        amounts = [100.0] * 6
        amounts[5] = bad_amount
        with pytest.raises(ValidationError) as exc:
            CreditApplication(**with_field("pay_amt", amounts))
        error = exc.value.errors()[0]
        assert error["loc"] == ("pay_amt",)
        assert "t-6" in error["msg"]
        assert "cannot be negative" in error["msg"]

    def test_both_validators_report_together(self):
        """A caller fixing their payload sees every problem in one round trip."""
        payload = with_field("pay_status", [9, 0, 0, 0, 0, 0])
        payload["pay_amt"] = [-1, 0, 0, 0, 0, 0]
        with pytest.raises(ValidationError) as exc:
            CreditApplication(**payload)
        assert error_fields(exc.value) == {"pay_status", "pay_amt"}


# =============================================================================
# Response and health schemas
# =============================================================================
class TestPredictionResponse:
    def test_valid(self):
        assert PredictionResponse(**VALID_RESPONSE).model_dump() == VALID_RESPONSE

    @pytest.mark.parametrize(
        "field,value",
        [
            ("default_probability", -0.01),
            ("default_probability", 1.01),
            ("risk_band", "VERY_HIGH"),
            ("decision", "MAYBE"),
            ("model_version", None),
        ],
    )
    def test_rejects_values_outside_contract(self, field, value):
        with pytest.raises(ValidationError):
            PredictionResponse(**with_field(field, value, base=VALID_RESPONSE))

    def test_has_exactly_six_fields(self):
        assert set(PredictionResponse.model_fields) == set(VALID_RESPONSE)


class TestHealthResponse:
    def test_valid(self):
        health = HealthResponse(status="healthy", model_loaded=True, model_version="1.0.0")
        assert health.model_dump() == {
            "status": "healthy",
            "model_loaded": True,
            "model_version": "1.0.0",
        }

    def test_status_is_a_closed_set(self):
        with pytest.raises(ValidationError):
            HealthResponse(status="ok", model_loaded=True, model_version="1.0.0")


class TestBatchRequest:
    @pytest.mark.parametrize("size", [1, 500])
    def test_accepts_1_to_500(self, size):
        assert len(BatchPredictionRequest(applications=[VALID] * size).applications) == size

    @pytest.mark.parametrize("size", [0, 501])
    def test_rejects_outside_1_to_500(self, size):
        with pytest.raises(ValidationError):
            BatchPredictionRequest(applications=[VALID] * size)
