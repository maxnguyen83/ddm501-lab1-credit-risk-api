"""
Pydantic schemas for request/response validation.

The schema is the API's contract. Everything the model assumes about its input
is stated here, so a malformed request fails at the edge with a clear 422
instead of producing a confident-looking wrong score.
"""

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Domain of the repayment-status codes in the UCI dataset:
# -2 = no consumption, -1 = paid in full, 0 = revolving credit, 1..8 = months late.
PAY_STATUS_MIN = -2
PAY_STATUS_MAX = 8

# The six monthly values, months t-1 .. t-6, always come as a group.
N_MONTHS = 6


# =============================================================================
# Request schema
# =============================================================================
class CreditApplication(BaseModel):
    """One applicant, as the core banking system sends it."""

    # NaN / Infinity are valid JSON for Python's parser and would slip past the
    # numeric bounds of the list fields; the gradient boosting model would then
    # score them silently as "missing". Reject them at the edge instead.
    model_config = ConfigDict(
        allow_inf_nan=False,
        json_schema_extra={
            "examples": [
                {
                    "limit_bal": 300000,
                    "sex": 2,
                    "education": 1,
                    "marriage": 2,
                    "age": 38,
                    "pay_status": [-1, -1, -1, -1, -1, -1],
                    "bill_amt": [12000, 11500, 11000, 10500, 10000, 9500],
                    "pay_amt": [12000, 11500, 11000, 10500, 10000, 9500],
                }
            ]
        },
    )

    limit_bal: float = Field(
        ..., gt=0, le=2_000_000, description="Credit limit in NT dollars", examples=[120000]
    )
    sex: Literal[1, 2] = Field(..., description="1 = male, 2 = female", examples=[2])
    education: Literal[1, 2, 3, 4] = Field(
        ..., description="1 = graduate school, 2 = university, 3 = high school, 4 = others",
        examples=[2],
    )
    marriage: Literal[1, 2, 3] = Field(
        ..., description="1 = married, 2 = single, 3 = others", examples=[2]
    )
    age: int = Field(..., ge=18, le=100, description="Age in years", examples=[34])
    pay_status: List[int] = Field(
        ...,
        min_length=N_MONTHS,
        max_length=N_MONTHS,
        description=(
            "Repayment status for months t-1 .. t-6. "
            "-2 = no consumption, -1 = paid in full, 0 = revolving credit, "
            "1..8 = months of payment delay."
        ),
        examples=[[0, 0, 0, 0, 0, 0]],
    )
    # No lower bound on purpose: a negative statement means the customer is in
    # credit (overpaid), which the UCI data contains.
    bill_amt: List[float] = Field(
        ...,
        min_length=N_MONTHS,
        max_length=N_MONTHS,
        description="Bill statement amount for months t-1 .. t-6, in NT dollars.",
        examples=[[20000, 19000, 18000, 17000, 16000, 15000]],
    )
    pay_amt: List[float] = Field(
        ...,
        min_length=N_MONTHS,
        max_length=N_MONTHS,
        description="Amount actually paid for months t-1 .. t-6, in NT dollars. Never negative.",
        examples=[[2000, 2000, 1500, 1500, 1000, 1000]],
    )

    # -------------------------------------------------------------------------
    # Custom validators: rules on every element of a list, which Field(...)
    # bounds cannot express (they apply to the list, not to its items).
    # -------------------------------------------------------------------------
    @field_validator("pay_status")
    @classmethod
    def pay_status_in_domain(cls, values: List[int]) -> List[int]:
        """Every repayment-status code must be one the model was trained on."""
        for month, code in enumerate(values, start=1):
            if not PAY_STATUS_MIN <= code <= PAY_STATUS_MAX:
                raise ValueError(
                    f"pay_status for month t-{month} must be between "
                    f"{PAY_STATUS_MIN} and {PAY_STATUS_MAX}"
                )
        return values

    @field_validator("pay_amt")
    @classmethod
    def pay_amt_not_negative(cls, values: List[float]) -> List[float]:
        """A payment is money received; it cannot be negative."""
        for month, amount in enumerate(values, start=1):
            if amount < 0:
                raise ValueError(f"pay_amt for month t-{month} cannot be negative")
        return values


# =============================================================================
# Response schemas
# =============================================================================
class PredictionResponse(BaseModel):
    """Scoring result plus the decision derived from it."""

    # Fields starting with "model_" clash with Pydantic's protected namespace
    # and raise a warning at import time; the names are part of the contract,
    # so switch the check off rather than rename them.
    model_config = ConfigDict(protected_namespaces=())

    default_probability: float = Field(
        ..., ge=0.0, le=1.0, description="Probability of default next month, 4 d.p.",
        examples=[0.1234],
    )
    risk_band: Literal["LOW", "MEDIUM", "HIGH"] = Field(
        ..., description="Band implied by the two thresholds", examples=["LOW"]
    )
    decision: Literal["APPROVE", "REVIEW", "DECLINE"] = Field(
        ..., description="Underwriting action for the band", examples=["APPROVE"]
    )
    review_threshold: float = Field(
        ..., description="Review threshold in force when this was scored", examples=[0.30]
    )
    decline_threshold: float = Field(
        ..., description="Decline threshold in force when this was scored", examples=[0.60]
    )
    model_version: str = Field(
        ..., description="Version of the model that scored it", examples=["1.0.0"]
    )


class HealthResponse(BaseModel):
    """Liveness and readiness of the service."""

    model_config = ConfigDict(protected_namespaces=())

    status: Literal["healthy", "unhealthy"] = Field(..., examples=["healthy"])
    model_loaded: bool = Field(..., description="True once the model can serve", examples=[True])
    model_version: str = Field(..., examples=["1.0.0"])


# =============================================================================
# Batch schemas (PROVIDED — used by /predict/batch in main.py)
# =============================================================================
class BatchPredictionRequest(BaseModel):
    """Up to 500 applications scored in one call."""

    applications: List[CreditApplication] = Field(..., min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    """Results in the same order as the request."""

    predictions: List[PredictionResponse]
    total_count: int
