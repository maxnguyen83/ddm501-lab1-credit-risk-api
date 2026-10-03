"""
Pydantic schemas for request/response validation.

The schema is the API's contract. Everything the model assumes about its input
is stated here, so a malformed request fails at the edge with a clear 422
instead of producing a confident-looking wrong score.

TODO: Complete the schema definitions below.
"""

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# TODO 1: Complete the CreditApplication request schema
# =============================================================================
# The first three fields are done for you as a worked example. Fill in the rest.
#
# Requirements:
#   marriage    1 = married, 2 = single, 3 = others
#   age         integer, 18 to 100 inclusive
#   pay_status  list of exactly 6 integers, months t-1 .. t-6
#   bill_amt    list of exactly 6 floats, months t-1 .. t-6
#   pay_amt     list of exactly 6 floats, months t-1 .. t-6
#

class CreditApplication(BaseModel):
    """One applicant, as the core banking system sends it."""

    limit_bal: float = Field(
        ..., gt=0, le=2_000_000, description="Credit limit in NT dollars", examples=[120000]
    )
    sex: Literal[1, 2] = Field(..., description="1 = male, 2 = female", examples=[2])
    education: Literal[1, 2, 3, 4] = Field(
        ..., description="1 = graduate school, 2 = university, 3 = high school, 4 = others",
        examples=[2],
    )

    # TODO 1a: marriage
    # marriage: Literal[???] = Field(..., description="...", examples=[2])

    # TODO 1b: age
    # age: int = Field(..., ge=???, le=???, description="Age in years", examples=[34])

    # TODO 1c: pay_status
    # pay_status: List[int] = Field(
    #     ...,
    #     min_length=???,
    #     max_length=???,
    #     description=(
    #         "Repayment status for months t-1 .. t-6. "
    #         "-2 = no consumption, -1 = paid in full, 0 = revolving credit, "
    #         "1..8 = months of payment delay."
    #     ),
    #     examples=[[0, 0, 0, 0, 0, 0]],
    # )

    # TODO 1d: bill_amt

    # TODO 1e: pay_amt

    # =========================================================================
    # TODO 2: Add the two custom validators
    # =========================================================================
    # Field(...) covers types and ranges. Some rules need real code:
    #
    #   2a. every value in pay_status must be between -2 and 8
    #   2b. no value in pay_amt may be negative
    #


# =============================================================================
# TODO 3: Complete the PredictionResponse schema
# =============================================================================
# Requirements:
#   default_probability  float between 0.0 and 1.0
#   risk_band            one of "LOW", "MEDIUM", "HIGH"
#   decision             one of "APPROVE", "REVIEW", "DECLINE"
#   review_threshold     float — the threshold in force when this was scored
#   decline_threshold    float — likewise
#   model_version        string
#

class PredictionResponse(BaseModel):
    """Scoring result plus the decision derived from it."""

    # TODO: define the six fields
    pass


# =============================================================================
# TODO 4: Complete the HealthResponse schema
# =============================================================================
# Requirements:
#   status         "healthy" or "unhealthy"
#   model_loaded   boolean
#   model_version  string

class HealthResponse(BaseModel):
    """Liveness and readiness of the service."""

    # TODO: define the three fields
    pass


# =============================================================================
# Batch schemas (PROVIDED — read them, you will need them for TODO 2 in main.py)
# =============================================================================
class BatchPredictionRequest(BaseModel):
    """Up to 500 applications scored in one call."""

    applications: List[CreditApplication] = Field(..., min_length=1, max_length=500)


class BatchPredictionResponse(BaseModel):
    """Results in the same order as the request."""

    predictions: List[PredictionResponse]
    total_count: int
