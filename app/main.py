"""
FastAPI application for credit default risk scoring.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import (
    API_DESCRIPTION,
    API_TITLE,
    API_VERSION,
    DECLINE_THRESHOLD,
    MODEL_VERSION,
    REVIEW_THRESHOLD,
)
from app.model import CreditRiskModel
from app.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    CreditApplication,
    HealthResponse,
    PredictionResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

model: CreditRiskModel | None = None


# =============================================================================
# Lifespan: load the model once, at startup
# =============================================================================
# Loading a model takes time. Doing it per-request would add that cost to every
# call; doing it at import time would break the test client. The lifespan hook
# is the right place — it runs once when the process starts.
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model when the process starts, release it when it stops."""
    global model
    try:
        model = CreditRiskModel()
        logger.info("Model loaded at startup")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load model: %s", exc)
        model = None
    yield
    model = None


app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Validation errors
# =============================================================================
# FastAPI's default 422 echoes every rejected value back under "input". That
# repeats applicant data into client and gateway logs, and a NaN in the body
# cannot be encoded as JSON at all, which turns the 422 into a 500. Keep the
# usual {"detail": [...]} shape, with only where (loc), what (msg) and the
# machine-readable error type.
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Return 422 with field locations and messages, without the rejected values."""
    errors = [
        {"type": error["type"], "loc": error["loc"], "msg": error["msg"]}
        for error in exc.errors()
    ]
    return JSONResponse(status_code=422, content={"detail": jsonable_encoder(errors)})


# =============================================================================
# Health (PROVIDED — do not modify)
# =============================================================================
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Report whether the service can serve predictions right now."""
    ready = model is not None and model.is_loaded()
    return HealthResponse(
        status="healthy" if ready else "unhealthy",
        model_loaded=ready,
        model_version=MODEL_VERSION,
    )


# =============================================================================
# Prediction endpoints
# =============================================================================
# 503 vs 500: a missing model means the request was fine but the service is not
# ready (retry elsewhere, page whoever deploys models); a 500 means something
# broke while scoring. Validation errors never reach this code — FastAPI turns
# them into a 422 from the schemas before the handler runs.
MODEL_NOT_READY = "Model is not loaded; the service cannot score yet"

# Documented in Swagger next to the automatic 422.
ERROR_RESPONSES = {
    500: {"description": "Unexpected error while scoring"},
    503: {"description": "Model not loaded; the service is not ready"},
}


def _require_model() -> CreditRiskModel:
    """Return the loaded model, or raise 503 if the service is not ready."""
    if model is None or not model.is_loaded():
        raise HTTPException(status_code=503, detail=MODEL_NOT_READY)
    return model


@app.post(
    "/predict", response_model=PredictionResponse, responses=ERROR_RESPONSES, tags=["Prediction"]
)
async def predict(application: CreditApplication):
    """Score one applicant and return the underwriting decision."""
    scorer = _require_model()
    try:
        result = scorer.score(application.model_dump())
    except Exception as exc:  # noqa: BLE001
        # Log the stack trace for the on-call engineer, but do not echo
        # internals (or applicant data) back to the caller.
        logger.exception("Scoring failed for /predict")
        raise HTTPException(status_code=500, detail="Internal error while scoring") from exc
    return PredictionResponse(**result)


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses=ERROR_RESPONSES,
    tags=["Prediction"],
)
async def predict_batch(request: BatchPredictionRequest):
    """Score up to 500 applicants in one call."""
    scorer = _require_model()
    payloads = [application.model_dump() for application in request.applications]
    try:
        # One vectorised call for the whole list, never one call per item.
        results = scorer.score_batch(payloads)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Scoring failed for /predict/batch (%d applications)", len(payloads))
        raise HTTPException(status_code=500, detail="Internal error while scoring") from exc
    predictions = [PredictionResponse(**result) for result in results]
    return BatchPredictionResponse(predictions=predictions, total_count=len(predictions))


# =============================================================================
# Info endpoints
# =============================================================================
@app.get("/", tags=["Info"])
async def root():
    """API metadata and where to find the docs."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/model/info", tags=["Info"])
async def model_info():
    """Model version, training metrics and the thresholds in force."""
    return {
        "model_version": MODEL_VERSION,
        "model_type": (model.metadata.get("model_type") if model else None),
        "trained_at": (model.metadata.get("trained_at") if model else None),
        "metrics": (model.metadata.get("metrics") if model else None),
        "review_threshold": REVIEW_THRESHOLD,
        "decline_threshold": DECLINE_THRESHOLD,
        "is_loaded": model is not None and model.is_loaded(),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
