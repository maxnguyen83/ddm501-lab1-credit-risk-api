# Credit Default Risk Scoring API

DDM501 Lab 1. The data science team handed over a trained model that estimates
the probability that a credit card customer misses their next payment. This
repository turns it into a service that the loan origination system can call:
a FastAPI app with a strict input contract, a container image, a Compose file
and a test suite that runs in CI.

| Deliverable | Where |
|---|---|
| REST API: `/health`, `/predict`, `/predict/batch`, `/model/info` | `app/main.py` |
| Contract that rejects malformed input with a 422 | `app/schemas.py` |
| Model wrapper: loading, feature frame, decision rule | `app/model.py` |
| Container and Compose | `Dockerfile`, `.dockerignore`, `docker-compose.yml` |
| Tests (166, 99% coverage) | `tests/` |
| CI smoke test | `.github/workflows/smoke.yml` |
| Swagger docs | `http://localhost:8000/docs` once running |

## Contents

1. [Quick start](#1-quick-start)
2. [Run with Docker Compose](#2-run-with-docker-compose)
3. [Decision thresholds](#3-decision-thresholds)
4. [The request contract](#4-the-request-contract)
5. [API examples with real responses](#5-api-examples-with-real-responses)
6. [Tests and CI](#6-tests-and-ci)
7. [Implementation notes per task](#7-implementation-notes-per-task)
8. [Design decisions](#8-design-decisions)
9. [Measured numbers](#9-measured-numbers)
10. [Known limitations](#10-known-limitations)
11. [Team](#11-team)

---

## 1. Quick start

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/train_model.py          # ~30 s, writes models/credit_model.joblib
pytest tests/ -v --cov=app --cov-report=term-missing
uvicorn app.main:app --reload          # http://localhost:8000/docs
```

Training on the shipped dataset (`data/credit_default.csv`, 30,000 rows x 24
columns, generated with the UCI schema so the lab runs offline) printed:

```
ROC AUC : 0.7481
PR  AUC : 0.5461   (baseline = 0.2328)
At threshold 0.3:  TN 4,088  FP 515  FN 726  TP 671
```

Python versions: the Docker image and CI use Python 3.11 (`python:3.11-slim`,
`setup-python 3.11`). We also developed on Python 3.12; every pin in
`requirements.txt` has a 3.12 wheel, so no pin had to change. The model file is
a pickle, so it must be trained with the same scikit-learn version the image
installs (1.6.0, pinned) — a model trained on the host with 3.12 loads fine in
the 3.11 container because the scikit-learn version is identical.

## 2. Run with Docker Compose

```bash
python scripts/train_model.py          # the model is mounted, not built in
docker compose up --build -d
docker inspect --format='{{.State.Health.Status}}' credit-risk-api   # -> healthy
curl http://localhost:8000/health
docker compose down
```

What the Compose file sets up (`docker-compose.yml`):

| Setting | Value | Why |
|---|---|---|
| `ports` | `8000:8000` | same port as local `uvicorn` |
| `volumes` | `./models:/app/models:ro` | model comes from the host, read-only |
| `MODEL_PATH` | `/app/models/credit_model.joblib` | where the volume puts it |
| `MODEL_VERSION` | `${MODEL_VERSION:-1.0.0}` | reported in every response |
| `REVIEW_THRESHOLD` | `${REVIEW_THRESHOLD:-0.30}` | business setting, see section 3 |
| `DECLINE_THRESHOLD` | `${DECLINE_THRESHOLD:-0.60}` | business setting, see section 3 |
| `restart` | `unless-stopped` | survives a crash or a host reboot |

The defaults are exactly the values the lab asks for; the `${VAR:-default}` form
only means operations can change a threshold from the shell or an `.env` file
without editing the file or rebuilding the image (example in section 5).

If port 8000 is already taken on your machine, do not edit the committed file;
put a local override next to it instead:

```yaml
# compose.local.yml (not committed)
services:
  api:
    ports: !override
      - "28000:8000"
```

```bash
docker compose -f docker-compose.yml -f compose.local.yml up --build -d
```

## 3. Decision thresholds

The model returns a probability. The service turns it into one of three
underwriting actions:

| Probability | Risk band | Decision | Meaning |
|---|---|---|---|
| `p < 0.30` | `LOW` | `APPROVE` | approved automatically |
| `0.30 <= p < 0.60` | `MEDIUM` | `REVIEW` | a human underwriter looks |
| `p >= 0.60` | `HIGH` | `DECLINE` | declined automatically |

- Both thresholds live in `app/config.py` and are read from the environment
  variables `REVIEW_THRESHOLD` and `DECLINE_THRESHOLD` (defaults 0.30 / 0.60).
  They are business settings, not model properties, so changing them needs a
  restart, not a retrain or a rebuild.
- The rule checks the **decline** threshold first (`CreditRiskModel.decide`).
  Checked the other way round, every score of 0.30 or more would stop at
  REVIEW and nothing would ever be declined.
- Both boundaries are inclusive on the upper band: exactly 0.30 is REVIEW,
  exactly 0.60 is DECLINE (tested in `tests/test_decision.py::TestDecide`).
- The decision is made on the probability **after** rounding to 4 decimals,
  i.e. on the number the caller sees. A raw score of 0.29996 is returned as
  `0.3`, so it is decided as REVIEW; deciding on the raw value would return
  `0.3` next to `APPROVE`. `score` and `score_batch` use the same rule, so
  `/predict` and `/predict/batch` can never disagree at a boundary.
- Every response carries the thresholds that were in force when it was
  scored, so a decision can be audited later even after the thresholds change.

## 4. The request contract

`CreditApplication` in `app/schemas.py`. Anything outside it gets a 422 and
never reaches the model.

| Field | Type | Rule |
|---|---|---|
| `limit_bal` | float | `0 < limit_bal <= 2,000,000` (NT dollars) |
| `sex` | int | one of `1` (male), `2` (female) |
| `education` | int | one of `1`..`4` |
| `marriage` | int | one of `1` (married), `2` (single), `3` (others) |
| `age` | int | `18 <= age <= 100` |
| `pay_status` | list[int] | exactly 6 values, months t-1..t-6; **custom validator**: each value in `-2..8` |
| `bill_amt` | list[float] | exactly 6 values; may be negative (customer in credit, as in the UCI data) |
| `pay_amt` | list[float] | exactly 6 values; **custom validator**: no value may be negative |

Two more rules apply to the whole model:

- `NaN` and `Infinity` are rejected (`allow_inf_nan=False`). Python's JSON
  parser accepts them, they pass the numeric bounds of the list fields, and the
  gradient boosting model would silently score them as "missing".
- The 422 body keeps FastAPI's usual `{"detail": [...]}` shape with `type`,
  `loc` and `msg` for every error, but does not echo the rejected value back
  (see design decision 5).

`pay_status[0..5]` maps to the training columns `PAY_0, PAY_2, ..., PAY_6`.
There is no `PAY_1`: the original UCI file skips it and the model was trained
on that naming, so `to_frame` reproduces it on purpose.

## 5. API examples with real responses

All responses below were captured from the container started with
`docker compose up --build` (host port remapped to 28000 on our machine,
shown here as 8000).

**Health**

```bash
curl http://localhost:8000/health
```
```json
{"status":"healthy","model_loaded":true,"model_version":"1.0.0"}
```

**Model info**

```bash
curl http://localhost:8000/model/info
```
```json
{"model_version":"1.0.0","model_type":"HistGradientBoostingClassifier",
 "trained_at":"2026-10-03T02:12:18+00:00",
 "metrics":{"roc_auc":0.7481,"pr_auc":0.5461,"true_positives":671,"false_positives":515,
            "false_negatives":726,"true_negatives":4088,"threshold":0.3},
 "review_threshold":0.3,"decline_threshold":0.6,"is_loaded":true}
```

**Score a good applicant** (always paid in full, low utilisation)

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"limit_bal":300000,"sex":2,"education":1,"marriage":2,"age":38,
       "pay_status":[-1,-1,-1,-1,-1,-1],
       "bill_amt":[12000,11500,11000,10500,10000,9500],
       "pay_amt":[12000,11500,11000,10500,10000,9500]}'
```
```json
{"default_probability":0.0637,"risk_band":"LOW","decision":"APPROVE",
 "review_threshold":0.3,"decline_threshold":0.6,"model_version":"1.0.0"}
```

**A borderline applicant** (one month late last month, high utilisation)

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"limit_bal":80000,"sex":1,"education":2,"marriage":2,"age":30,
       "pay_status":[1,0,0,0,0,0],
       "bill_amt":[45000,44000,43000,42000,41000,40000],
       "pay_amt":[4000,4000,4000,4000,4000,4000]}'
```
```json
{"default_probability":0.5158,"risk_band":"MEDIUM","decision":"REVIEW",
 "review_threshold":0.3,"decline_threshold":0.6,"model_version":"1.0.0"}
```

**A risky applicant** (months behind, near the limit, paying almost nothing)

```bash
curl -X POST http://localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"limit_bal":20000,"sex":1,"education":3,"marriage":1,"age":24,
       "pay_status":[4,3,3,2,2,2],
       "bill_amt":[19800,19500,19000,18500,18000,17500],
       "pay_amt":[0,0,200,0,300,0]}'
```
```json
{"default_probability":0.8995,"risk_band":"HIGH","decision":"DECLINE",
 "review_threshold":0.3,"decline_threshold":0.6,"model_version":"1.0.0"}
```

**Batch** (up to 500 applications, results in request order)

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H 'Content-Type: application/json' \
  -d '{"applications":[<good>, <borderline>, <risky>]}'
```
```json
{"predictions":[
   {"default_probability":0.0637,"risk_band":"LOW","decision":"APPROVE","review_threshold":0.3,"decline_threshold":0.6,"model_version":"1.0.0"},
   {"default_probability":0.5158,"risk_band":"MEDIUM","decision":"REVIEW","review_threshold":0.3,"decline_threshold":0.6,"model_version":"1.0.0"},
   {"default_probability":0.8995,"risk_band":"HIGH","decision":"DECLINE","review_threshold":0.3,"decline_threshold":0.6,"model_version":"1.0.0"}],
 "total_count":3}
```

**Invalid input** (`pay_status` code 99, a negative payment) -> HTTP 422

```json
{"detail":[
  {"type":"value_error","loc":["body","pay_status"],"msg":"Value error, pay_status for month t-1 must be between -2 and 8"},
  {"type":"value_error","loc":["body","pay_amt"],"msg":"Value error, pay_amt for month t-6 cannot be negative"}]}
```

**No model mounted** (`docker run` of the same image without the volume)

```
GET  /health  -> 200 {"status":"unhealthy","model_loaded":false,"model_version":"1.0.0"}
POST /predict -> 503 {"detail":"Model is not loaded; the service cannot score yet"}
docker inspect ... -> unhealthy
```

**Changing a threshold without a rebuild**

```bash
REVIEW_THRESHOLD=0.55 docker compose up -d     # recreates the container only
# the borderline applicant above, same score, new decision:
{"default_probability":0.5158,"risk_band":"LOW","decision":"APPROVE",
 "review_threshold":0.55,"decline_threshold":0.6,"model_version":"1.0.0"}
```

## 6. Tests and CI

```bash
pytest tests/ -v --cov=app --cov-report=term-missing
```

```
Name              Stmts   Miss  Cover   Missing
-----------------------------------------------
app/__init__.py       1      0   100%
app/config.py        13      0   100%
app/main.py          68      2    97%   194-196
app/model.py         74      0   100%
app/schemas.py       47      0   100%
-----------------------------------------------
TOTAL               203      2    99%
============================= 166 passed in 2.60s =============================
```

The two uncovered lines are the `if __name__ == "__main__"` block of `main.py`.

| File | Tests | What it covers |
|---|---|---|
| `tests/test_api.py` | 44 | Every endpoint through `TestClient`: happy path, exact response fields, decision consistent with the thresholds, determinism, behaviour (risky > good, more delay never lowers risk), the instructor's validation cases (all 422), 422 without echoed input, NaN and malformed JSON, batch size / order / parity with `/predict` / one model call for 500 rows, and the failure modes (503 without a model, 500 when scoring breaks, 422 still wins when the model is down, startup survives a missing model) |
| `tests/test_validation.py` | 24 | The contract at the HTTP edge beyond the core cases: boundary values that must still be scored (age 18/100, codes -2/8, zero payments, negative bill), wrong types, seven months, both validators in one response, oversized batch, one bad item in a batch, batch without its wrapper |
| `tests/test_schemas.py` | 70 | The contract on its own, no HTTP: inclusive bounds, every field required, list lengths 0/5/7 rejected, NaN/Infinity rejected, both custom validators with their messages, the Swagger example is itself valid, response and health schemas are closed sets, batch size 1..500 |
| `tests/test_model.py` | 16 | Loading (missing file is logged with the fix and re-raised, model trained on other columns is refused), `to_frame` column order and month mapping, a training CSV row rebuilt from its API form is identical, rounding edge (0.29996 -> 0.3 -> REVIEW in both paths), `score == score_batch` |
| `tests/test_decision.py` | 12 | `decide` at every boundary, comparison order, thresholds from the environment and their defaults |

Two tests deserve a note:

- `test_batch_calls_the_model_once` wraps `predict_proba` and checks that 500
  applications produce exactly one call with 500 rows. It fails if someone
  "simplifies" the batch endpoint into a loop over `/predict`.
- `test_round_trips_a_training_row` reads rows from `data/credit_default.csv`,
  turns them into API payloads and back through `to_frame`, and asserts the
  frame equals the training columns. This is the training-serving skew check.

**CI** (`.github/workflows/smoke.yml`, runs on push to `main`/`develop` and on
pull requests to `main`): install with pip, check every dependency is a wheel,
check the dataset, train, run the tests with coverage, validate the Compose
file, build the image, run it with the model mounted read-only, call
`/health`, `/predict`, `/predict/batch`, `/model/info` and an invalid request
(must be 422), then check the container runs as uid 1000 and that Docker's own
HEALTHCHECK reaches `healthy`. Container logs are printed if any step fails.
The workflow passes `actionlint`.

## 7. Implementation notes per task

**Task 1 — `app/schemas.py`.** `marriage` is `Literal[1, 2, 3]` and `age` is
`int` in `[18, 100]`; the three monthly lists use `min_length = max_length = 6`.
The element rules use `@field_validator` because `Field(ge=..., le=...)`
applies to the list itself, not to its items. Each validator reports which
month is wrong (`t-1`..`t-6`). `PredictionResponse` and `HealthResponse` use
`Literal` for the band, decision and status, so the response can only contain
values the consumers know. Both set `protected_namespaces=()` because Pydantic
v2 warns about fields starting with `model_`; the names are part of the
contract, so we kept them. The model-level example makes Swagger's "Try it out"
work with one click.

**Task 2 — `app/model.py`.** `_load_model` loads the `{"pipeline", "metadata"}`
bundle, logs the path, model type and training time, and on
`FileNotFoundError` logs how to fix it (run `scripts/train_model.py` or set
`MODEL_PATH`) before re-raising. It also compares `metadata["features"]` with
`FEATURE_COLUMNS` and refuses to load a model trained on a different column
list. `to_frame` builds one row per application and passes
`columns=FEATURE_COLUMNS` so the order is fixed by the list, not by how the
dict was built. `decide` checks the highest threshold first. `score` rounds to
4 decimals, then decides; `score_batch` was changed in one line to do the same.

**Task 3 — `app/main.py`.** Both endpoints share a `_require_model()` guard that
raises 503 when the model is not loaded. `/predict` calls
`model.score(application.model_dump())`; `/predict/batch` dumps every
application and calls `score_batch` once. Any other exception is logged with
its stack trace and returned as a generic 500 (no internals, no applicant data
in the body). The 500/503 responses are declared on the routes so they show in
Swagger. A `RequestValidationError` handler returns the 422 without echoing
input (design decision 5).

**Task 4 — `Dockerfile`.** `python:3.11-slim`; `PYTHONDONTWRITEBYTECODE`,
`PYTHONUNBUFFERED`, `PYTHONPATH=/app`; `requirements.txt` copied and installed
with `--no-cache-dir` before the source; `app/`, `scripts/`, `data/` copied
(not `models/`); user `appuser` (uid 1000) owns `/app` and `/app/models`;
`EXPOSE 8000`; HEALTHCHECK every 30 s, timeout 10 s, start period 10 s,
3 retries; `CMD` in exec form on `0.0.0.0:8000`. `.dockerignore` keeps `.venv`,
`models/`, `tests/` and git data out of the build context.

**Task 5 — `docker-compose.yml`.** One `api` service built from `.`, container
name `credit-risk-api`, port 8000, `./models` mounted read-only, the four
environment variables, `restart: unless-stopped`.

## 8. Design decisions

**1. Validation at the edge with Pydantic, not inside the model.**
The model will happily score anything numeric: `pay_status = 99`, `age = 12`, a
negative payment. It returns a confident probability and nobody notices. By
putting every assumption into the schema, FastAPI rejects bad input with a 422
before the handler runs, the caller gets the field and the reason, and the
model code only ever sees valid data. The alternative, checking inside
`score()`, mixes business rules with ML code, returns 500s instead of 422s,
and does not appear in the Swagger docs; the schema is also the documentation
the loan origination team reads. Test `test_validation_still_wins_without_model`
shows the order: even with the model down, a bad request is a 422, not a 503.

**2. The model is a read-only volume, not baked into the image.**
The image holds code and dependencies, which change with a release; the model
changes on its own schedule (retraining). Mounting `./models` means a new model
is rolled out by replacing one file and restarting, and the same image digest
runs in staging and production with different models if needed. Baking the
model in (`COPY models/`) would give a self-contained image, which is simpler to
ship, but every retrain would need a rebuild and a new image, and a large
artifact would bloat every layer push. `:ro` means a compromised or buggy
process cannot overwrite the artifact it serves (we checked:
`touch /app/models/x` fails with "Read-only file system"). The cost is that the
container is useless without the volume — which is exactly what the
HEALTHCHECK reports.

**3. Non-root user.**
If someone finds a remote code execution bug in a dependency, a process running
as root inside the container can rewrite system files and installed packages,
and any container escape lands on the host as root. `appuser` (uid 1000) cannot
modify `/usr/local` (where pip installed everything) or other system paths, and
cannot write to the model because the mount is read-only. The alternative,
dropping privileges only at the orchestrator level (`docker run --user`), is
easy to forget; putting `USER` in the image makes it the default everywhere
the image runs. The model file is created with mode 644, so it is readable by
uid 1000 without any permission changes. `docker exec credit-risk-api id`
prints `uid=1000(appuser)`, and CI asserts it.

**4. The HEALTHCHECK checks `model_loaded`, not just "the process answers".**
On startup the lifespan hook catches a failed model load and keeps the process
up, so `/health` returns HTTP 200 even when the model is missing. A check like
`curl -f /health` or a TCP probe would therefore report `healthy` for a
container that answers every `/predict` with 503. Parsing the JSON and
requiring `model_loaded == true` makes Docker (and any orchestrator or load
balancer that reads the status) see the real readiness. We verified both
sides: with the volume the container was `healthy` 5-10 s after start (two
runs); without it the same image failed every probe and was reported
`unhealthy` within two minutes, while `/health` still returned 200. The
alternative, letting the process crash when the model is missing, would also
be visible, but as a restart loop with no way to ask the service what is wrong.
The check uses Python's `urllib`, already in the image, instead of installing
`curl` (one less package and one less binary in the attack surface).

**5. 503 vs 500 vs 422, and no echo in the 422.**
503 means "the request was fine, the service is not ready" (retry elsewhere,
page whoever deploys models); 500 means "scoring broke" (page the service
owner); 422 means "fix your request". FastAPI's default 422 also echoes the
rejected value under `input`. We drop it: it copies applicant data into client
and gateway logs, and a `NaN` in the body could not be encoded as JSON, which
turned the 422 into a 500 (found by `test_nan_is_rejected`).

**6. One model call per batch.**
Measured on our machine: 500 applications through `/predict/batch` take about
10 ms; the same 500 through `/predict` one by one take about 2.2 s. The batch
endpoint also returns results in request order and gives exactly the same
numbers as `/predict` (tested).

## 9. Measured numbers

All on a MacBook (Apple Silicon), single uvicorn worker, model above.

| What | Value |
|---|---|
| ROC AUC / PR AUC on the 6,000-row test split | 0.7481 / 0.5461 |
| `/predict` latency, 300 sequential calls (local uvicorn) | p50 4.4 ms, p95 5.8 ms |
| `/predict/batch` with 100 / 500 applications | p50 6.0 ms / 10.2 ms |
| 500 applications via `/predict` one by one | ~2,200 ms |
| Model artifact | 218 KB |
| Image `docker image ls` size | 782 MB |
| Time to `healthy` after `docker compose up` | 5-10 s (two runs) |
| Test suite | 166 tests, 99% coverage, ~3 s |

## 10. Known limitations

- The image is 782 MB, mostly scikit-learn, SciPy, pandas and NumPy. It also
  contains pytest and httpx because the lab uses one `requirements.txt`; a
  `requirements-dev.txt` split or a multi-stage build would trim it.
- `REVIEW_THRESHOLD < DECLINE_THRESHOLD` is not enforced. A misconfiguration
  like 0.7 / 0.6 would silently remove the REVIEW band. A startup check in
  `config.py` would be the next step.
- The endpoints are `async def` and call scikit-learn directly, which blocks the
  event loop for the few milliseconds a prediction takes. Fine at this scale; with more
  traffic we would make them plain `def` (FastAPI runs those in a thread pool)
  or run several workers.
- No authentication and `CORS allow_origins=["*"]` (from the starter). Inside
  the bank network this would sit behind the API gateway.
- The dataset is generated with the UCI schema so the lab runs offline;
  `scripts/download_data.py` swaps in the real UCI file without code changes.

## 11. Team

| Member | Part |
|---|---|
| maxnguyen83 | Task 1: `app/schemas.py` (contract, two validators), `tests/test_schemas.py`, `tests/test_validation.py` |
| Ducmanh2212 | Task 2: `app/model.py` (load, `to_frame`, `decide`, `score`), `tests/test_model.py`, `tests/test_decision.py` |
| hieunt-fsb-ai | Task 3: `app/main.py` (endpoints), `tests/test_api.py`, this README |
| thientd2609 | Tasks 4-5: `Dockerfile`, `.dockerignore`, `docker-compose.yml`, `.github/workflows/smoke.yml` |
