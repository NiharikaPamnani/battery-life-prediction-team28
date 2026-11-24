"""
app/main.py -- Battery prediction API.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

GET  /health          -> {"status": "ok", "team": 28, "params": 1281, "val_mae": 5.5}
GET  /model_info      -> the per-layer table: params, bytes, MACs, and whether it fits the doorbell
POST /predict         one reading  -> {"days_left": 52.1, "timings_ms": {...}}
POST /predict_batch   {"rows": [reading, ...]} -> {"days_left": [...], "timings_ms": {...}}
A reading that is out of range gets HTTP 422 with the reason, never a guess.
"""
import json
import math
import time

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field, ValidationInfo, field_validator
from starlette.responses import JSONResponse

from app.model import API_FEATURE_RANGES, FEATURES, BatteryNet, model_metrics

with open("model/config.json", encoding="utf-8") as file:
    cfg = json.load(file)
net = BatteryNet(tuple(cfg["hidden"]))
net.load_state_dict(torch.load("model/model.pt", map_location="cpu", weights_only=True))
net.eval()
torch.set_num_threads(1)
MU, SD = np.array(cfg["mu"], np.float32), np.array(cfg["sd"], np.float32)
N_PARAMS = sum(p.numel() for p in net.parameters())


class Reading(BaseModel):
    battery_v: float
    temp_c: float
    rings_per_day: float
    wifi_rssi_dbm: float
    age_months: float

    @field_validator(*FEATURES)
    @classmethod
    def validate_observed_range(cls, value: float, info: ValidationInfo) -> float:
        if not math.isfinite(value):
            raise ValueError(f"{info.field_name} must be finite")
        low, high = API_FEATURE_RANGES[info.field_name]
        if not low <= value <= high:
            raise ValueError(
                f"{info.field_name} must be between {low:g} and {high:g} (inclusive)"
            )
        return value


class Batch(BaseModel):
    rows: list[Reading] = Field(min_length=1, max_length=10000)


app = FastAPI(title="Doorbell battery API")
ms = lambda t0: round((time.perf_counter() - t0) * 1000, 3)


def json_safe_validation_details(value):
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: json_safe_validation_details(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe_validation_details(item) for item in value]
    return value


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({"detail": json_safe_validation_details(exc.errors())}),
    )


@app.get("/model_info")
def model_info():
    return model_metrics(net)


@app.get("/health")
def health():
    return {"status": "ok", "team": cfg["team"], "params": N_PARAMS, "val_mae": cfg["val_mae"]}


async def run(data: Reading | Batch):
    t_all = time.perf_counter()
    timings = {}
    t0 = time.perf_counter()
    rows = data.rows if isinstance(data, Batch) else [data]
    values = [[getattr(r, f) for f in FEATURES] for r in rows]
    timings["parse"] = ms(t0)

    t0 = time.perf_counter()
    x = (np.array(values, np.float32) - MU) / SD
    timings["normalise"] = ms(t0)

    t0 = time.perf_counter()
    with torch.no_grad():
        y = net(torch.from_numpy(x)).numpy()
    timings["inference"] = ms(t0)

    t0 = time.perf_counter()
    days = [round(float(v), 2) for v in np.maximum(y, 0)]
    body = {"days_left": days if isinstance(data, Batch) else days[0]}
    timings["build"] = ms(t0)
    timings["server_total"] = ms(t_all)
    body["timings_ms"] = timings
    return body


@app.post("/predict")
async def predict(reading: Reading):
    return await run(reading)


@app.post("/predict_batch")
async def predict_batch(batch: Batch):
    return await run(batch)
