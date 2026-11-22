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
import time

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

from app.model import FEATURES, FEATURE_RANGES, BatteryNet, model_metrics

with open("model/config.json", encoding="utf-8") as file:
    cfg = json.load(file)
net = BatteryNet(tuple(cfg["hidden"]))
net.load_state_dict(torch.load("model/model.pt", map_location="cpu", weights_only=True))
net.eval()
torch.set_num_threads(1)
MU, SD = np.array(cfg["mu"], np.float32), np.array(cfg["sd"], np.float32)
N_PARAMS = sum(p.numel() for p in net.parameters())


class Reading(BaseModel):
    battery_v: float = Field(
        ge=FEATURE_RANGES["battery_v"][0], le=FEATURE_RANGES["battery_v"][1], allow_inf_nan=False
    )
    temp_c: float = Field(
        ge=FEATURE_RANGES["temp_c"][0], le=FEATURE_RANGES["temp_c"][1], allow_inf_nan=False
    )
    rings_per_day: float = Field(
        ge=FEATURE_RANGES["rings_per_day"][0], le=FEATURE_RANGES["rings_per_day"][1], allow_inf_nan=False
    )
    wifi_rssi_dbm: float = Field(
        ge=FEATURE_RANGES["wifi_rssi_dbm"][0], le=FEATURE_RANGES["wifi_rssi_dbm"][1], allow_inf_nan=False
    )
    age_months: float = Field(
        ge=FEATURE_RANGES["age_months"][0], le=FEATURE_RANGES["age_months"][1], allow_inf_nan=False
    )


class Batch(BaseModel):
    rows: list[Reading] = Field(min_length=1, max_length=10000)


app = FastAPI(title="Doorbell battery API")
ms = lambda t0: round((time.perf_counter() - t0) * 1000, 3)


@app.get("/model_info")
def model_info():
    return model_metrics(net)


@app.get("/health")
def health():
    return {"status": "ok", "team": cfg["team"], "params": N_PARAMS, "val_mae": cfg["val_mae"]}


async def run(request: Request, schema: type[BaseModel]):
    t_all = time.perf_counter()
    timings = {}
    t0 = time.perf_counter()
    try:
        data = schema.model_validate_json(await request.body())
    except ValidationError as e:
        return JSONResponse(status_code=422, content={"error": "bad reading", "detail": json.loads(e.json())})
    timings["parse"] = ms(t0)

    t0 = time.perf_counter()
    rows = data.rows if isinstance(data, Batch) else [data]
    x = (np.array([[getattr(r, f) for f in FEATURES] for r in rows], np.float32) - MU) / SD
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
async def predict(request: Request):
    return await run(request, Reading)


@app.post("/predict_batch")
async def predict_batch(request: Request):
    return await run(request, Batch)
