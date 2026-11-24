# BatteryLife AI — Smart Battery Intelligence

**AI255TBD Edge AI · Module 1 · Team 28**
Battery-life regression from sensor readings, exposed through FastAPI and an interactive Streamlit dashboard.

## Project overview

The application predicts the remaining life of a doorbell battery in days from five sensor readings. A small PyTorch MLP is the deployed model. The same cleaned dataset and deterministic 80/20 split are used to compare mean, linear regression, MLP, Random Forest, and XGBoost validation results.

The dashboard calls the FastAPI service; it does not duplicate model inference. It includes a single-reading prediction workflow, model and comparison views, dataset analytics, an edge-memory analysis, and CSV batch prediction with per-row errors.

## Results

Validation results below are measured on the fixed 395-row validation split, not the instructor's hidden 400 readings. All regressors use the same cleaned rows and target. Training time and model-only inference latency were measured on the local Windows/Python 3.13 environment; they are not Codespace or ESP32 measurements.

| Model | Validation MAE (days) | RMSE (days) | R² | Training (s) | Inference (ms/sample) | Serialized file |
|---|---:|---:|---:|---:|---:|---:|
| Neural network (5-32-32-1) | 5.499 | 7.679 | 0.987 | 33.930 | 0.155 | 7,869 B |
| XGBoost | 5.675 | 8.849 | 0.983 | 0.174 | 0.499 | 654,500 B |
| Random Forest | 7.212 | 11.285 | 0.972 | 1.947 | 40.838 | 35,007,857 B |
| Linear regression | 23.366 | 31.281 | 0.786 | 0.0004 | 0.010 | 302 B |
| Mean baseline | 53.972 | 67.647 | -0.002 | 0.0001 | 0.002 | 21 B |

The deployed MLP remains selected: it has the best measured validation MAE among these candidates and a compact 1,281-parameter network. Tree-file sizes are serialized artifacts, not SRAM estimates. The comparison does not imply a quantized model has been deployed.

Training removed 24 of 2,000 rows: 11 zero-voltage readings, 3 `-127 °C` temperature sentinels, and 10 zero-dBm RSSI readings. The cleaned dataset has 1,976 rows and no missing values. See [`mission_outputs/`](mission_outputs/) for feature plots and the MLP width/depth sweep, and [`model_comparison/`](model_comparison/) for fitted tree models, metrics, predictions, and comparison plots.

## Architecture

```text
CSV / sensor client
       |
       +--> Streamlit dashboard (port 8501)
       |       | prediction, analytics, model comparison, CSV upload
       |       v
       +--> FastAPI service (port 8000)
               | Pydantic range validation and stage timings
               v
          shared PyTorch model (5 -> 32 -> 32 -> 1)
```

The selected network has ReLU activations, training-row-only standardization, and a non-negative output. The API loads the committed `model/model.pt` weights and `model/config.json` metadata. The dashboard sends inference requests to the API's `/predict` and `/predict_batch` endpoints.

## Dataset and preprocessing

`data/team_28_train.csv` contains 2,000 records, five inputs, and the `days_left` target. Training cleanup uses broad physical sensor limits, while the API accepts only values within the observed ranges of the cleaned training data:

| Field | Meaning | Accepted API range |
|---|---|---:|
| `battery_v` | Battery voltage (V) | 3.35 to 4.20 |
| `temp_c` | Temperature (°C) | 8 to 46 |
| `rings_per_day` | Daily doorbell rings | 0 to 43 |
| `wifi_rssi_dbm` | Wi-Fi received signal (dBm) | -88 to -45 |
| `age_months` | Device age in months | 0 to 36 |
| `days_left` | Prediction target (days) | finite and non-negative in training data |

Inputs outside the accepted API ranges receive HTTP 422 responses. The fixed split shuffles cleaned row positions with NumPy seed 0: 1,581 training rows and 395 validation rows. The MLP's mean and standard deviation are fitted only on its training rows and reused at serving time. The linear baseline uses `numpy.linalg.lstsq` with an intercept. The tree-model selection uses validation MAE; their candidate grids and selected settings are recorded in `model_comparison/metrics.csv`.

## Dashboard

Run `streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501`, then open `http://localhost:8501`.

Dashboard pages:

- **Battery Prediction:** sensor form, visual remaining-days gauge, sensor summary, and API stage timing.
- **Model Dashboard:** deployed architecture, training/validation MAE, parameter counts, model sizes, MACs, and per-layer accounting.
- **Model Comparison:** MAE/RMSE/R² charts, validation predictions, residual distributions, and best-model identification.
- **Dataset Analytics:** row/missing/cleaning summaries, actual feature distributions, target relationships, and correlation heatmap.
- **Edge AI Analysis:** FP32/int8 estimates and SRAM/PSRAM feasibility with quantization caveats.
- **Batch Prediction:** CSV upload, batch API use, per-row rejection details, and downloadable prediction CSV.

Set `BATTERY_API_URL` or edit the sidebar's API URL when the backend is not at `http://127.0.0.1:8000`.

## API reference

Start the service with `uvicorn app.main:app --host 0.0.0.0 --port 8000`. Interactive OpenAPI documentation is available at `/docs`.

| Endpoint | Request | Response / purpose |
|---|---|---|
| `GET /health` | None | Status, team 28, parameter count, validation MAE |
| `GET /model_info` | None | Per-linear-layer parameters, FP32 bytes, MACs, totals, int8 size estimate, SRAM-fit flags |
| `POST /predict` | One sensor object | `days_left` and `timings_ms` |
| `POST /predict_batch` | `{"rows": [sensor, ...]}` | Ordered predictions and batch timings |

Example request:

```json
{
  "battery_v": 3.86,
  "temp_c": 28.0,
  "rings_per_day": 5,
  "wifi_rssi_dbm": -66,
  "age_months": 22
}
```

Example response:

```json
{
  "days_left": 136.93,
  "timings_ms": {
    "parse": 4.164,
    "normalise": 1.154,
    "inference": 10.34,
    "build": 0.332,
    "server_total": 16.338
  }
}
```

This is one real local response; timings vary per request and machine. A valid but out-of-range reading is rejected with HTTP 422 and validation details.

## Edge memory and latency

For the deployed network:

- 1,281 parameters: 5,124 bytes at FP32 or 1,281 bytes as a one-byte-per-parameter INT8 estimate.
- 1,216 MACs per prediction.
- Both weight estimates fit within a 300 KiB SRAM budget; FP32 weights also fit in 8 MiB PSRAM.
- The largest single hidden activation is 32 FP32 values (128 bytes). Input/output buffers can overlap during layer evaluation, so a simple two-buffer estimate is 256 bytes.
- ESP32-S3 compute-only timing at 240 MHz and one MAC/cycle is about 5.1 microseconds. This is a theoretical estimate, not a measured board result; it excludes framework and memory overhead.

The width-by-width memory/MAC table and SRAM thresholds are discussed in the report. One byte per parameter is only an INT8 size estimate; an actual quantized model requires calibration, conversion, and target-device tests.

## Reproducibility

Python 3.11 or newer is recommended. From the repository root:

```powershell
python -m pip install -r requirements.txt
python train.py --team 28 --hidden 32,32 --epochs 1500 --plots-dir mission_outputs
python evaluate_models.py
python -m unittest discover -s tests -v
uvicorn app.main:app --host 0.0.0.0 --port 8000
streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501
python docs/build_report.py
```

The model comparison evaluates mean, linear regression, the committed MLP, Random Forest, and XGBoost. XGBoost is a declared dependency and was successfully installed and run for this project.

## GitHub Codespaces deployment

The devcontainer installs `requirements.txt` and forwards ports 8000 (API) and 8501 (dashboard). On GitHub, create a Codespace on `main`; in its terminal run:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
streamlit run dashboard.py --server.address 0.0.0.0 --server.port 8501
```

Set port **8501** to Public to access the dashboard. Set port **8000** to Public as well if the instructor or a laptop client needs to call the API directly. After a Codespace restart, start both services and check port visibility again.

## Project structure

```text
app/                  FastAPI API and shared PyTorch network
client/               External API benchmark client
data/                  Team 28 source CSV
docs/                  Final report source, PDF builder, generated PDF
mission_outputs/       Dataset plots and neural-network width/depth sweep
model/                 Deployed MLP weights and training configuration
model_comparison/      Model metrics, validation predictions, tree models, charts
tests/                 API and model-comparison tests
dashboard.py           Streamlit + Plotly dashboard
evaluate_models.py     Shared-split model comparison
train.py               Cleaning, baselines, MLP training, and plots
```

## Final report

- Editable report: [`docs/FINAL_REPORT.md`](docs/FINAL_REPORT.md)
- PDF report: [`docs/Battery_Life_Prediction_Edge_AI_Report.pdf`](docs/Battery_Life_Prediction_Edge_AI_Report.pdf)
- Regenerate the PDF with `python docs/build_report.py`.
