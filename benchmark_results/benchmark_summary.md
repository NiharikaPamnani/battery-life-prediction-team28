# Deployed model benchmark

Ran the existing `client/bench.py` against the API loaded from the original
`model/model.pt` artifact in this Codespace. All **200/200** requests succeeded
with keep-alive enabled.

| Request timing | p50 | p95 | p99 |
|---|---:|---:|---:|
| Round trip | 3.096 ms | 4.968 ms | 5.764 ms |
| API server total | 0.262 ms | 0.409 ms | 0.734 ms |
| Model inference | 0.195 ms | 0.312 ms | 0.603 ms |
| Network + framework overhead | 2.796 ms | 4.642 ms | 5.360 ms |

Percentiles cover requests 2–200, matching the script's warm-request
calculation. The first, cold request took **432.611 ms** round trip
(424.850 ms inference), reflecting first-use runtime initialization.

## Environment and memory

- **CPU:** AMD EPYC 7763; 2 logical processors visible.
- **RAM:** 7.8 GiB total; 4.9 GiB available after the run. The Codespace
  memory cgroup has no configured limit.
- **Runtime:** Python 3.11.18; PyTorch 2.15.0+cpu; NumPy 2.4.6.
- **API process memory:** 259,544 KiB RSS / peak RSS after the benchmark. This
  includes the Python runtime and loaded libraries, not just model weights.
- **Model:** 1,281 parameters; 5,124 FP32 weight bytes; 7,869-byte artifact;
  1,216 MACs per prediction.

This is a **loopback API measurement inside the Codespace**, not public-network
latency or an ESP32 measurement. Detailed stage percentiles, means, memory
scope, runtime versions, and the model artifact SHA-256 are in
[`benchmark_results.json`](./benchmark_results.json).
