"""model.py -- the network, shared by train.py and the server so both build the same thing."""
import torch
from torch import nn

FEATURES = ["battery_v", "temp_c", "rings_per_day", "wifi_rssi_dbm", "age_months"]
# Broad physical sensor limits, intentionally not inferred from the training sample extrema.
FEATURE_RANGES = {
    "battery_v": (2.0, 4.5),
    "temp_c": (-40.0, 85.0),
    "rings_per_day": (0.0, 1000.0),
    "wifi_rssi_dbm": (-110.0, -1.0),
    "age_months": (0.0, 600.0),
}
SRAM_BUDGET_BYTES = 300 * 1024
PSRAM_BUDGET_BYTES = 8 * 1024 * 1024


class BatteryNet(nn.Module):
    def __init__(self, hidden: tuple[int, ...] = (32, 32)):
        super().__init__()
        layers, n_in = [], len(FEATURES)
        for h in hidden:
            layers += [nn.Linear(n_in, h), nn.ReLU()]
            n_in = h
        layers.append(nn.Linear(n_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def model_metrics(network: BatteryNet) -> dict:
    linear_layers = [layer for layer in network.net if isinstance(layer, nn.Linear)]
    layers = []
    for index, layer in enumerate(linear_layers, start=1):
        parameters = layer.in_features * layer.out_features + layer.out_features
        layers.append({
            "name": f"linear{index}",
            "in": layer.in_features,
            "out": layer.out_features,
            "params": parameters,
            "bytes_f32": parameters * 4,
            "macs": layer.in_features * layer.out_features,
        })

    total_params = sum(layer["params"] for layer in layers)
    total_bytes_f32 = total_params * 4
    total_bytes_int8 = total_params
    return {
        "layers": layers,
        "total_params": total_params,
        "total_bytes_f32": total_bytes_f32,
        "total_bytes_int8": total_bytes_int8,
        "total_macs": sum(layer["macs"] for layer in layers),
        "fits_300kb_sram_f32": total_bytes_f32 <= SRAM_BUDGET_BYTES,
        "fits_300kb_sram_int8": total_bytes_int8 <= SRAM_BUDGET_BYTES,
    }
