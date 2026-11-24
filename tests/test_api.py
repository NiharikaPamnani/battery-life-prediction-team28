import unittest

from fastapi.testclient import TestClient

from app.main import app
from app.model import API_FEATURE_RANGES


class BatteryApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.reading = {
            "battery_v": 3.86,
            "temp_c": 28.0,
            "rings_per_day": 5,
            "wifi_rssi_dbm": -66,
            "age_months": 22,
        }

    def test_health_and_model_information_match(self):
        health = self.client.get("/health")
        model = self.client.get("/model_info")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json()["team"], 28)
        self.assertEqual(health.json()["params"], model.json()["total_params"])
        self.assertEqual(model.json()["total_params"], 1281)
        self.assertEqual(model.json()["total_macs"], 1216)
        self.assertEqual(model.json()["total_bytes_int8"], 1281)

    def test_prediction_includes_every_stage_and_accepts_realistic_boundaries(self):
        for field, values in API_FEATURE_RANGES.items():
            for value in values:
                reading = {**self.reading, field: value}
                with self.subTest(field=field, value=value):
                    response = self.client.post("/predict", json=reading)
                    self.assertEqual(response.status_code, 200, response.text)
                    body = response.json()
                    self.assertIsInstance(body["days_left"], float)
                    self.assertEqual(
                        set(body["timings_ms"]),
                        {"parse", "normalise", "inference", "build", "server_total"},
                    )

    def test_valid_prediction_is_unchanged(self):
        response = self.client.post("/predict", json=self.reading)
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["days_left"], 136.93)

    def test_each_out_of_range_sensor_value_is_rejected_with_allowed_range(self):
        for field, (low, high) in API_FEATURE_RANGES.items():
            for value in (low - 0.01, high + 0.01):
                reading = {**self.reading, field: value}
                with self.subTest(field=field, value=value):
                    response = self.client.post("/predict", json=reading)
                    self.assertEqual(response.status_code, 422)
                    error = response.json()["detail"][0]
                    self.assertEqual(error["loc"], ["body", field])
                    self.assertIn(f"{field} must be between {low:g} and {high:g}", error["msg"])

    def test_unrealistic_feature_values_are_rejected(self):
        cases = (("age_months", 600.0), ("rings_per_day", 1000.0))
        for field, value in cases:
            with self.subTest(field=field):
                response = self.client.post(
                    "/predict", json={**self.reading, field: value}
                )
                self.assertEqual(response.status_code, 422)
                self.assertEqual(response.json()["detail"][0]["loc"], ["body", field])

    def test_batch_rejects_an_invalid_row_with_row_and_feature_location(self):
        rows = [
            self.reading,
            {**self.reading, "age_months": 600.0},
        ]
        response = self.client.post("/predict_batch", json={"rows": rows})
        self.assertEqual(response.status_code, 422)
        error = response.json()["detail"][0]
        self.assertEqual(error["loc"], ["body", "rows", 1, "age_months"])
        self.assertIn("age_months must be between 0 and 36", error["msg"])

    def test_batch_matches_individual_predictions(self):
        rows = [
            {**self.reading, "rings_per_day": index % 31, "age_months": index % 37}
            for index in range(50)
        ]
        batch = self.client.post("/predict_batch", json={"rows": rows})
        self.assertEqual(batch.status_code, 200, batch.text)
        individual = [self.client.post("/predict", json=row).json()["days_left"] for row in rows]
        self.assertEqual(batch.json()["days_left"], individual)
        self.assertEqual(
            set(batch.json()["timings_ms"]),
            {"parse", "normalise", "inference", "build", "server_total"},
        )

    def test_docs_page_is_available(self):
        response = self.client.get("/docs")
        self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
