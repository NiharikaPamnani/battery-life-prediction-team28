import unittest

from fastapi.testclient import TestClient

from app.main import app


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
        lower_boundaries = {
            "battery_v": 2.0,
            "temp_c": -40.0,
            "rings_per_day": 1000.0,
            "wifi_rssi_dbm": -110.0,
            "age_months": 600.0,
        }
        upper_boundaries = {
            "battery_v": 4.5,
            "temp_c": 85.0,
            "rings_per_day": 0.0,
            "wifi_rssi_dbm": -1.0,
            "age_months": 0.0,
        }
        for reading in (lower_boundaries, upper_boundaries):
            with self.subTest(reading=reading):
                response = self.client.post("/predict", json=reading)
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertIsInstance(body["days_left"], float)
                self.assertEqual(
                    set(body["timings_ms"]),
                    {"parse", "normalise", "inference", "build", "server_total"},
                )

    def test_each_out_of_range_sensor_value_is_rejected(self):
        broken_values = {
            "battery_v": 0.0,
            "temp_c": -127.0,
            "rings_per_day": -1.0,
            "wifi_rssi_dbm": 0.0,
            "age_months": -1.0,
        }
        for field, value in broken_values.items():
            with self.subTest(field=field):
                reading = {**self.reading, field: value}
                response = self.client.post("/predict", json=reading)
                self.assertEqual(response.status_code, 422)
                self.assertIn("detail", response.json())

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
