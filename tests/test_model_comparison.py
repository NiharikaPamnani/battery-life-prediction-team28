import unittest
from pathlib import Path

import pandas as pd


class ModelComparisonTests(unittest.TestCase):
    def test_metrics_and_saved_models_exist_for_every_candidate(self):
        output = Path("model_comparison")
        metrics = pd.read_csv(output / "metrics.csv")
        expected = {
            "Mean baseline",
            "Linear regression",
            "Neural network",
            "Random Forest",
            "XGBoost",
        }
        self.assertEqual(set(metrics["model"]), expected)
        self.assertTrue({
            "validation_mae",
            "validation_rmse",
            "validation_r2",
            "training_time_seconds",
            "inference_ms_per_sample",
        }.issubset(metrics.columns))
        self.assertTrue(metrics["validation_mae"].notna().all())
        self.assertTrue((metrics["training_time_seconds"] >= 0).all())
        self.assertTrue((metrics["inference_ms_per_sample"] >= 0).all())
        self.assertTrue((output / "models" / "random_forest.joblib").is_file())
        self.assertTrue((output / "models" / "xgboost.json").is_file())

    def test_prediction_artifact_contains_the_shared_validation_rows(self):
        predictions = pd.read_csv(Path("model_comparison") / "validation_predictions.csv")
        self.assertEqual(len(predictions), 395)
        self.assertEqual(len(predictions.columns), 7)
        self.assertTrue(predictions["actual_days_left"].notna().all())


if __name__ == "__main__":
    unittest.main()
