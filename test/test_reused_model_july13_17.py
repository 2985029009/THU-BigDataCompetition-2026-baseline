import importlib.util
import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = (
    ROOT
    / "codex_generated"
    / "scripts"
    / "diagnostics"
    / "predict_reused_global0p20_july13_17.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("july_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load frozen inference runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReusedModelJulyWindowTests(unittest.TestCase):
    def test_protocol_is_single_signal_then_five_score_days(self):
        runner = load_runner()
        self.assertEqual(runner.PROTOCOL, "A_single_signal_5day_score")
        self.assertEqual(runner.SIGNAL_DATE, "2026-07-10")
        self.assertEqual(
            runner.SCORE_DATES,
            [
                "2026-07-13",
                "2026-07-14",
                "2026-07-15",
                "2026-07-16",
                "2026-07-17",
            ],
        )
        date_audit = runner.validate_score_window(
            [
                "2026-07-09",
                "2026-07-10",
                "2026-07-13",
                "2026-07-14",
                "2026-07-15",
                "2026-07-16",
                "2026-07-17",
            ]
        )
        self.assertTrue(date_audit["score_window_complete"])
        with self.assertRaises(ValueError):
            runner.validate_score_window(
                ["2026-07-10", "2026-07-13", "2026-07-15"]
            )

    def test_source_config_expectations_and_commands_are_inference_only(self):
        runner = load_runner()
        expected = runner.expected_config_fields()
        self.assertEqual(expected["global_rank_weight"], 0.20)
        self.assertEqual(expected["stability_weight"], 0.05)
        self.assertFalse(expected["soft_gate_enabled"])
        command = runner.build_predict_command(Path("output"))
        self.assertIn("predict.py", command[2])
        self.assertNotIn("train.py", " ".join(command))
        self.assertEqual(runner.build_score_self_command(Path("audit"))[2].split("\\")[-1], "score_self.py")

    def test_signal_label_uses_first_to_fifth_score_open(self):
        runner = load_runner()
        rows = []
        dates = pd.to_datetime(
            [
                "2026-07-10",
                "2026-07-13",
                "2026-07-14",
                "2026-07-15",
                "2026-07-16",
                "2026-07-17",
            ]
        )
        for stock, opens in {
            "000001": [100, 101, 102, 103, 104, 105],
            "000002": [100, 99, 98, 97, 96, 95],
        }.items():
            rows.extend(
                {"股票代码": stock, "日期": date, "开盘": float(open_)}
                for date, open_ in zip(dates, opens)
            )
        labels = runner.build_signal_labels(pd.DataFrame(rows))
        labels = labels.set_index("股票代码")
        self.assertAlmostEqual(labels.loc["000001", "actual_return"], 4 / 101)
        self.assertAlmostEqual(labels.loc["000002", "actual_return"], -4 / 99)


if __name__ == "__main__":
    unittest.main()
