import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

from train import build_linear_scheduler  # noqa: E402


def learning_rates(num_epochs, scheduler_total_epochs):
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    scheduler = build_linear_scheduler(
        optimizer,
        {
            "num_epochs": num_epochs,
            "scheduler_total_epochs": scheduler_total_epochs,
        },
    )
    rates = []
    for _ in range(num_epochs):
        rates.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return rates


class FixedRefitSchedulerTests(unittest.TestCase):
    def test_outer_prefix_matches_inner_schedule(self):
        self.assertEqual(learning_rates(50, 50)[:10], learning_rates(10, 50))

    def test_old_short_schedule_is_detectably_different(self):
        self.assertNotEqual(learning_rates(50, 50)[:10], learning_rates(10, 10))

    def test_config_snapshot_records_both_lengths(self):
        payload = {"num_epochs": 10, "scheduler_total_epochs": 50}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["num_epochs"], 10)
        self.assertEqual(loaded["scheduler_total_epochs"], 50)


if __name__ == "__main__":
    unittest.main()
