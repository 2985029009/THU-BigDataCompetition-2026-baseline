import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

from train import (  # noqa: E402
    HeadFocusedRankingLoss,
    apply_stability_weight_override,
    build_ranking_criterion,
)


RUNNER_PATH = (
    ROOT
    / "codex_generated"
    / "scripts"
    / "experiments"
    / "run_hybrid_weak_20250531_20260531_stability_weight_ablation_20260804.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("stability_ablation_runner", RUNNER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load runner from {RUNNER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class StabilityWeightAblationTests(unittest.TestCase):
    def test_cli_override_sets_weight_and_rejects_negative(self):
        cfg = {"stability_weight": 0.05}
        self.assertIs(apply_stability_weight_override(cfg, 0.10), cfg)
        self.assertEqual(cfg["stability_weight"], 0.10)
        with self.assertRaisesRegex(ValueError, "stability-weight"):
            apply_stability_weight_override(cfg, -0.01)

    def test_head_focused_criterion_uses_stability_weight(self):
        base_cfg = {
            "ranking_loss_type": "head_focused",
            "head_k": 5,
            "head_loss_weight": 1.0,
            "global_rank_weight": 0.20,
        }
        criterion_05 = build_ranking_criterion(
            {**base_cfg, "stability_weight": 0.05}
        )
        criterion_10 = build_ranking_criterion(
            {**base_cfg, "stability_weight": 0.10}
        )
        self.assertIsInstance(criterion_05, HeadFocusedRankingLoss)
        self.assertIsInstance(criterion_10, HeadFocusedRankingLoss)

        # Reversed predictions make the correlation -1, so the stability raw
        # component is exactly one and the two runs differ only by its weight.
        prediction = torch.tensor([[5.0, 4.0, 3.0, 2.0, 1.0, 0.0]])
        target = torch.tensor([[0.0, 1.0, 2.0, 3.0, 4.0, 5.0]])
        _, components_05 = criterion_05.forward_with_components(prediction, target)
        _, components_10 = criterion_10.forward_with_components(prediction, target)
        self.assertAlmostEqual(float(components_05["stability"]), 0.05, places=6)
        self.assertAlmostEqual(float(components_10["stability"]), 0.10, places=6)
        self.assertAlmostEqual(
            float(components_10["stability"] - components_05["stability"]),
            0.05,
            places=6,
        )

    def test_runner_commands_differ_only_by_stability_and_output(self):
        runner = load_runner()
        command_05 = runner.build_command(0.05, Path("run_05"))
        command_10 = runner.build_command(0.10, Path("run_10"))
        normalized_05 = list(command_05)
        normalized_10 = list(command_10)
        for command in (normalized_05, normalized_10):
            output_index = command.index("--output-dir") + 1
            stability_index = command.index("--stability-weight") + 1
            command[output_index] = "<output-dir>"
            command[stability_index] = "<stability-weight>"
        self.assertEqual(normalized_05, normalized_10)
        self.assertEqual(
            command_05[command_05.index("--stability-weight") + 1], "0.05"
        )
        self.assertEqual(
            command_10[command_10.index("--stability-weight") + 1], "0.10"
        )
        self.assertEqual(runner.GLOBAL_RANK_WEIGHT, 0.20)

    def test_runner_manifest_is_side_effect_free_and_complete(self):
        runner = load_runner()
        manifest = runner.build_manifest()
        self.assertEqual(manifest["status"], "running")
        self.assertEqual(manifest["global_rank_weight"], 0.20)
        self.assertEqual(manifest["stability_weights"], [0.10, 0.05])
        self.assertEqual(manifest["train_days"], 241)
        self.assertEqual(manifest["evaluation_days"], 5)
        self.assertEqual(manifest["runs"], [])

    def test_cli_negative_value_is_rejected_before_training(self):
        process = subprocess.run(
            [
                sys.executable,
                str(SRC_DIR / "train.py"),
                "--stability-weight",
                "-0.01",
            ],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("stability-weight", process.stdout + process.stderr)


if __name__ == "__main__":
    unittest.main()
