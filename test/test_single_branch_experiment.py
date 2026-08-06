import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = (
    ROOT / "codex_generated" / "scripts" / "experiments"
    / "run_single_branch_lr5e5_val2m_test_july06_10_20260805.py"
)


def load_runner():
    spec = importlib.util.spec_from_file_location("single_branch_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SingleBranchExperimentTests(unittest.TestCase):
    def test_commands_differ_only_by_variant_and_output(self):
        runner = load_runner()
        left = runner.command("itransformer_only", Path("it"))
        right = runner.command("tcn_only", Path("tcn"))
        normalized = []
        for command in (left, right):
            command = list(command)
            command[command.index("--model-variant") + 1] = "<variant>"
            command[command.index("--output-dir") + 1] = "<output>"
            normalized.append(command)
        self.assertEqual(normalized[0], normalized[1])

    def test_protocol_is_validation_then_refit_then_test(self):
        runner = load_runner()
        command = runner.command("tcn_only", Path("tcn"))
        self.assertIn("--refit-after-validation", command)
        self.assertEqual(command[command.index("--train-days") + 1], "439")
        self.assertEqual(command[command.index("--val-days") + 1], "39")
        self.assertEqual(command[command.index("--gap2-days") + 1], "24")
        self.assertEqual(command[command.index("--test-days") + 1], "5")
        self.assertEqual(command[command.index("--learning-rate") + 1], "0.00005")


if __name__ == "__main__":
    unittest.main()
