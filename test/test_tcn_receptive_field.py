import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

from config import config  # noqa: E402
from model import TemporalConvNet  # noqa: E402


class TCNReceptiveFieldTests(unittest.TestCase):
    def test_current_tcn_covers_full_sixty_day_sequence(self):
        branch = TemporalConvNet(input_dim=24, d_model=32, config=config)

        self.assertEqual(config["tcn_num_layers"], 5)
        self.assertEqual(len(branch.tcn_layers), 5)
        self.assertEqual(branch.receptive_field, 63)
        self.assertGreaterEqual(
            branch.receptive_field,
            config["sequence_length"],
        )

    def test_coverage_guard_rejects_old_four_layer_mainline(self):
        insufficient = {
            "tcn_channels": 8,
            "tcn_kernel_size": 3,
            "tcn_num_layers": 4,
            "tcn_dropout": 0.0,
            "tcn_min_receptive_field": 60,
        }

        with self.assertRaisesRegex(ValueError, "31 < 60"):
            TemporalConvNet(input_dim=4, d_model=8, config=insufficient)

    def test_historical_config_without_guard_remains_loadable(self):
        historical = {
            "tcn_channels": 8,
            "tcn_kernel_size": 3,
            "tcn_num_layers": 4,
            "tcn_dropout": 0.0,
        }

        branch = TemporalConvNet(input_dim=4, d_model=8, config=historical)

        self.assertEqual(branch.receptive_field, 31)
        self.assertEqual(len(branch.tcn_layers), 4)


if __name__ == "__main__":
    unittest.main()
