import copy
import sys
import unittest
from pathlib import Path

import torch


SRC_DIR = Path(__file__).resolve().parents[1] / "code" / "src"
sys.path.insert(0, str(SRC_DIR))

from model import INFO_TCN_iTransformer  # noqa: E402
from utils import BASELINE24_FEATURES, CROSS4_FEATURES, MARKET4_FEATURES  # noqa: E402


def gate_config():
    return {
        "d_model": 8,
        "sequence_length": 4,
        "dropout": 0.0,
        "model_variant": "hybrid",
        "tcn_channels": 8,
        "tcn_kernel_size": 2,
        "tcn_num_layers": 2,
        "tcn_dropout": 0.0,
        "tcn_norm_type": "layernorm",
        "it_nhead": 2,
        "it_num_layers": 1,
        "it_dim_feedforward": 16,
        "it_variable_identity_encoding": "learned",
        "fusion_dropout": 0.0,
        "nhead": 2,
        "feature_names": BASELINE24_FEATURES + MARKET4_FEATURES,
        "soft_gate_enabled": True,
        "soft_gate_rule_mode": True,
        "soft_gate_hidden_dim": 8,
        "soft_gate_initial_bias": -1.5,
        "soft_gate_stress_strength": 1.25,
        "soft_gate_latest_weight": 0.75,
        "soft_gate_history_weight": 0.25,
        "soft_gate_threshold_low": 0.5,
        "soft_gate_threshold_high": 1.5,
        "soft_gate_trigger_mode": "two_of_four",
        "soft_gate_min_abnormal_components": 2,
        "soft_gate_component_threshold_low": [0.5, 0.5, 0.5, 0.5],
        "soft_gate_component_threshold_high": [1.5, 1.5, 1.5, 1.5],
        "soft_gate_rule_min": 0.3,
        "soft_gate_rule_max": 0.5,
        "soft_gate_cap": 0.6,
        "soft_gate_residual_fraction": 0.2,
        "soft_gate_market_center": [0.0, 0.0, 0.0, 0.0],
        "soft_gate_market_scale": [1.0, 1.0, 1.0, 1.0],
        "soft_gate_market_features": MARKET4_FEATURES,
        "soft_gate_low_volatility_features": [
            "振幅", "STD5", "STD10", "STD20", "STD60",
            "volatility_10", "volatility_20", "atr_14",
        ],
        "soft_gate_reversal_feature": "ROC60",
        "soft_gate_low_volatility_weight": 0.5,
        "soft_gate_reversal_weight": 0.5,
    }


class SoftMarketGateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.features = BASELINE24_FEATURES + MARKET4_FEATURES
        self.model = INFO_TCN_iTransformer(
            input_dim=len(self.features), config=gate_config(), num_stocks=3
        ).eval()

    def test_calm_market_keeps_gate_closed(self):
        src = torch.zeros(1, 2, 4, len(self.features))
        with torch.no_grad():
            output = self.model(src)
        self.assertEqual(output.shape, (1, 2))
        torch.testing.assert_close(self.model.last_gate_values, torch.zeros(1))

    def test_manual_gate_override_is_exact(self):
        cfg = gate_config()
        cfg["soft_gate_override"] = 0.4
        model = INFO_TCN_iTransformer(
            input_dim=len(self.features), config=cfg, num_stocks=3
        ).eval()
        src = torch.randn(1, 3, 4, len(self.features))
        with torch.no_grad():
            model(src)
        torch.testing.assert_close(
            model.last_gate_values, torch.tensor([0.4])
        )

    def test_gate_does_not_change_backbone_initialization_for_same_seed(self):
        enabled = gate_config()
        disabled = copy.deepcopy(enabled)
        disabled["soft_gate_enabled"] = False
        torch.manual_seed(123)
        model_with_gate = INFO_TCN_iTransformer(
            input_dim=len(self.features), config=enabled, num_stocks=3
        )
        torch.manual_seed(123)
        model_without_gate = INFO_TCN_iTransformer(
            input_dim=len(self.features), config=disabled, num_stocks=3
        )
        without_gate_state = model_without_gate.state_dict()
        for name, value in model_with_gate.state_dict().items():
            if name.startswith("market_gate.") or name.startswith("soft_gate_"):
                continue
            torch.testing.assert_close(value, without_gate_state[name])

    def test_causal_stress_prior_opens_gate_more_in_weak_market(self):
        strong = torch.zeros(1, 2, 4, len(self.features))
        weak = strong.clone()
        market_indices = [self.features.index(name) for name in MARKET4_FEATURES]
        # return low, volatility high, drawdown deep, breadth low
        weak_state = torch.tensor([-2.0, 2.0, -2.0, -2.0])
        weak[:, :, :, market_indices] = weak_state
        with torch.no_grad():
            self.model(strong)
            strong_gate = self.model.last_gate_values.clone()
            self.model(weak)
            weak_gate = self.model.last_gate_values.clone()
        self.assertGreater(float(weak_gate), float(strong_gate))

    def test_one_abnormal_component_does_not_trigger(self):
        src = torch.zeros(1, 2, 4, len(self.features))
        return_index = self.features.index("market_return_20")
        src[:, :, -1, return_index] = -2.0
        with torch.no_grad():
            self.model(src)
        torch.testing.assert_close(self.model.last_gate_values, torch.zeros(1))

    def test_two_abnormal_components_trigger_at_least_rule_minimum(self):
        src = torch.zeros(1, 2, 4, len(self.features))
        return_index = self.features.index("market_return_20")
        volatility_index = self.features.index("market_volatility_20")
        src[:, :, -1, return_index] = -0.5
        src[:, :, -1, volatility_index] = 0.5
        with torch.no_grad():
            self.model.market_gate[-1].bias.fill_(-20.0)
            self.model(src)
        self.assertGreaterEqual(float(self.model.last_gate_values), 0.3)

    def test_learned_residual_cannot_reduce_rule_gate(self):
        src = torch.zeros(1, 2, 4, len(self.features))
        return_index = self.features.index("market_return_20")
        volatility_index = self.features.index("market_volatility_20")
        src[:, :, -1, return_index] = -1.0
        src[:, :, -1, volatility_index] = 1.0
        low = copy.deepcopy(self.model)
        high = copy.deepcopy(self.model)
        with torch.no_grad():
            low.market_gate[-1].bias.fill_(-20.0)
            high.market_gate[-1].bias.fill_(20.0)
            low(src)
            high(src)
        self.assertGreaterEqual(float(low.last_gate_values), 0.3)
        self.assertGreaterEqual(float(high.last_gate_values), float(low.last_gate_values))

    def test_full_gate_prefers_low_volatility_long_term_loser(self):
        src = torch.zeros(1, 2, 4, len(self.features))
        low_vol_indices = [
            self.features.index(name)
            for name in gate_config()["soft_gate_low_volatility_features"]
        ]
        reversal_index = self.features.index("ROC60")
        src[:, 0, -1, low_vol_indices] = -1.0
        src[:, 0, -1, reversal_index] = 1.0
        src[:, 1, -1, low_vol_indices] = 1.0
        src[:, 1, -1, reversal_index] = -1.0
        with torch.no_grad():
            self.model.soft_gate_component_threshold_low.fill_(-2.0)
            self.model.soft_gate_component_threshold_high.fill_(-1.0)
            for parameter in self.model.score_head.parameters():
                parameter.zero_()
            output = self.model(src)
        self.assertGreater(float(output[0, 0]), float(output[0, 1]))

    def test_padding_content_does_not_change_gate_or_valid_scores(self):
        valid = torch.randn(1, 2, 4, len(self.features))
        market_indices = [self.features.index(name) for name in MARKET4_FEATURES]
        valid[:, 1, :, market_indices] = valid[:, 0, :, market_indices]
        mask = torch.tensor([[False, False, True]])
        a = torch.cat([valid, torch.zeros(1, 1, 4, len(self.features))], dim=1)
        b = torch.cat([valid, torch.full((1, 1, 4, len(self.features)), 100.0)], dim=1)
        model_a = copy.deepcopy(self.model)
        model_b = copy.deepcopy(self.model)
        with torch.no_grad():
            output_a = model_a(a, mask)
            output_b = model_b(b, mask)
        torch.testing.assert_close(output_a[:, :2], output_b[:, :2])
        torch.testing.assert_close(
            model_a.last_gate_values, model_b.last_gate_values
        )

    def test_cross4_inputs_keep_rule_gate_compatible(self):
        crossed = gate_config()
        crossed["feature_names"] = self.features + CROSS4_FEATURES
        torch.manual_seed(42)
        model = INFO_TCN_iTransformer(
            input_dim=len(crossed["feature_names"]), config=crossed, num_stocks=3
        ).eval()
        src = torch.zeros(1, 2, 4, len(crossed["feature_names"]))
        with torch.no_grad():
            output = model(src)
        self.assertEqual(output.shape, (1, 2))
        self.assertEqual(
            model.market_feature_indices,
            [crossed["feature_names"].index(name) for name in MARKET4_FEATURES],
        )


if __name__ == "__main__":
    unittest.main()
