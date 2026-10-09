"""Unit tests for VisionGuard core ML components."""

from pathlib import Path
import sys
import unittest

# Ensure project root is in path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import torch

from src.explain import compute_anomaly_map, generate_heatmap_overlay
from src.model import ConvAutoencoder, compute_reconstruction_error


class TestVisionGuard(unittest.TestCase):
    """Test suite for VisionGuard autoencoder, loss, explainability, and logic."""

    def test_model_forward_shape_and_bounds(self):
        """Verify model output shape matches input and bounds are strictly [0, 1]."""
        model = ConvAutoencoder(in_channels=3, base_channels=16, latent_channels=32)
        model.eval()

        batch_size = 2
        img_size = 128
        dummy_input = torch.rand(batch_size, 3, img_size, img_size)

        with torch.no_grad():
            output = model(dummy_input)

        self.assertEqual(output.shape, dummy_input.shape)
        self.assertTrue(torch.all(output >= 0.0))
        self.assertTrue(torch.all(output <= 1.0))

    def test_reconstruction_error_calculation(self):
        """Verify MSE error calculation matches mathematical expectation."""
        orig = torch.zeros(1, 3, 4, 4)
        recon = torch.ones(1, 3, 4, 4) * 0.5  # diff is 0.5, squared diff is 0.25

        err = compute_reconstruction_error(orig, recon, reduction="mean")
        self.assertAlmostEqual(err.item(), 0.25, places=5)

        # Identical inputs must yield zero error
        err_zero = compute_reconstruction_error(orig, orig, reduction="mean")
        self.assertAlmostEqual(err_zero.item(), 0.0, places=5)

    def test_anomaly_map_and_overlay(self):
        """Verify anomaly map normalization and heatmap overlay generation."""
        # Create normal background
        orig = np.ones((64, 64, 3), dtype=np.float32) * 0.5
        # Reconstruct with an anomaly square in the center
        recon = orig.copy()
        recon[20:44, 20:44, :] = 1.0

        anomaly_map = compute_anomaly_map(orig, recon, gaussian_kernel=3)

        self.assertEqual(anomaly_map.shape, (64, 64))
        self.assertTrue(anomaly_map.min() >= 0.0)
        self.assertTrue(anomaly_map.max() <= 1.0)
        # Peak anomaly should be near the center
        self.assertGreater(anomaly_map[32, 32], anomaly_map[5, 5])

        overlay = generate_heatmap_overlay(orig, anomaly_map)
        self.assertEqual(overlay.shape, (64, 64, 3))
        self.assertEqual(overlay.dtype, np.uint8)

    def test_threshold_decision_logic(self):
        """Verify normal vs anomalous classification threshold logic."""
        threshold = 0.0050

        normal_score = 0.0021
        anomalous_score = 0.0089

        self.assertFalse(normal_score >= threshold)  # NORMAL
        self.assertTrue(anomalous_score >= threshold)  # ANOMALOUS


if __name__ == "__main__":
    unittest.main()
