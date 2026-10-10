"""VisionGuard Baseline Image-Difference Anomaly Scoring.

========================================================================================
IMPORTANT NOTICE:
This module provides a HEURISTIC BASELINE anomaly score based on direct image subtraction.
It is NOT a trained machine learning model.

PURPOSE:
In computer vision pipelines, it is standard practice to establish an empirical baseline
using direct pixel difference metrics (e.g. MSE, MAE) before introducing learned models
(like Convolutional Autoencoders). This provides a reproducible lower-bound benchmark
for detection accuracy and sanity checks on preprocessing.
========================================================================================
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import List, Optional, Tuple, Union
import numpy as np
from PIL import Image

# Enable running as script or module
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing import ImagePreprocessor, preprocess_image


@dataclass
class BaselineAnomalyResult:
    """Container for baseline anomaly detection output.

    Attributes:
        score: Scalar numerical anomaly score.
        threshold: Decision threshold used for classification.
        is_anomalous: Boolean flag indicating if score >= threshold.
        metric: Metric used ('mse', 'mae', or 'peak').
        anomaly_map: 2D numpy array of shape (H, W) showing localized pixel difference.
        prediction: Human-readable string ('ANOMALOUS' or 'NORMAL').
    """
    score: float
    threshold: float
    is_anomalous: bool
    metric: str
    anomaly_map: np.ndarray
    prediction: str

    def to_dict(self) -> dict:
        """Convert result summary to dictionary."""
        return {
            "score": round(self.score, 6),
            "threshold": round(self.threshold, 6),
            "is_anomalous": self.is_anomalous,
            "prediction": self.prediction,
            "metric": self.metric,
        }


def compute_pixel_difference(
    test_image: np.ndarray,
    reference_image: np.ndarray,
    metric: str = "mse",
) -> Tuple[float, np.ndarray]:
    """Compute direct pixel-difference error between test and reference images.

    Args:
        test_image: Normalized image array of shape (H, W, C) in [0.0, 1.0].
        reference_image: Normalized reference image array of identical shape (H, W, C).
        metric: Error metric ('mse', 'mae', 'peak').

    Returns:
        Tuple of (scalar_score, 2D_anomaly_map of shape (H, W)).

    Raises:
        ValueError: If array dimensions do not match or unsupported metric.
    """
    if test_image.shape != reference_image.shape:
        raise ValueError(
            f"Shape mismatch: test_image shape {test_image.shape} != "
            f"reference_image shape {reference_image.shape}"
        )

    # Pixel-wise absolute difference
    diff = np.abs(test_image - reference_image)

    # 2D anomaly heatmap: mean absolute difference across RGB channels
    anomaly_map = np.mean(diff, axis=-1).astype(np.float32)

    # Calculate scalar score
    metric_lower = metric.lower()
    if metric_lower == "mse":
        score = float(np.mean((test_image - reference_image) ** 2))
    elif metric_lower == "mae":
        score = float(np.mean(diff))
    elif metric_lower == "peak":
        score = float(np.max(diff))
    else:
        raise ValueError(
            f"Unsupported metric '{metric}'. Supported metrics are 'mse', 'mae', 'peak'."
        )

    return score, anomaly_map


class BaselineAnomalyDetector:
    """Heuristic baseline detector comparing test images against a reference image.

    NOTE: Baseline heuristic only. Not a trained neural network.
    """

    def __init__(
        self,
        metric: str = "mse",
        threshold: float = 0.02,
        target_size: Union[int, Tuple[int, int]] = (128, 128),
    ) -> None:
        """Initialize baseline detector.

        Args:
            metric: Error metric ('mse', 'mae', 'peak'). Default is 'mse'.
            threshold: Decision boundary for classifying as anomaly.
            target_size: Target dimensions for image preprocessing (H, W).
        """
        self.metric = metric
        self.threshold = threshold
        self.target_size = target_size
        self.preprocessor = ImagePreprocessor(target_size=target_size, range_mode="zero_one", to_tensor=False)

    def score(
        self,
        test_image: Union[str, Path, Image.Image, np.ndarray],
        reference_image: Union[str, Path, Image.Image, np.ndarray],
        threshold: Optional[float] = None,
    ) -> BaselineAnomalyResult:
        """Score a test image against a reference normal image.

        Args:
            test_image: Path, PIL Image, or preprocessed array for the test sample.
            reference_image: Path, PIL Image, or preprocessed array for the reference normal sample.
            threshold: Optional threshold override.

        Returns:
            BaselineAnomalyResult containing score, decision, and anomaly map.
        """
        test_arr = self.preprocessor.preprocess(test_image)
        ref_arr = self.preprocessor.preprocess(reference_image)

        score, anomaly_map = compute_pixel_difference(test_arr, ref_arr, metric=self.metric)

        active_threshold = threshold if threshold is not None else self.threshold
        is_anomalous = bool(score >= active_threshold)
        prediction = "ANOMALOUS" if is_anomalous else "NORMAL"

        return BaselineAnomalyResult(
            score=score,
            threshold=active_threshold,
            is_anomalous=is_anomalous,
            metric=self.metric,
            anomaly_map=anomaly_map,
            prediction=prediction,
        )

    def calibrate_threshold(
        self,
        normal_samples: List[Union[str, Path, Image.Image, np.ndarray]],
        reference_image: Union[str, Path, Image.Image, np.ndarray],
        percentile: float = 95.0,
    ) -> float:
        """Calibrate decision threshold using normal validation samples.

        Args:
            normal_samples: List of normal images.
            reference_image: Golden reference image.
            percentile: Percentile threshold (e.g. 95th percentile).

        Returns:
            Calibrated threshold value.
        """
        if not normal_samples:
            raise ValueError("Cannot calibrate threshold with empty sample list")

        scores = [self.score(sample, reference_image).score for sample in normal_samples]
        self.threshold = float(np.percentile(scores, percentile))
        return self.threshold


def score_image_difference(
    test_image: Union[str, Path, Image.Image, np.ndarray],
    reference_image: Union[str, Path, Image.Image, np.ndarray],
    metric: str = "mse",
    threshold: float = 0.02,
    target_size: Union[int, Tuple[int, int]] = (128, 128),
) -> BaselineAnomalyResult:
    """Convenience function for baseline anomaly scoring."""
    detector = BaselineAnomalyDetector(metric=metric, threshold=threshold, target_size=target_size)
    return detector.score(test_image, reference_image)


def parse_args():
    parser = argparse.ArgumentParser(
        description="VisionGuard Baseline Image-Difference Anomaly Scoring (HEURISTIC BASELINE, NOT A TRAINED MODEL)"
    )
    parser.add_argument("--test", type=str, required=True, help="Path to test image")
    parser.add_argument("--reference", type=str, required=True, help="Path to reference normal image")
    parser.add_argument("--metric", type=str, default="mse", choices=["mse", "mae", "peak"], help="Difference metric")
    parser.add_argument("--threshold", type=float, default=0.02, help="Decision threshold")
    parser.add_argument("--save-map", type=str, default=None, help="Optional path to save anomaly difference map")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("=" * 65)
    print("  VisionGuard - Baseline Image-Difference Anomaly Detector")
    print("  [NOTE: Empirical Baseline Heuristic - NOT a Trained AI Model]")
    print("=" * 65)

    result = score_image_difference(
        test_image=args.test,
        reference_image=args.reference,
        metric=args.metric,
        threshold=args.threshold,
    )

    print(f"Test Image   : {args.test}")
    print(f"Reference    : {args.reference}")
    print(f"Metric       : {result.metric.upper()}")
    print(f"Score        : {result.score:.6f}")
    print(f"Threshold    : {result.threshold:.6f}")
    print(f"Prediction   : {result.prediction} {'[DEFECT DETECTED]' if result.is_anomalous else '[NORMAL]'}")
    print("=" * 65)

    if args.save_map:
        # Normalize heatmap to [0, 255] and save
        map_normalized = (result.anomaly_map / (result.anomaly_map.max() + 1e-8) * 255).astype(np.uint8)
        map_img = Image.fromarray(map_normalized, mode="L")
        out_path = Path(args.save_map)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        map_img.save(out_path)
        print(f"Saved anomaly map to: {out_path}")
