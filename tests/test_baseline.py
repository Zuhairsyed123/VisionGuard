"""Unit tests for VisionGuard baseline image-difference anomaly scoring."""

from pathlib import Path
import sys
import numpy as np
from PIL import Image
import pytest

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.baseline import (
    BaselineAnomalyDetector,
    BaselineAnomalyResult,
    compute_pixel_difference,
    score_image_difference,
)


@pytest.fixture
def baseline_test_images():
    """Fixture creating normal reference image and an anomalous image with a defect."""
    # Normal image: uniform gray background
    normal_img = np.ones((64, 64, 3), dtype=np.float32) * 0.5

    # Identical copy
    identical_img = normal_img.copy()

    # Anomalous image: same background with a bright defect square in center
    defective_img = normal_img.copy()
    defective_img[24:40, 24:40, :] = 1.0  # Bright defect patch

    return {
        "normal": normal_img,
        "identical": identical_img,
        "defective": defective_img,
    }


def test_identical_images_zero_anomaly_score(baseline_test_images):
    """Test that identical images produce zero anomaly score across all metrics."""
    normal = baseline_test_images["normal"]
    identical = baseline_test_images["identical"]

    for metric in ["mse", "mae", "peak"]:
        score, diff_map = compute_pixel_difference(normal, identical, metric=metric)
        assert score == pytest.approx(0.0, abs=1e-7)
        assert np.all(diff_map == 0.0)
        assert diff_map.shape == (64, 64)


def test_defective_image_positive_anomaly_score(baseline_test_images):
    """Test that a defective image yields positive score and localized anomaly map."""
    normal = baseline_test_images["normal"]
    defective = baseline_test_images["defective"]

    score, diff_map = compute_pixel_difference(defective, normal, metric="mse")

    assert score > 0.0
    assert diff_map.shape == (64, 64)
    # Peak anomaly should be inside the defect region (32, 32)
    assert diff_map[32, 32] > diff_map[5, 5]


def test_baseline_detector_decision_logic(baseline_test_images):
    """Test normal vs anomalous classification threshold logic."""
    detector = BaselineAnomalyDetector(metric="mse", threshold=0.01, target_size=(64, 64))

    normal = baseline_test_images["normal"]
    identical = baseline_test_images["identical"]
    defective = baseline_test_images["defective"]

    # Identical pair -> NORMAL
    result_normal = detector.score(identical, normal)
    assert isinstance(result_normal, BaselineAnomalyResult)
    assert not result_normal.is_anomalous
    assert result_normal.prediction == "NORMAL"

    # Defective pair -> ANOMALOUS (if above threshold)
    result_defective = detector.score(defective, normal)
    assert result_defective.is_anomalous
    assert result_defective.prediction == "ANOMALOUS"


def test_shape_mismatch_raises_error():
    """Test that comparing images with mismatched dimensions raises ValueError."""
    img_a = np.zeros((64, 64, 3), dtype=np.float32)
    img_b = np.zeros((32, 32, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="Shape mismatch"):
        compute_pixel_difference(img_a, img_b)


def test_threshold_calibration():
    """Test calibrating baseline threshold using validation normal samples."""
    detector = BaselineAnomalyDetector(metric="mse", target_size=(32, 32))

    ref = np.ones((32, 32, 3), dtype=np.float32) * 0.5
    # Normal samples with slight sensor noise
    sample1 = ref + 0.001
    sample2 = ref + 0.002
    sample3 = ref + 0.003

    calibrated_thresh = detector.calibrate_threshold([sample1, sample2, sample3], ref, percentile=90.0)
    assert calibrated_thresh > 0.0
    assert detector.threshold == calibrated_thresh


def test_score_image_difference_convenience(tmp_path):
    """Test score_image_difference helper with file paths."""
    ref_path = tmp_path / "ref.png"
    test_path = tmp_path / "test.png"

    arr_ref = (np.ones((40, 40, 3)) * 128).astype(np.uint8)
    arr_test = arr_ref.copy()
    arr_test[15:25, 15:25] = 255

    Image.fromarray(arr_ref).save(ref_path)
    Image.fromarray(arr_test).save(test_path)

    result = score_image_difference(test_path, ref_path, threshold=0.01, target_size=(40, 40))
    assert result.is_anomalous
    assert result.score > 0.0
