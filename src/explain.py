"""VisionGuard Explainability and Anomaly Heatmap Generation.

Generates pixel-level anomaly maps and visual overlays using the absolute
difference between original and reconstructed images.
"""

from pathlib import Path
from typing import Optional, Tuple, Union
import cv2
import matplotlib.pyplot as plt
import numpy as np


def compute_anomaly_map(
    original: np.ndarray,
    reconstructed: np.ndarray,
    gaussian_kernel: int = 7,
    gaussian_sigma: float = 4.0,
) -> np.ndarray:
    """Compute normalized 2D anomaly heatmap from reconstruction error.

    Args:
        original: RGB image array of shape (H, W, 3) with float values in [0, 1].
        reconstructed: RGB image array of shape (H, W, 3) with float values in [0, 1].
        gaussian_kernel: Size of Gaussian blur filter kernel (odd int).
        gaussian_sigma: Standard deviation for Gaussian blur.

    Returns:
        2D float array of shape (H, W) normalized to [0, 1].
    """
    # Absolute difference per pixel and channel
    diff = np.abs(original.astype(np.float32) - reconstructed.astype(np.float32))

    # Average difference across RGB channels
    anomaly_map = np.mean(diff, axis=-1)

    # Apply Gaussian smoothing to reduce sensor noise and emphasize defect clusters
    if gaussian_kernel > 1:
        if gaussian_kernel % 2 == 0:
            gaussian_kernel += 1
        anomaly_map = cv2.GaussianBlur(
            anomaly_map,
            (gaussian_kernel, gaussian_kernel),
            sigmaX=gaussian_sigma,
            sigmaY=gaussian_sigma,
        )

    # Normalize map to [0, 1] range
    map_min = float(anomaly_map.min())
    map_max = float(anomaly_map.max())
    if map_max - map_min > 1e-8:
        anomaly_map = (anomaly_map - map_min) / (map_max - map_min)
    else:
        anomaly_map = np.zeros_like(anomaly_map)

    return anomaly_map


def generate_heatmap_overlay(
    original: np.ndarray,
    anomaly_map: np.ndarray,
    alpha: float = 0.5,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Generate colorized heatmap overlaid onto the original image.

    Args:
        original: RGB image array of shape (H, W, 3) in [0, 1] or uint8 [0, 255].
        anomaly_map: 2D float array of shape (H, W) in [0, 1].
        alpha: Blend factor for heatmap (0.0 = original only, 1.0 = heatmap only).
        colormap: OpenCV colormap constant (default: cv2.COLORMAP_JET).

    Returns:
        RGB uint8 array of shape (H, W, 3) representing the blended overlay.
    """
    # Ensure original is uint8 RGB
    if original.dtype != np.uint8:
        original_uint8 = np.clip(original * 255.0, 0, 255).astype(np.uint8)
    else:
        original_uint8 = original.copy()

    # Convert normalized anomaly map to uint8
    heatmap_uint8 = np.clip(anomaly_map * 255.0, 0, 255).astype(np.uint8)

    # Apply color map (cv2 outputs BGR)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    # Blend original and heatmap
    overlay = cv2.addWeighted(
        heatmap_rgb,
        alpha,
        original_uint8,
        1.0 - alpha,
        0,
    )

    return overlay


def save_explanation(
    original: np.ndarray,
    reconstructed: np.ndarray,
    anomaly_map: np.ndarray,
    overlay: np.ndarray,
    save_path: Union[str, Path],
    score: Optional[float] = None,
    threshold: Optional[float] = None,
    prediction: Optional[str] = None,
    sample_name: Optional[str] = None,
) -> str:
    """Save multi-panel visualization showing detection and explainability results.

    Panels:
        1. Original Image
        2. Autoencoder Reconstruction
        3. Anomaly Heatmap (Normalized Error)
        4. Heatmap Overlay on Original Image

    Args:
        original: RGB array in [0, 1] or uint8.
        reconstructed: RGB array in [0, 1] or uint8.
        anomaly_map: 2D float array in [0, 1].
        overlay: Blended RGB uint8 array.
        save_path: Destination file path (e.g. outputs/result.png).
        score: Computed anomaly score (MSE).
        threshold: Decision threshold.
        prediction: "NORMAL" or "ANOMALOUS".
        sample_name: Optional title string or image filename.

    Returns:
        String path to the saved visualization image.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # Format values for display
    if original.dtype == np.uint8:
        orig_disp = original / 255.0
    else:
        orig_disp = np.clip(original, 0.0, 1.0)

    if reconstructed.dtype == np.uint8:
        recon_disp = reconstructed / 255.0
    else:
        recon_disp = np.clip(reconstructed, 0.0, 1.0)

    overlay_disp = overlay / 255.0 if overlay.dtype == np.uint8 else overlay

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=120)

    # 1. Original
    axes[0].imshow(orig_disp)
    axes[0].set_title("1. Original Image", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # 2. Reconstructed
    axes[1].imshow(recon_disp)
    axes[1].set_title("2. Reconstructed", fontsize=11, fontweight="bold")
    axes[1].axis("off")

    # 3. Anomaly Heatmap
    im_hm = axes[2].imshow(anomaly_map, cmap="jet", vmin=0.0, vmax=1.0)
    axes[2].set_title("3. Error Heatmap", fontsize=11, fontweight="bold")
    axes[2].axis("off")
    fig.colorbar(im_hm, ax=axes[2], fraction=0.046, pad=0.04)

    # 4. Overlay
    axes[3].imshow(overlay_disp)
    axes[3].set_title("4. Anomaly Overlay", fontsize=11, fontweight="bold")
    axes[3].axis("off")

    # Construct status header
    header = "VisionGuard Anomaly Inspection"
    if sample_name:
        header = f"VisionGuard - {sample_name}"

    color = "black"
    if prediction is not None:
        if prediction.upper() == "ANOMALOUS":
            pred_text = "ANOMALOUS [DEFECT DETECTED]"
            color = "#D32F2F"
        else:
            pred_text = "NORMAL [INSPECTION PASSED]"
            color = "#2E7D32"

        details = ""
        if score is not None and threshold is not None:
            details = f" | Score: {score:.4f} (Threshold: {threshold:.4f})"
        fig.suptitle(f"{header}\nPrediction: {pred_text}{details}", fontsize=13, fontweight="bold", color=color)
    else:
        fig.suptitle(header, fontsize=13, fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return str(save_path)
