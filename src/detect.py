"""VisionGuard Inference and Detection CLI.

Loads the trained Convolutional Autoencoder and anomaly threshold,
processes a single test image, calculates its anomaly score, classifies it
as NORMAL or ANOMALOUS, and outputs an explainable heatmap visualization.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Tuple

# Enable running both `python src/detect.py` and `python -m src.detect`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

from src.explain import compute_anomaly_map, generate_heatmap_overlay, save_explanation
from src.model import ConvAutoencoder, compute_reconstruction_error, get_device


def load_model_and_threshold(
    checkpoint_path: str = "checkpoints/visionguard_model.pth",
    device: torch.device = None,
) -> Tuple[torch.nn.Module, float, int, str]:
    """Load trained Autoencoder weights and saved decision threshold."""
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'.\n"
            f"Please train the model first by running:\n"
            f"    python src/train.py"
        )

    if device is None:
        device = get_device()

    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
    img_size = checkpoint.get("img_size", 128)
    category = checkpoint.get("category", "unknown")
    threshold = float(checkpoint.get("threshold", 0.01))

    model = ConvAutoencoder(in_channels=3, base_channels=32, latent_channels=128)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, threshold, img_size, category


def detect_anomaly(
    image_path: str,
    checkpoint_path: str = "checkpoints/visionguard_model.pth",
    save_output: str = "outputs/detection_result.png",
    custom_threshold: float = None,
) -> Dict:
    """Run anomaly detection on a single image and save explanation."""
    img_path = Path(image_path)
    if not img_path.exists():
        raise FileNotFoundError(f"Input image not found: '{image_path}'")

    try:
        pil_image = Image.open(img_path).convert("RGB")
    except Exception as e:
        raise ValueError(f"Failed to open image '{image_path}': {e}")

    device = get_device()
    model, threshold, img_size, category = load_model_and_threshold(checkpoint_path, device=device)

    if custom_threshold is not None:
        threshold = custom_threshold

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    tensor_input = transform(pil_image).unsqueeze(0).to(device)

    # Forward pass
    with torch.no_grad():
        reconstructed_tensor = model(tensor_input)
        error_tensor = compute_reconstruction_error(tensor_input, reconstructed_tensor, reduction="none")
        anomaly_score = float(error_tensor.item())

    # Decision
    is_anomalous = anomaly_score >= threshold
    prediction = "ANOMALOUS" if is_anomalous else "NORMAL"

    # Convert to numpy for explainability
    orig_np = tensor_input.squeeze(0).permute(1, 2, 0).cpu().numpy()
    recon_np = reconstructed_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

    # Generate Heatmap and Overlay
    anomaly_map = compute_anomaly_map(orig_np, recon_np)
    overlay = generate_heatmap_overlay(orig_np, anomaly_map)

    # Save visualization
    saved_path = save_explanation(
        original=orig_np,
        reconstructed=recon_np,
        anomaly_map=anomaly_map,
        overlay=overlay,
        save_path=save_output,
        score=anomaly_score,
        threshold=threshold,
        prediction=prediction,
        sample_name=img_path.name,
    )

    print("=" * 60)
    print("                  VisionGuard Anomaly Detection")
    print("=" * 60)
    print(f"Image       : {img_path.name}")
    print(f"Path        : {img_path}")
    print(f"Category    : {category}")
    print(f"Device      : {device.type.upper()}")
    print(f"Anomaly Score: {anomaly_score:.6f}")
    print(f"Threshold   : {threshold:.6f}")
    print(f"Prediction  : {prediction} {'[DEFECT DETECTED]' if is_anomalous else '[PASSED]'}")
    print(f"Explanation : {saved_path}")
    print("=" * 60)

    return {
        "image": str(img_path),
        "score": anomaly_score,
        "threshold": threshold,
        "prediction": prediction,
        "is_anomalous": is_anomalous,
        "visualization_path": saved_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="VisionGuard Image Anomaly Detection")
    parser.add_argument("--image", type=str, required=True, help="Path to input image file")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/visionguard_model.pth", help="Model checkpoint path")
    parser.add_argument("--threshold", type=float, default=None, help="Optional custom anomaly threshold override")
    parser.add_argument("--save-output", type=str, default="outputs/detection_result.png", help="Path to save explanation image")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    detect_anomaly(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        save_output=args.save_output,
        custom_threshold=args.threshold,
    )
