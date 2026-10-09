"""VisionGuard Evaluation Pipeline.

Evaluates the trained Convolutional Autoencoder on the test dataset across
normal and all defect types, computing Accuracy, Precision, Recall, F1-Score,
ROC-AUC, and the Confusion Matrix with detailed breakdowns.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

# Enable running both `python src/evaluate.py` and `python -m src.evaluate`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch
from torchvision import transforms

from src.detect import load_model_and_threshold
from src.model import compute_reconstruction_error, get_device


def evaluate_dataset(
    data_dir: str = "data/bottle",
    checkpoint_path: str = "checkpoints/visionguard_model.pth",
    output_dir: str = "outputs",
    threshold_override: float = None,
) -> Dict:
    """Evaluate trained VisionGuard model on the test split."""
    test_dir = Path(data_dir) / "test"
    if not test_dir.exists():
        raise FileNotFoundError(
            f"Test directory not found: '{test_dir}'.\n"
            f"Please verify dataset exists at '{data_dir}/test'."
        )

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = get_device()
    model, threshold, img_size, category = load_model_and_threshold(checkpoint_path, device=device)

    if threshold_override is not None:
        threshold = threshold_override

    # Discover test images and categorize them
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    subdirs = [d for d in test_dir.iterdir() if d.is_dir()]

    if not subdirs:
        raise ValueError(f"No subdirectories found in test folder '{test_dir}'.")

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    results: List[Dict] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    scores: List[float] = []
    defect_types: List[str] = []

    print("=" * 65)
    print("                    VisionGuard Model Evaluation")
    print("=" * 65)
    print(f"Dataset category : {category}")
    print(f"Device           : {device.type.upper()}")
    print(f"Anomaly threshold: {threshold:.6f}")
    print(f"Test directory   : {test_dir}")
    print("-" * 65)

    with torch.no_grad():
        for subdir in sorted(subdirs):
            folder_name = subdir.name
            is_good = folder_name.lower() == "good"
            true_label = 0 if is_good else 1  # 0: Normal, 1: Defect

            images = sorted([p for p in subdir.glob("*") if p.suffix.lower() in extensions])
            print(f"Evaluating '{folder_name}' ({len(images)} images)...")

            for img_path in images:
                pil_img = Image.open(img_path).convert("RGB")
                img_tensor = transform(pil_img).unsqueeze(0).to(device)
                recon_tensor = model(img_tensor)

                err = compute_reconstruction_error(img_tensor, recon_tensor, reduction="none")
                score = float(err.item())

                pred_label = 1 if score >= threshold else 0

                y_true.append(true_label)
                y_pred.append(pred_label)
                scores.append(score)
                defect_types.append(folder_name)

                results.append({
                    "file": str(img_path),
                    "folder": folder_name,
                    "true_label": true_label,
                    "score": score,
                    "pred_label": pred_label,
                    "correct": true_label == pred_label,
                })

    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    scores_arr = np.array(scores)

    # Compute classification metrics
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    prec = float(precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec = float(recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred_arr, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true_arr, scores_arr))
    except Exception:
        roc_auc = float("nan")

    # Confusion matrix
    cm = confusion_matrix(y_true_arr, y_pred_arr)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    else:
        tn = int(np.sum((y_true_arr == 0) & (y_pred_arr == 0)))
        fp = int(np.sum((y_true_arr == 0) & (y_pred_arr == 1)))
        fn = int(np.sum((y_true_arr == 1) & (y_pred_arr == 0)))
        tp = int(np.sum((y_true_arr == 1) & (y_pred_arr == 1)))

    total_samples = len(y_true_arr)
    normal_samples = int(np.sum(y_true_arr == 0))
    anomalous_samples = int(np.sum(y_true_arr == 1))
    correct_predictions = tn + tp

    print("=" * 65)
    print("                     EVALUATION SUMMARY")
    print("=" * 65)
    print(f"Total Test Samples     : {total_samples}")
    print(f"Normal Samples         : {normal_samples}")
    print(f"Anomalous Samples      : {anomalous_samples}")
    print(f"Correct Predictions    : {correct_predictions} / {total_samples}")
    print(f"False Positives (FP)   : {fp}")
    print(f"False Negatives (FN)   : {fn}")
    print("-" * 65)
    print(f"Accuracy               : {acc * 100:.2f}%")
    print(f"Precision              : {prec * 100:.2f}%")
    print(f"Recall (Sensitivity)   : {rec * 100:.2f}%")
    print(f"F1-Score               : {f1:.4f}")
    if not np.isnan(roc_auc):
        print(f"ROC-AUC                : {roc_auc:.4f}")
    print("-" * 65)
    print("Confusion Matrix:")
    print(f"  [TN: {tn:3d} | FP: {fp:3d}]")
    print(f"  [FN: {fn:3d} | TP: {tp:3d}]")
    print("-" * 65)

    # Per-category breakdown
    print("Breakdown by defect subtype:")
    unique_subtypes = sorted(list(set(defect_types)))
    breakdown = {}
    for st in unique_subtypes:
        indices = [i for i, d in enumerate(defect_types) if d == st]
        st_count = len(indices)
        st_correct = sum(1 for i in indices if results[i]["correct"])
        st_acc = (st_correct / st_count) * 100 if st_count > 0 else 0
        mean_score = float(np.mean([scores[i] for i in indices]))
        breakdown[st] = {
            "count": st_count,
            "correct": st_correct,
            "accuracy_pct": round(st_acc, 2),
            "mean_score": round(mean_score, 6),
        }
        print(f"  - {st:<16}: {st_correct:2d}/{st_count:2d} correct ({st_acc:5.1f}%) | Mean Score: {mean_score:.6f}")
    print("=" * 65)

    # Save metrics JSON
    metrics_data = {
        "category": category,
        "anomaly_threshold": threshold,
        "total_samples": total_samples,
        "normal_samples": normal_samples,
        "anomalous_samples": anomalous_samples,
        "correct_predictions": correct_predictions,
        "false_positives": fp,
        "false_negatives": fn,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": {
            "true_negatives": tn,
            "false_positives": fp,
            "false_negatives": fn,
            "true_positives": tp,
        },
        "subtype_breakdown": breakdown,
    }

    metrics_file = out_path / "evaluation_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"Metrics saved to: {metrics_file}")

    # Generate and save ROC curve and score distribution figure
    try:
        plot_file = out_path / "evaluation_roc_curve.png"
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

        # ROC Curve
        if not np.isnan(roc_auc) and len(set(y_true_arr)) > 1:
            fpr, tpr, _ = roc_curve(y_true_arr, scores_arr)
            axes[0].plot(fpr, tpr, color="#1976D2", lw=2, label=f"ROC Curve (AUC = {roc_auc:.3f})")
            axes[0].plot([0, 1], [0, 1], color="#9E9E9E", linestyle="--")
            axes[0].set_xlim([0.0, 1.0])
            axes[0].set_ylim([0.0, 1.05])
            axes[0].set_xlabel("False Positive Rate", fontsize=10)
            axes[0].set_ylabel("True Positive Rate (Recall)", fontsize=10)
            axes[0].set_title("Receiver Operating Characteristic (ROC)", fontsize=11, fontweight="bold")
            axes[0].legend(loc="lower right")
            axes[0].grid(alpha=0.3)

        # Anomaly Score Distribution Histogram
        normal_scores = scores_arr[y_true_arr == 0]
        anom_scores = scores_arr[y_true_arr == 1]
        axes[1].hist(normal_scores, bins=15, alpha=0.6, color="#2E7D32", label="Normal", density=True)
        axes[1].hist(anom_scores, bins=15, alpha=0.6, color="#D32F2F", label="Anomalous", density=True)
        axes[1].axvline(threshold, color="black", linestyle="--", lw=1.5, label=f"Threshold ({threshold:.4f})")
        axes[1].set_xlabel("Reconstruction Error (MSE)", fontsize=10)
        axes[1].set_ylabel("Density", fontsize=10)
        axes[1].set_title("Anomaly Score Distribution", fontsize=11, fontweight="bold")
        axes[1].legend()
        axes[1].grid(alpha=0.3)

        plt.tight_layout()
        fig.savefig(plot_file, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Evaluation plot saved to: {plot_file}")
    except Exception as e:
        print(f"Warning: could not generate evaluation plot: {e}")

    print("=" * 65)
    return metrics_data


def parse_args():
    parser = argparse.ArgumentParser(description="VisionGuard Model Evaluation")
    parser.add_argument("--data-dir", type=str, default="data/bottle", help="Category dataset directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/visionguard_model.pth", help="Checkpoint path")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--threshold", type=float, default=None, help="Threshold override")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_dataset(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        threshold_override=args.threshold,
    )
