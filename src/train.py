"""VisionGuard Training Pipeline.

Trains the Convolutional Autoencoder on normal (defect-free) images,
computes reconstruction errors on normal validation data, calculates the
anomaly decision threshold, and saves the trained checkpoint and metadata.
"""

import argparse
import json
import os
from pathlib import Path
import random
import sys
from typing import Dict, List, Tuple

# Enable running both `python src/train.py` and `python -m src.train`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.model import ConvAutoencoder, compute_reconstruction_error, get_device


class NormalImageDataset(Dataset):
    """Dataset for loading normal training/validation images."""

    def __init__(self, file_paths: List[Path], transform=None) -> None:
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.file_paths[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, str(path)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_transforms(img_size: int = 128):
    """Return image transformations for training and inference."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # Scales pixels to [0.0, 1.0]
    ])


def find_images(directory: Path) -> List[Path]:
    """Find all image files in a directory."""
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    return sorted([p for p in directory.rglob("*") if p.suffix.lower() in extensions])


def compute_threshold(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    percentile: float = 95.0,
) -> Tuple[float, List[float]]:
    """Compute anomaly threshold from reconstruction errors on normal images.

    Args:
        model: Trained Autoencoder.
        dataloader: DataLoader containing normal images.
        device: CPU or CUDA device.
        percentile: Cutoff percentile (e.g. 95.0 or 99.0).

    Returns:
        threshold: Numeric anomaly threshold.
        scores: List of reconstruction MSE scores.
    """
    model.eval()
    scores: List[float] = []

    with torch.no_grad():
        for batch_images, _ in dataloader:
            batch_images = batch_images.to(device)
            reconstructed = model(batch_images)
            errors = compute_reconstruction_error(batch_images, reconstructed, reduction="none")
            scores.extend(errors.cpu().numpy().tolist())

    threshold = float(np.percentile(scores, percentile))
    return threshold, scores


def download_mvtec_category(category: str = "bottle", data_root: str = "data") -> Path:
    """Download and extract an official MVTec AD category if missing."""
    target_dir = Path(data_root) / category
    train_good = target_dir / "train" / "good"
    if train_good.exists() and len(list(train_good.glob("*.png"))) > 0:
        print(f"Dataset category '{category}' already present at '{target_dir}'.")
        return target_dir

    print(f"Downloading MVTec AD '{category}' category archive...")
    import ssl
    import urllib.request
    import pyarrow.parquet as pq

    ctx = ssl._create_unverified_context()
    temp_dir = Path("scratch")
    temp_dir.mkdir(parents=True, exist_ok=True)

    base_url = "https://huggingface.co/datasets/TheoM55/mvtec_all_objects_split/resolve/main/data"
    for split in ["train", "test"]:
        pq_name = f"{category}.{split}-00000-of-00001.parquet"
        pq_url = f"{base_url}/{pq_name}"
        pq_path = temp_dir / pq_name

        if not pq_path.exists():
            print(f"Downloading {split} split ({pq_name})...")
            req = urllib.request.Request(pq_url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, context=ctx) as r, open(pq_path, "wb") as f:
                f.write(r.read())

        print(f"Extracting {split} images for '{category}'...")
        table = pq.read_table(pq_path)
        df = table.to_pandas()

        if split == "train":
            out_good = target_dir / "train" / "good"
            out_good.mkdir(parents=True, exist_ok=True)
            for idx, row in df.iterrows():
                with open(out_good / f"{idx:03d}.png", "wb") as f:
                    f.write(row["image_path"]["bytes"])
        else:
            for idx, row in df.iterrows():
                defect = row["defect"]
                out_defect = target_dir / "test" / defect
                out_defect.mkdir(parents=True, exist_ok=True)
                with open(out_defect / f"{idx:03d}.png", "wb") as f:
                    f.write(row["image_path"]["bytes"])

    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    print(f"Successfully prepared '{category}' dataset at '{target_dir}'.")
    return target_dir


def train_autoencoder(
    data_dir: str = "data/bottle",
    category: str = "bottle",
    epochs: int = 25,
    batch_size: int = 16,
    lr: float = 1e-3,
    img_size: int = 128,
    percentile: float = 95.0,
    output_dir: str = "checkpoints",
    seed: int = 42,
    download_if_missing: bool = False,
) -> Dict:
    """Train Convolutional Autoencoder and save checkpoint and threshold."""
    set_seed(seed)
    device = get_device()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    train_good_dir = Path(data_dir) / "train" / "good"
    if not train_good_dir.exists():
        if download_if_missing:
            print(f"Training data not found at '{train_good_dir}'. Attempting automatic download...")
            download_mvtec_category(category=category, data_root=Path(data_dir).parent)
        else:
            raise FileNotFoundError(
                f"Training directory not found: '{train_good_dir}'.\n"
                f"To automatically download and prepare the dataset, run:\n"
                f"    python src/train.py --download\n"
                f"Or download MVTec AD '{category}' manually and place under '{data_dir}'."
            )

    all_images = find_images(train_good_dir)
    if len(all_images) == 0:
        raise ValueError(f"No image files found in '{train_good_dir}'.")

    # Split into train and validation (85% train, 15% val for threshold verification)
    random.shuffle(all_images)
    val_count = max(1, int(len(all_images) * 0.15))
    val_files = all_images[:val_count]
    train_files = all_images[val_count:]

    transform = get_transforms(img_size)
    train_dataset = NormalImageDataset(train_files, transform=transform)
    val_dataset = NormalImageDataset(val_files, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("=" * 60)
    print("                     VisionGuard Training")
    print("=" * 60)
    print(f"Device: {device.type.upper()}")
    print(f"Category: {category}")
    print(f"Training images: {len(train_files)}")
    print(f"Validation images: {len(val_files)}")
    print(f"Image resolution: {img_size}x{img_size}")
    print(f"Epochs: {epochs} | Batch size: {batch_size} | Learning rate: {lr}")
    print("=" * 60)

    model = ConvAutoencoder(in_channels=3, base_channels=32, latent_channels=128).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    history: List[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for batch_images, _ in train_loader:
            batch_images = batch_images.to(device)
            optimizer.zero_grad()
            reconstructed = model(batch_images)
            loss = criterion(reconstructed, batch_images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_images.size(0)

        epoch_loss = running_loss / len(train_files)
        history.append(epoch_loss)

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train MSE Loss: {epoch_loss:.6f}")

    # Compute threshold on validation normal images (and verify on all train normal images)
    print("-" * 60)
    print("Computing anomaly threshold from normal images...")
    threshold, val_scores = compute_threshold(model, val_loader, device, percentile=percentile)
    mean_val_score = float(np.mean(val_scores))
    max_val_score = float(np.max(val_scores))

    print(f"Normal Validation Mean Score : {mean_val_score:.6f}")
    print(f"Normal Validation Max Score  : {max_val_score:.6f}")
    print(f"Anomaly Threshold ({percentile}th pct): {threshold:.6f}")
    print("-" * 60)

    # Save model checkpoint
    checkpoint_file = output_path / "visionguard_model.pth"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "threshold": threshold,
            "threshold_percentile": percentile,
            "img_size": img_size,
            "category": category,
            "epochs": epochs,
            "final_loss": history[-1],
        },
        checkpoint_file,
    )

    # Save metadata JSON for easy inspection
    metadata_file = output_path / "model_meta.json"
    metadata = {
        "category": category,
        "img_size": img_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "threshold": threshold,
        "threshold_percentile": percentile,
        "val_mean_score": mean_val_score,
        "val_max_score": max_val_score,
        "final_train_loss": history[-1],
        "device": device.type,
    }
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Model checkpoint saved : {checkpoint_file}")
    print(f"Model metadata saved   : {metadata_file}")
    print("=" * 60)
    return metadata


def parse_args():
    parser = argparse.ArgumentParser(description="VisionGuard Autoencoder Training")
    parser.add_argument("--data-dir", type=str, default="data/bottle", help="Path to category dataset directory")
    parser.add_argument("--category", type=str, default="bottle", help="Dataset category name")
    parser.add_argument("--epochs", type=int, default=25, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=128, help="Image resolution (width and height)")
    parser.add_argument("--percentile", type=float, default=95.0, help="Percentile for anomaly threshold (e.g. 95.0 or 99.0)")
    parser.add_argument("--output-dir", type=str, default="checkpoints", help="Output directory for checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--download", action="store_true", help="Download and extract MVTec dataset if missing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_autoencoder(
        data_dir=args.data_dir,
        category=args.category,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        img_size=args.img_size,
        percentile=args.percentile,
        output_dir=args.output_dir,
        seed=args.seed,
        download_if_missing=args.download,
    )
