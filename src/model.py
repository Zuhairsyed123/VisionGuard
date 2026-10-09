"""VisionGuard Convolutional Autoencoder Model.

A lightweight, modular Convolutional Autoencoder implemented in PyTorch
for unsupervised image anomaly detection.
"""

from typing import Tuple
import torch
import torch.nn as nn


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class ConvAutoencoder(nn.Module):
    """Convolutional Autoencoder for normal image reconstruction.

    Encoder compresses normal images into a low-dimensional latent representation.
    Decoder reconstructs the input image from the latent space.
    When given anomalous images, the model struggles to reconstruct defective
    regions, resulting in high reconstruction error.
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        latent_channels: int = 128,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.latent_channels = latent_channels

        # Encoder: H, W -> H/8, W/8
        self.encoder = nn.Sequential(
            # Stage 1: (B, C, H, W) -> (B, base_channels, H/2, W/2)
            nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 2: (B, base_channels, H/2, W/2) -> (B, base_channels*2, H/4, W/4)
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Stage 3: (B, base_channels*2, H/4, W/4) -> (B, latent_channels, H/8, W/8)
            nn.Conv2d(base_channels * 2, latent_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(latent_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Decoder: H/8, W/8 -> H, W
        self.decoder = nn.Sequential(
            # Stage 1: (B, latent_channels, H/8, W/8) -> (B, base_channels*2, H/4, W/4)
            nn.ConvTranspose2d(latent_channels, base_channels * 2, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),

            # Stage 2: (B, base_channels*2, H/4, W/4) -> (B, base_channels, H/2, W/2)
            nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),

            # Stage 3: (B, base_channels, H/2, W/2) -> (B, in_channels, H, W)
            nn.ConvTranspose2d(base_channels, in_channels, kernel_size=2, stride=2),
            nn.Sigmoid(),  # Bound outputs to [0, 1] range matching normalized pixel values
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input tensor into latent feature maps."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation back to reconstructed image space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run full forward pass: input -> latent -> reconstructed."""
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return reconstruction


def compute_reconstruction_error(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    reduction: str = "none",
) -> torch.Tensor:
    """Compute per-image Mean Squared Error (MSE) between original and reconstructed tensors.

    Args:
        original: Tensor of shape (B, C, H, W) in range [0, 1]
        reconstructed: Tensor of shape (B, C, H, W) in range [0, 1]
        reduction: "none" returns shape (B,), "mean" returns scalar tensor

    Returns:
        Per-sample MSE error or scalar average MSE.
    """
    # Square difference per pixel across all channels
    diff_sq = (original - reconstructed) ** 2
    # Mean per sample over (C, H, W)
    per_sample_mse = diff_sq.view(original.size(0), -1).mean(dim=1)
    if reduction == "mean":
        return per_sample_mse.mean()
    return per_sample_mse
