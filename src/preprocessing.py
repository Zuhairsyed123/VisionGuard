"""VisionGuard Image Preprocessing Pipeline.

Provides modular image transformations: RGB conversion, resizing, and normalization
to ensure consistent numerical inputs for computer vision and anomaly detection models.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union
import numpy as np
from PIL import Image
import torch

from src.dataset import load_image


# Standard ImageNet mean and std for deep feature extractors
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def to_rgb(image: Union[Image.Image, np.ndarray]) -> Image.Image:
    """Convert an image (PIL or numpy) from any color format to standard 3-channel RGB.

    Handles:
        - Grayscale ('L' or single-channel 2D array)
        - RGBA (composited over a clean white background or alpha stripped)
        - CMYK / Palette ('P')
        - NumPy arrays in (H, W), (H, W, 1), (H, W, 3), (H, W, 4)

    Args:
        image: PIL Image or NumPy array.

    Returns:
        RGB PIL Image.
    """
    if isinstance(image, np.ndarray):
        # Handle numpy array formats
        if image.ndim == 2:
            # Grayscale 2D array
            if image.dtype in (np.float32, np.float64) and image.max() <= 1.0:
                image = (image * 255.0).clip(0, 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
            pil_img = Image.fromarray(image, mode="L").convert("RGB")
            return pil_img
        elif image.ndim == 3:
            h, w, c = image.shape
            if image.dtype in (np.float32, np.float64) and image.max() <= 1.0:
                uint8_arr = (image * 255.0).clip(0, 255).astype(np.uint8)
            else:
                uint8_arr = image.astype(np.uint8)

            if c == 1:
                return Image.fromarray(uint8_arr.squeeze(-1), mode="L").convert("RGB")
            elif c == 3:
                return Image.fromarray(uint8_arr, mode="RGB")
            elif c == 4:
                # RGBA
                rgba_img = Image.fromarray(uint8_arr, mode="RGBA")
                return to_rgb(rgba_img)
            else:
                raise ValueError(f"Unsupported number of channels in array: {c}")
        else:
            raise ValueError(f"NumPy array must be 2D or 3D, got ndim={image.ndim}")

    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image or np.ndarray, got {type(image)}")

    if image.mode == "RGB":
        return image.copy()

    if image.mode == "RGBA":
        # Composite against white background to handle semi-transparent pixels properly
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background

    # For other modes (L, CMYK, P, 1), convert directly to RGB
    return image.convert("RGB")


def resize_image(
    image: Image.Image,
    target_size: Union[int, Tuple[int, int]] = (128, 128),
    resample: Image.Resampling = Image.Resampling.BILINEAR,
) -> Image.Image:
    """Resize PIL image to target dimensions.

    Args:
        image: PIL Image to resize.
        target_size: Target dimensions as (height, width) or int for square (size, size).
        resample: Interpolation filter (default: BILINEAR).

    Returns:
        Resized PIL Image.
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image, got {type(image)}")

    if isinstance(target_size, int):
        size = (target_size, target_size)
    elif isinstance(target_size, (tuple, list)) and len(target_size) == 2:
        size = (int(target_size[0]), int(target_size[1]))
    else:
        raise ValueError(f"target_size must be an int or a 2-tuple (H, W), got {target_size}")

    if size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"Target size dimensions must be positive, got {size}")

    # PIL resize takes (width, height)
    # By convention in computer vision, size is given as (height, width) or (H, W)
    # We map (height, width) -> PIL.resize((width, height))
    target_h, target_w = size
    return image.resize((target_w, target_h), resample=resample)


def normalize_image(
    image: Union[Image.Image, np.ndarray],
    range_mode: str = "zero_one",
    to_tensor: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """Normalize image pixel values into a specified numerical range.

    Args:
        image: PIL Image or NumPy array.
        range_mode:
            - 'zero_one': scales [0, 255] to [0.0, 1.0] (default for autoencoders)
            - 'minus_one_one': scales [0, 255] to [-1.0, 1.0]
            - 'imagenet': scales [0, 1] then subtracts ImageNet mean and divides by std
        to_tensor: If True, returns torch.FloatTensor in (C, H, W) format.
            If False, returns np.ndarray in (H, W, C) format with dtype float32.

    Returns:
        Normalized np.ndarray (H, W, C) or torch.Tensor (C, H, W).
    """
    if isinstance(image, Image.Image):
        arr = np.array(image, dtype=np.float32)
    elif isinstance(image, np.ndarray):
        arr = image.astype(np.float32)
    else:
        raise TypeError(f"Expected PIL Image or np.ndarray, got {type(image)}")

    # Ensure array is in [0, 255] base if it was integer
    if arr.max() > 1.0:
        arr = arr / 255.0

    # Ensure 3 channels
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=-1)

    # Apply selected range mode
    if range_mode == "zero_one":
        norm_arr = np.clip(arr, 0.0, 1.0)
    elif range_mode == "minus_one_one":
        norm_arr = np.clip((arr * 2.0) - 1.0, -1.0, 1.0)
    elif range_mode == "imagenet":
        norm_arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    else:
        raise ValueError(
            f"Unknown range_mode '{range_mode}'. Supported modes: 'zero_one', 'minus_one_one', 'imagenet'"
        )

    norm_arr = norm_arr.astype(np.float32)

    if to_tensor:
        # Convert (H, W, C) -> (C, H, W) torch tensor
        tensor = torch.from_numpy(norm_arr).permute(2, 0, 1)
        return tensor

    return norm_arr


def denormalize_image(
    data: Union[np.ndarray, torch.Tensor],
    range_mode: str = "zero_one",
) -> np.ndarray:
    """Reverse normalization and convert back to uint8 [0, 255] format.

    Args:
        data: Normalized array (H, W, C) or torch.Tensor (C, H, W).
        range_mode: Range used during normalization ('zero_one', 'minus_one_one', 'imagenet').

    Returns:
        NumPy uint8 array of shape (H, W, C).
    """
    if isinstance(data, torch.Tensor):
        # If tensor of shape (C, H, W)
        if data.ndim == 3:
            arr = data.detach().cpu().permute(1, 2, 0).numpy()
        elif data.ndim == 4 and data.size(0) == 1:
            arr = data.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
        else:
            arr = data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        arr = data.copy()
    else:
        raise TypeError(f"Expected np.ndarray or torch.Tensor, got {type(data)}")

    if range_mode == "zero_one":
        unscaled = arr * 255.0
    elif range_mode == "minus_one_one":
        unscaled = ((arr + 1.0) / 2.0) * 255.0
    elif range_mode == "imagenet":
        unscaled = ((arr * IMAGENET_STD) + IMAGENET_MEAN) * 255.0
    else:
        raise ValueError(f"Unknown range_mode '{range_mode}'")

    return np.clip(unscaled, 0.0, 255.0).astype(np.uint8)


class ImagePreprocessor:
    """Configurable image preprocessing pipeline for VisionGuard."""

    def __init__(
        self,
        target_size: Union[int, Tuple[int, int]] = (128, 128),
        range_mode: str = "zero_one",
        to_tensor: bool = False,
        resample: Image.Resampling = Image.Resampling.BILINEAR,
    ) -> None:
        """Initialize preprocessor.

        Args:
            target_size: Desired output dimensions as (height, width) or int.
            range_mode: Numerical scaling mode ('zero_one', 'minus_one_one', 'imagenet').
            to_tensor: If True, output is a torch.FloatTensor (C, H, W).
                       If False, output is a float32 np.ndarray (H, W, C).
            resample: Interpolation method for resizing.
        """
        self.target_size = (target_size, target_size) if isinstance(target_size, int) else tuple(target_size)
        self.range_mode = range_mode
        self.to_tensor = to_tensor
        self.resample = resample

    def __call__(
        self,
        input_data: Union[str, Path, Image.Image, np.ndarray],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply pipeline: load (if path) -> to_rgb -> resize -> normalize."""
        return self.preprocess(input_data)

    def preprocess(
        self,
        input_data: Union[str, Path, Image.Image, np.ndarray],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Process a single image item through the pipeline."""
        if isinstance(input_data, (str, Path)):
            pil_img = load_image(input_data, mode="RGB")
        else:
            pil_img = to_rgb(input_data)

        resized_img = resize_image(pil_img, target_size=self.target_size, resample=self.resample)
        normalized = normalize_image(resized_img, range_mode=self.range_mode, to_tensor=self.to_tensor)
        return normalized

    def preprocess_batch(
        self,
        inputs: Sequence[Union[str, Path, Image.Image, np.ndarray]],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Process a sequence of image items and stack into a batch.

        Returns:
            If to_tensor=True: torch.Tensor of shape (B, C, H, W)
            If to_tensor=False: np.ndarray of shape (B, H, W, C)
        """
        if len(inputs) == 0:
            raise ValueError("Cannot preprocess an empty batch")

        processed_list = [self.preprocess(item) for item in inputs]

        if self.to_tensor:
            return torch.stack(processed_list, dim=0)
        else:
            return np.stack(processed_list, axis=0)


def preprocess_image(
    image_or_path: Union[str, Path, Image.Image, np.ndarray],
    target_size: Union[int, Tuple[int, int]] = (128, 128),
    range_mode: str = "zero_one",
    to_tensor: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """Convenience function to preprocess a single image."""
    pipeline = ImagePreprocessor(target_size=target_size, range_mode=range_mode, to_tensor=to_tensor)
    return pipeline(image_or_path)
