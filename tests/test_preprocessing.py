"""Unit tests for VisionGuard image preprocessing pipeline."""

from pathlib import Path
import sys
import numpy as np
from PIL import Image
import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing import (
    ImagePreprocessor,
    denormalize_image,
    normalize_image,
    preprocess_image,
    resize_image,
    to_rgb,
)


@pytest.fixture
def sample_pil_images():
    """Fixture providing PIL images of different modes."""
    # RGB
    arr_rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    arr_rgb[20:80, 20:80] = [200, 100, 50]
    img_rgb = Image.fromarray(arr_rgb, mode="RGB")

    # Grayscale
    arr_gray = np.full((100, 100), 128, dtype=np.uint8)
    img_gray = Image.fromarray(arr_gray, mode="L")

    # RGBA
    arr_rgba = np.zeros((100, 100, 4), dtype=np.uint8)
    arr_rgba[:, :, :3] = 180
    arr_rgba[:, :, 3] = 255
    img_rgba = Image.fromarray(arr_rgba, mode="RGBA")

    return {
        "rgb": img_rgb,
        "gray": img_gray,
        "rgba": img_rgba,
    }


def test_to_rgb_modes(sample_pil_images):
    """Test converting grayscale and RGBA images to 3-channel RGB."""
    # From PIL Grayscale
    rgb_from_gray = to_rgb(sample_pil_images["gray"])
    assert rgb_from_gray.mode == "RGB"
    assert len(rgb_from_gray.split()) == 3

    # From PIL RGBA
    rgb_from_rgba = to_rgb(sample_pil_images["rgba"])
    assert rgb_from_rgba.mode == "RGB"

    # From 2D numpy array
    arr_2d = np.ones((50, 50), dtype=np.uint8) * 100
    rgb_from_arr = to_rgb(arr_2d)
    assert rgb_from_arr.mode == "RGB"
    assert rgb_from_arr.size == (50, 50)


def test_resize_image():
    """Test image resizing to square and rectangular dimensions."""
    img = Image.new("RGB", (200, 150), color=(100, 150, 200))

    # Test (height=128, width=128)
    resized_square = resize_image(img, target_size=(128, 128))
    # PIL image.size returns (width, height)
    assert resized_square.size == (128, 128)

    # Test (height=64, width=96)
    resized_rect = resize_image(img, target_size=(64, 96))
    assert resized_rect.size == (96, 64)

    # Test int size
    resized_int = resize_image(img, target_size=100)
    assert resized_int.size == (100, 100)

    # Test invalid size
    with pytest.raises(ValueError):
        resize_image(img, target_size=(0, 100))


def test_normalize_image_zero_one(sample_pil_images):
    """Test normalizing image to [0.0, 1.0] range."""
    img = sample_pil_images["rgb"]

    # As numpy array
    norm_arr = normalize_image(img, range_mode="zero_one", to_tensor=False)
    assert isinstance(norm_arr, np.ndarray)
    assert norm_arr.dtype == np.float32
    assert norm_arr.shape == (100, 100, 3)
    assert norm_arr.min() >= 0.0
    assert norm_arr.max() <= 1.0

    # As torch tensor
    norm_tensor = normalize_image(img, range_mode="zero_one", to_tensor=True)
    assert isinstance(norm_tensor, torch.Tensor)
    assert norm_tensor.dtype == torch.float32
    assert norm_tensor.shape == (3, 100, 100)
    assert norm_tensor.min() >= 0.0
    assert norm_tensor.max() <= 1.0


def test_normalize_image_minus_one_one(sample_pil_images):
    """Test normalizing image to [-1.0, 1.0] range."""
    img = sample_pil_images["rgb"]
    norm_arr = normalize_image(img, range_mode="minus_one_one")
    assert norm_arr.min() >= -1.0
    assert norm_arr.max() <= 1.0


def test_denormalize_image():
    """Test denormalizing [0, 1] array back to [0, 255] uint8 array."""
    orig_norm = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    denorm = denormalize_image(orig_norm, range_mode="zero_one")

    assert denorm.dtype == np.uint8
    assert denorm[0, 0, 0] == 0
    assert 127 <= denorm[0, 0, 1] <= 128
    assert denorm[0, 0, 2] == 255

    # Test with torch tensor input
    tensor_norm = torch.tensor([[[0.0]], [[0.5]], [[1.0]]], dtype=torch.float32)  # (3, 1, 1)
    denorm_from_tensor = denormalize_image(tensor_norm, range_mode="zero_one")
    assert denorm_from_tensor.shape == (1, 1, 3)
    assert denorm_from_tensor.dtype == np.uint8


def test_image_preprocessor_pipeline(tmp_path):
    """Test ImagePreprocessor pipeline with file path, PIL Image, and batch."""
    # Create temp image file
    test_path = tmp_path / "test_pipe.png"
    arr = (np.random.rand(80, 80, 3) * 255).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(test_path)

    preprocessor = ImagePreprocessor(target_size=(128, 128), range_mode="zero_one", to_tensor=True)

    # Process from file path
    tensor_from_path = preprocessor(test_path)
    assert isinstance(tensor_from_path, torch.Tensor)
    assert tensor_from_path.shape == (3, 128, 128)
    assert tensor_from_path.dtype == torch.float32

    # Process batch
    batch_tensors = preprocessor.preprocess_batch([test_path, test_path])
    assert isinstance(batch_tensors, torch.Tensor)
    assert batch_tensors.shape == (2, 3, 128, 128)


def test_preprocess_image_convenience_function():
    """Test preprocess_image convenience function."""
    img = Image.new("RGB", (50, 50), color=(255, 0, 0))
    result = preprocess_image(img, target_size=(64, 64), to_tensor=False)

    assert isinstance(result, np.ndarray)
    assert result.shape == (64, 64, 3)
    assert result.dtype == np.float32
    assert result.max() <= 1.0
