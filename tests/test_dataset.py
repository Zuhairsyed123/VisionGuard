"""Unit tests for VisionGuard dataset and image loading utilities."""

from pathlib import Path
import sys
import numpy as np
from PIL import Image
import pytest
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.dataset import (
    SUPPORTED_IMAGE_EXTENSIONS,
    CorruptedImageError,
    ImageDataset,
    ImageNotFoundError,
    UnsupportedImageFormatError,
    get_image_paths,
    load_image,
    validate_image_path,
)


@pytest.fixture
def temp_image_dir(tmp_path: Path):
    """Fixture creating temporary valid, invalid, and corrupted image files."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    # 1. Valid RGB PNG
    valid_png = img_dir / "sample_valid.png"
    arr_rgb = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
    Image.fromarray(arr_rgb, mode="RGB").save(valid_png)

    # 2. Valid JPEG
    valid_jpg = img_dir / "sample_valid.jpg"
    Image.fromarray(arr_rgb, mode="RGB").save(valid_jpg, format="JPEG")

    # 3. Valid Grayscale PNG
    valid_gray = img_dir / "sample_gray.png"
    arr_gray = (np.random.rand(64, 64) * 255).astype(np.uint8)
    Image.fromarray(arr_gray, mode="L").save(valid_gray)

    # 4. Corrupted file (PNG extension but invalid arbitrary bytes)
    corrupted_png = img_dir / "sample_corrupt.png"
    with open(corrupted_png, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\nGARBAGE_PAYLOAD_NOT_A_VALID_IMAGE_DATA")

    # 5. Unsupported file format (.txt and .pdf)
    unsupported_txt = img_dir / "document.txt"
    unsupported_txt.write_text("Hello, this is plain text.")

    unsupported_pdf = img_dir / "document.pdf"
    unsupported_pdf.write_bytes(b"%PDF-1.4 dummy pdf bytes")

    # 6. Subdirectory with another valid PNG
    sub_dir = img_dir / "subdir"
    sub_dir.mkdir()
    sub_png = sub_dir / "sub_valid.png"
    Image.fromarray(arr_rgb, mode="RGB").save(sub_png)

    return {
        "dir": img_dir,
        "valid_png": valid_png,
        "valid_jpg": valid_jpg,
        "valid_gray": valid_gray,
        "corrupted_png": corrupted_png,
        "unsupported_txt": unsupported_txt,
        "unsupported_pdf": unsupported_pdf,
        "sub_png": sub_png,
    }


def test_validate_image_path_valid(temp_image_dir):
    """Test validating an existing image path with supported extension."""
    valid_path = temp_image_dir["valid_png"]
    resolved = validate_image_path(valid_path)
    assert resolved == valid_path.resolve()
    assert resolved.exists()


def test_validate_image_path_missing(temp_image_dir):
    """Test that validating a non-existent path raises ImageNotFoundError."""
    non_existent = temp_image_dir["dir"] / "does_not_exist.png"
    with pytest.raises(ImageNotFoundError, match="Image file does not exist"):
        validate_image_path(non_existent)


def test_validate_image_path_unsupported_format(temp_image_dir):
    """Test that validating an unsupported file extension raises UnsupportedImageFormatError."""
    unsupported = temp_image_dir["unsupported_txt"]
    with pytest.raises(UnsupportedImageFormatError, match="Unsupported image extension"):
        validate_image_path(unsupported)


def test_load_image_valid_rgb(temp_image_dir):
    """Test loading a valid RGB image returns PIL Image in RGB mode."""
    img = load_image(temp_image_dir["valid_png"], mode="RGB")
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert img.size == (64, 64)


def test_load_image_converts_grayscale_to_rgb(temp_image_dir):
    """Test loading a grayscale image converts to RGB when mode='RGB'."""
    img = load_image(temp_image_dir["valid_gray"], mode="RGB")
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"


def test_load_image_corrupted_raises_error(temp_image_dir):
    """Test that loading a corrupted image file raises CorruptedImageError."""
    corrupted_path = temp_image_dir["corrupted_png"]
    with pytest.raises(CorruptedImageError, match="Failed to decode or parse"):
        load_image(corrupted_path)


def test_get_image_paths_filtering_and_recursion(temp_image_dir):
    """Test directory scanning finds valid images and excludes text/pdf/unsupported files."""
    root_dir = temp_image_dir["dir"]

    # Recursive scan by extension: 5 files with valid extensions (excluding text/pdf)
    paths_recursive = get_image_paths(root_dir, recursive=True, validate_readable=False)
    assert len(paths_recursive) == 5
    for p in paths_recursive:
        assert p.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS

    # Recursive scan with readability verification: excludes corrupted png -> exactly 4 readable images
    paths_readable = get_image_paths(root_dir, recursive=True, validate_readable=True)
    assert len(paths_readable) == 4

    # Non-recursive scan: 4 files in root_dir (excluding sub_png in subdir)
    paths_flat = get_image_paths(root_dir, recursive=False, validate_readable=False)
    assert len(paths_flat) == 4

    # Non-recursive scan with readability check: 3 files in root_dir (excluding corrupt)
    paths_flat_readable = get_image_paths(root_dir, recursive=False, validate_readable=True)
    assert len(paths_flat_readable) == 3


def test_get_image_paths_invalid_directory(tmp_path):
    """Test scanning a non-existent or invalid directory raises FileNotFoundError."""
    missing_dir = tmp_path / "non_existent_folder"
    with pytest.raises(FileNotFoundError):
        get_image_paths(missing_dir)


def test_image_dataset_pytorch(temp_image_dir):
    """Test PyTorch ImageDataset indexing, len, and loading."""
    valid_paths = [temp_image_dir["valid_png"], temp_image_dir["valid_jpg"]]
    dataset = ImageDataset(file_paths=valid_paths, mode="RGB")

    assert len(dataset) == 2

    img, path_str = dataset[0]
    assert isinstance(img, Image.Image)
    assert img.mode == "RGB"
    assert path_str == str(temp_image_dir["valid_png"])


def test_image_dataset_from_directory(temp_image_dir):
    """Test ImageDataset.from_directory constructor with validate_readable."""
    dataset = ImageDataset.from_directory(
        temp_image_dir["dir"],
        recursive=False,
        validate_readable=True,
    )
    assert len(dataset) == 3
