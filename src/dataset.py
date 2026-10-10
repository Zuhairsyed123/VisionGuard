"""VisionGuard Dataset and Image-Loading Utilities.

Provides robust image validation, format verification, error handling for missing
or corrupted files, and directory scanning utilities.
"""

from pathlib import Path
from typing import List, Optional, Sequence, Set, Tuple, Union
import numpy as np
from PIL import Image, UnidentifiedImageError
import torch
from torch.utils.data import Dataset


# Supported image formats for VisionGuard computer-vision pipeline
SUPPORTED_IMAGE_EXTENSIONS: Set[str] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tiff",
    ".webp",
}


class VisionGuardError(Exception):
    """Base exception for all VisionGuard errors."""
    pass


class ImageNotFoundError(VisionGuardError, FileNotFoundError):
    """Raised when an image file does not exist on the filesystem."""
    pass


class UnsupportedImageFormatError(VisionGuardError, ValueError):
    """Raised when an image format/extension is not supported."""
    pass


class CorruptedImageError(VisionGuardError, IOError):
    """Raised when an image file cannot be decoded or is corrupted."""
    pass


def validate_image_path(
    path: Union[str, Path],
    supported_extensions: Optional[Set[str]] = None,
) -> Path:
    """Validate that an image path exists and has an approved file extension.

    Args:
        path: Path to the image file.
        supported_extensions: Optional custom set of lowercase extensions (e.g. {'.png', '.jpg'}).
            Defaults to SUPPORTED_IMAGE_EXTENSIONS.

    Returns:
        Resolved Path object.

    Raises:
        ImageNotFoundError: If the file does not exist.
        UnsupportedImageFormatError: If the extension is not in supported_extensions.
    """
    img_path = Path(path)
    if not img_path.is_file():
        raise ImageNotFoundError(f"Image file does not exist: '{path}'")

    valid_exts = supported_extensions or SUPPORTED_IMAGE_EXTENSIONS
    ext = img_path.suffix.lower()
    if ext not in valid_exts:
        sorted_exts = sorted(list(valid_exts))
        raise UnsupportedImageFormatError(
            f"Unsupported image extension '{img_path.suffix}'. "
            f"Supported extensions are: {sorted_exts}"
        )

    return img_path.resolve()


def load_image(
    path: Union[str, Path],
    mode: Optional[str] = "RGB",
    supported_extensions: Optional[Set[str]] = None,
) -> Image.Image:
    """Safely load an image from disk with validation and corruption checks.

    PIL's Image.open performs lazy header decoding. To ensure the image is not
    truncated or corrupted, this function forces pixel rasterization via load().

    Args:
        path: Path to the image file.
        mode: Target color mode, typically 'RGB' or 'L' (Grayscale). If None, keeps original.
        supported_extensions: Optional set of allowed file extensions.

    Returns:
        Loaded and validated PIL Image.

    Raises:
        ImageNotFoundError: If file does not exist.
        UnsupportedImageFormatError: If extension is unsupported.
        CorruptedImageError: If the file content is corrupt or cannot be decoded.
    """
    valid_path = validate_image_path(path, supported_extensions=supported_extensions)

    try:
        with Image.open(valid_path) as img:
            # Force rasterization to detect truncated or corrupted byte streams
            img.load()
            # Convert to target mode if requested
            if mode is not None and img.mode != mode:
                converted_img = img.convert(mode)
                return converted_img
            # Return a copy detached from the file handle
            return img.copy()
    except (UnidentifiedImageError, OSError, IOError) as exc:
        raise CorruptedImageError(
            f"Failed to decode or parse corrupted image '{valid_path}': {exc}"
        ) from exc


def get_image_paths(
    directory: Union[str, Path],
    recursive: bool = True,
    supported_extensions: Optional[Set[str]] = None,
    validate_readable: bool = False,
) -> List[Path]:
    """Scan a directory for all valid image files.

    Args:
        directory: Root directory path to scan.
        recursive: Whether to search subdirectories recursively.
        supported_extensions: Optional set of extensions to include.
        validate_readable: If True, attempts to open each file to ensure it is not corrupt.
            Corrupted files are excluded.

    Returns:
        Sorted list of Path objects pointing to valid image files.

    Raises:
        FileNotFoundError: If the directory does not exist.
        NotADirectoryError: If the path is not a directory.
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: '{directory}'")
    if not dir_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: '{directory}'")

    valid_exts = supported_extensions or SUPPORTED_IMAGE_EXTENSIONS
    iterator = dir_path.rglob("*") if recursive else dir_path.glob("*")

    image_paths: List[Path] = [
        p.resolve()
        for p in iterator
        if p.is_file() and p.suffix.lower() in valid_exts
    ]

    if validate_readable:
        readable_paths: List[Path] = []
        for p in image_paths:
            try:
                load_image(p)
                readable_paths.append(p)
            except CorruptedImageError:
                continue
        image_paths = readable_paths

    image_paths.sort()
    return image_paths


class ImageDataset(Dataset):
    """PyTorch Dataset for loading images from disk with validation."""

    def __init__(
        self,
        file_paths: Sequence[Union[str, Path]],
        transform=None,
        mode: str = "RGB",
    ) -> None:
        """Initialize dataset.

        Args:
            file_paths: Sequence of image file paths.
            transform: Optional torchvision or callable transform.
            mode: Color mode for loading ('RGB', 'L', etc.).
        """
        self.file_paths = [Path(p) for p in file_paths]
        self.transform = transform
        self.mode = mode

    @classmethod
    def from_directory(
        cls,
        directory: Union[str, Path],
        recursive: bool = True,
        transform=None,
        mode: str = "RGB",
        validate_readable: bool = False,
    ) -> "ImageDataset":
        """Construct an ImageDataset by scanning a directory."""
        paths = get_image_paths(directory, recursive=recursive, validate_readable=validate_readable)
        return cls(file_paths=paths, transform=transform, mode=mode)

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[Union[Image.Image, torch.Tensor], str]:
        path = self.file_paths[idx]
        image = load_image(path, mode=self.mode)
        if self.transform is not None:
            image = self.transform(image)
        return image, str(path)
