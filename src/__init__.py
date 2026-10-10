"""VisionGuard - Visual Anomaly Detection System."""

__version__ = "0.1.0"

# Dataset and I/O utilities
from src.dataset import (
    SUPPORTED_IMAGE_EXTENSIONS,
    CorruptedImageError,
    ImageDataset,
    ImageNotFoundError,
    UnsupportedImageFormatError,
    VisionGuardError,
    get_image_paths,
    load_image,
    validate_image_path,
)

# Preprocessing pipeline
from src.preprocessing import (
    ImagePreprocessor,
    denormalize_image,
    normalize_image,
    preprocess_image,
    resize_image,
    to_rgb,
)

# Baseline difference scoring
from src.baseline import (
    BaselineAnomalyDetector,
    BaselineAnomalyResult,
    compute_pixel_difference,
    score_image_difference,
)

# Autoencoder model and error computation
from src.model import ConvAutoencoder, compute_reconstruction_error

__all__ = [
    "__version__",
    "SUPPORTED_IMAGE_EXTENSIONS",
    "VisionGuardError",
    "ImageNotFoundError",
    "UnsupportedImageFormatError",
    "CorruptedImageError",
    "validate_image_path",
    "load_image",
    "get_image_paths",
    "ImageDataset",
    "to_rgb",
    "resize_image",
    "normalize_image",
    "denormalize_image",
    "ImagePreprocessor",
    "preprocess_image",
    "BaselineAnomalyDetector",
    "BaselineAnomalyResult",
    "compute_pixel_difference",
    "score_image_difference",
    "ConvAutoencoder",
    "compute_reconstruction_error",
]
