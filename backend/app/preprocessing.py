"""Image preprocessing pipeline matching notebook specifications."""

import io
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image, UnidentifiedImageError

import torch
from torchvision import transforms

from .config import settings
from .utils import InvalidImageException


def get_inference_transforms() -> transforms.Compose:
    """Build torchvision transform pipeline.

    Replicates 'eval_tf' from Cell 7 of the research notebook:
    - Resize to (224, 224)
    - ToTensor (scales pixel values to [0.0, 1.0])
    - Normalize with ImageNet mean and std
    """
    return transforms.Compose([
        transforms.Resize((settings.IMG_SIZE, settings.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=settings.IMAGENET_MEAN,
            std=settings.IMAGENET_STD
        )
    ])


def validate_and_read_image(
    file_bytes: bytes,
    filename: Optional[str] = None
) -> Image.Image:
    """Safely validate, decode, and convert an uploaded chest X-ray image.

    Args:
        file_bytes: Raw binary content of the uploaded file.
        filename: Optional filename for extension verification.

    Returns:
        PIL.Image.Image in RGB mode.

    Raises:
        InvalidImageException: If the file is empty, corrupted, oversized,
                               or has an unsupported format.
    """
    if not file_bytes or len(file_bytes) == 0:
        raise InvalidImageException("Uploaded file is empty. Please provide a valid chest X-ray image.")

    if len(file_bytes) > settings.MAX_UPLOAD_SIZE_BYTES:
        max_mb = settings.MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
        raise InvalidImageException(
            f"File size exceeds maximum allowed limit of {max_mb:.1f} MB."
        )

    if filename:
        ext = Path(filename).suffix.lower()
        if ext and ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
            allowed = ", ".join(settings.ALLOWED_IMAGE_EXTENSIONS)
            raise InvalidImageException(
                f"Unsupported file extension '{ext}'. Allowed extensions are: {allowed}"
            )

    try:
        # Validate integrity using PIL verify()
        byte_stream = io.BytesIO(file_bytes)
        with Image.open(byte_stream) as img:
            img.verify()

        # Re-open after verify() to load image data into memory safely
        byte_stream.seek(0)
        image = Image.open(byte_stream)

        # Confirm image format is supported
        if image.format and image.format.lower() not in ["jpeg", "jpg", "png"]:
            raise InvalidImageException(
                f"Unsupported image format '{image.format}'. Only JPEG and PNG formats are supported."
            )

        # Convert to RGB (chest X-rays may be 8-bit grayscale 'L' or 16-bit 'I;16')
        rgb_image = image.convert("RGB")
        # Force load pixel data
        rgb_image.load()
        return rgb_image

    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise InvalidImageException(
            f"Could not decode image file. File may be corrupted or not a valid image: {str(exc)}"
        )


def preprocess_image(
    image: Image.Image,
    device: torch.device
) -> Tuple[torch.Tensor, Image.Image]:
    """Apply the notebook's evaluation transforms and prepare batch tensor.

    Args:
        image: RGB PIL image.
        device: Torch compute device (CPU or CUDA).

    Returns:
        Tuple of (input_tensor of shape (1, 3, 224, 224) on target device, resized PIL image).
    """
    transform = get_inference_transforms()
    tensor = transform(image)
    tensor = tensor.unsqueeze(0).to(device)

    # Also keep a resized PIL image for pixel-aligned overlays
    resized_pil = image.resize((settings.IMG_SIZE, settings.IMG_SIZE))

    return tensor, resized_pil
