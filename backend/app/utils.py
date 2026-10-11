"""Utility helpers, custom exceptions, and image serialization functions."""

import io
import base64
import numpy as np
from PIL import Image
import cv2


class ModelNotLoadedException(Exception):
    """Raised when an inference request is received but the model checkpoint is not loaded."""
    pass


class InvalidImageException(Exception):
    """Raised when an uploaded image fails validation, decoding, or sizing checks."""
    pass


class InferenceException(Exception):
    """Raised when an error occurs during model forward pass, uncertainty, or explainability."""
    pass


def encode_pil_to_base64_data_uri(image: Image.Image, format: str = "PNG") -> str:
    """Encode a PIL Image object to a Base64 data URI string.

    Args:
        image: PIL Image instance.
        format: Image format (default 'PNG').

    Returns:
        Data URI formatted string: 'data:image/png;base64,...'
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"


def encode_cv2_to_base64_data_uri(image_bgr: np.ndarray, format: str = "PNG") -> str:
    """Encode an OpenCV BGR/RGB image array to a Base64 data URI string.

    Args:
        image_bgr: NumPy array representing image (BGR format for OpenCV).
        format: Image format (default 'PNG').

    Returns:
        Data URI formatted string: 'data:image/png;base64,...'
    """
    ext = f".{format.lower()}"
    success, buffer = cv2.imencode(ext, image_bgr)
    if not success:
        raise ValueError(f"Failed to encode image to {format}")
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/{format.lower()};base64,{encoded}"
