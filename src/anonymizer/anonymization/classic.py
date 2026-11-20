"""Classic anonymization baselines: Gaussian blur, pixelation, solid fill.

All functions take a BGR uint8 image and a boolean mask and return a new image;
pixels outside the mask are left untouched.
"""

from __future__ import annotations

import cv2
import numpy as np


def _apply(image: np.ndarray, mask: np.ndarray, replacement: np.ndarray) -> np.ndarray:
    if image.shape[:2] != mask.shape:
        raise ValueError(f"mask shape {mask.shape} does not match image {image.shape[:2]}")
    out = image.copy()
    out[mask] = replacement[mask]
    return out


def gaussian_blur(image: np.ndarray, mask: np.ndarray, kernel_size: int = 51) -> np.ndarray:
    """Blur the masked region. ``kernel_size`` is forced to be odd."""
    k = kernel_size | 1
    return _apply(image, mask, cv2.GaussianBlur(image, (k, k), 0))


def pixelate(image: np.ndarray, mask: np.ndarray, pixel_size: int = 16) -> np.ndarray:
    """Replace the masked region with blocks of ``pixel_size`` x ``pixel_size`` pixels."""
    h, w = image.shape[:2]
    small = cv2.resize(
        image, (max(1, w // pixel_size), max(1, h // pixel_size)), interpolation=cv2.INTER_AREA
    )
    blocks = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    return _apply(image, mask, blocks)


def solid_fill(
    image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """Cover the masked region with a solid color (black box by default)."""
    return _apply(image, mask, np.full_like(image, color, dtype=np.uint8))
