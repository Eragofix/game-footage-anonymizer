"""Shared data types: the contract between detection, segmentation and anonymization.

Conventions (keep them stable, every module relies on them):
- Images are ``np.ndarray`` of shape (H, W, 3), dtype uint8, **BGR** (OpenCV order).
- Masks are ``np.ndarray`` of shape (H, W), dtype bool; True = pixel to anonymize.
- Boxes are (x1, y1, x2, y2) in pixel coordinates, x2/y2 exclusive.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Box = tuple[int, int, int, int]


@dataclass
class Detection:
    """A single detected object (face or person) on one frame."""

    box: Box
    score: float
    label: str  # "face" | "person" | "text"
    mask: np.ndarray | None = None  # (H, W) bool, filled by segmentation
    track_id: int | None = None  # filled by video tracking

    def to_mask(self, height: int, width: int) -> np.ndarray:
        """Return the instance mask, falling back to the bounding box."""
        if self.mask is not None:
            return self.mask
        mask = np.zeros((height, width), dtype=bool)
        x1, y1, x2, y2 = clip_box(self.box, height, width)
        mask[y1:y2, x1:x2] = True
        return mask


def clip_box(box: Box, height: int, width: int) -> Box:
    x1, y1, x2, y2 = box
    return max(0, x1), max(0, y1), min(width, x2), min(height, y2)
