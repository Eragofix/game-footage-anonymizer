"""Lightweight metrics. Perceptual metrics (LPIPS, FID) and temporal warping error come later."""

from __future__ import annotations

import numpy as np

from anonymizer.types import Box


def box_iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def psnr(reference: np.ndarray, test: np.ndarray, max_value: float = 255.0) -> float:
    mse = np.mean((reference.astype(np.float64) - test.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(max_value**2 / mse))
