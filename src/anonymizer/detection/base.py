from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from anonymizer.types import Detection


class Detector(ABC):
    """Common interface for face/person detectors (RetinaFace, YOLO, Mask R-CNN, ...)."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[Detection]:
        """Detect objects on a BGR uint8 image."""
