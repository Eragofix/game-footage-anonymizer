"""Baseline demo: anonymize given boxes on an image with a classic method.

Example:
    python scripts/anonymize_image.py frame.png out.png --box 100 50 220 190 --method pixelate

Detectors are added in stage 1; until then boxes are passed by hand.
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np

from anonymizer.anonymization import gaussian_blur, pixelate, solid_fill
from anonymizer.types import Detection

METHODS = {"gaussian_blur": gaussian_blur, "pixelate": pixelate, "solid": solid_fill}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument(
        "--box", nargs=4, type=int, action="append", required=True, metavar=("X1", "Y1", "X2", "Y2")
    )
    parser.add_argument("--method", choices=METHODS, default="gaussian_blur")
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        raise SystemExit(f"cannot read {args.input}")
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    for box in args.box:
        mask |= Detection(box=tuple(box), score=1.0, label="face").to_mask(h, w)

    cv2.imwrite(args.output, METHODS[args.method](image, mask))


if __name__ == "__main__":
    main()
