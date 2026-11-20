"""Anonymization methods: classic (baseline) and generative inpainting."""

from anonymizer.anonymization.classic import gaussian_blur, pixelate, solid_fill

__all__ = ["gaussian_blur", "pixelate", "solid_fill"]
