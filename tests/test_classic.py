import numpy as np
import pytest

from anonymizer.anonymization import gaussian_blur, pixelate, solid_fill
from anonymizer.evaluation import box_iou, mask_iou, psnr
from anonymizer.types import Detection


@pytest.fixture
def image():
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(64, 80, 3), dtype=np.uint8)


@pytest.fixture
def mask():
    return Detection(box=(10, 10, 40, 30), score=1.0, label="face").to_mask(64, 80)


@pytest.mark.parametrize("method", [gaussian_blur, pixelate, solid_fill])
def test_only_masked_region_changes(method, image, mask):
    out = method(image, mask)
    assert out.shape == image.shape and out.dtype == np.uint8
    assert np.array_equal(out[~mask], image[~mask])
    assert not np.array_equal(out[mask], image[mask])


def test_solid_fill_is_black(image, mask):
    assert (solid_fill(image, mask)[mask] == 0).all()


def test_mask_shape_mismatch_raises(image):
    with pytest.raises(ValueError):
        gaussian_blur(image, np.zeros((10, 10), dtype=bool))


def test_box_clipped_to_image():
    m = Detection(box=(-5, -5, 5, 5), score=1.0, label="face").to_mask(10, 10)
    assert m.sum() == 25


def test_metrics():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    a = np.zeros((4, 4), bool)
    a[:2] = True
    assert mask_iou(a, a) == 1.0
    img = np.zeros((4, 4, 3), np.uint8)
    assert psnr(img, img) == float("inf")
