"""Shared pytest fixtures for S2 Segmentation Workflow tests."""

import os
import tempfile

import cv2
import numpy as np
import pytest


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temporary directory for test outputs."""
    return str(tmp_path)


@pytest.fixture
def synthetic_image_250(tmp_path):
    """Create a 250x250 RGB PNG test image with distinct color regions."""
    img = np.zeros((250, 250, 3), dtype=np.uint8)
    img[:125, :125] = [255, 255, 255]   # white (ice)
    img[:125, 125:] = [100, 100, 100]   # gray (thin ice)
    img[125:, :] = [10, 10, 30]         # dark (water)
    path = str(tmp_path / "test_250.png")
    cv2.imwrite(path, img)
    return path


@pytest.fixture
def synthetic_image_500(tmp_path):
    """Create a 500x500 RGB PNG (splits into 4 tiles at tile_size=250)."""
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    img[:250, :250] = [255, 255, 255]
    img[:250, 250:] = [100, 100, 100]
    img[250:, :250] = [10, 10, 30]
    img[250:, 250:] = [200, 200, 200]
    path = str(tmp_path / "test_500.png")
    cv2.imwrite(path, img)
    return path


@pytest.fixture
def synthetic_training_data(tmp_path):
    """Create synthetic 256x256 training images and masks (4 samples)."""
    img_dir = tmp_path / "train_images"
    mask_dir = tmp_path / "train_masks"
    img_dir.mkdir()
    mask_dir.mkdir()

    rng = np.random.RandomState(42)
    for i in range(4):
        train = rng.randint(0, 255, (256, 256), dtype=np.uint8)
        cv2.imwrite(str(img_dir / f"img_{i:02d}.png"), train)
        mask = rng.choice([0, 128, 255], size=(256, 256)).astype(np.uint8)
        cv2.imwrite(str(mask_dir / f"mask_{i:02d}.png"), mask)

    return str(img_dir), str(mask_dir)
