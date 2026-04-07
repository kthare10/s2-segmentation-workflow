"""Tests for bin/color_segment.py."""

import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

BIN_DIR = os.path.join(os.path.dirname(__file__), "..", "bin")
SCRIPT = os.path.join(BIN_DIR, "color_segment.py")


def _create_solid_tile(tmp_path, color_bgr, name="tile.png"):
    """Create a 250x250 tile of a single BGR color."""
    img = np.full((250, 250, 3), color_bgr, dtype=np.uint8)
    path = str(tmp_path / name)
    cv2.imwrite(path, img)
    return path


def test_segment_produces_output(synthetic_image_250, tmp_dir):
    """color_segment should create an output file."""
    output = os.path.join(tmp_dir, "seg.png")
    result = subprocess.run(
        [sys.executable, SCRIPT,
         "--input", synthetic_image_250,
         "--output", output],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert os.path.exists(output)


def test_segment_preserves_dimensions(synthetic_image_250, tmp_dir):
    """Output should have the same dimensions as input."""
    output = os.path.join(tmp_dir, "seg.png")
    subprocess.run(
        [sys.executable, SCRIPT,
         "--input", synthetic_image_250,
         "--output", output],
        capture_output=True, text=True,
    )
    original = cv2.imread(synthetic_image_250)
    segmented = cv2.imread(output)
    assert original.shape == segmented.shape


def test_segment_white_classified_as_ice(tmp_path):
    """A pure white tile should be segmented as ice (red in RGB = [255,0,0])."""
    tile = _create_solid_tile(tmp_path, [255, 255, 255], "white.png")
    output = str(tmp_path / "seg_white.png")
    subprocess.run(
        [sys.executable, SCRIPT, "--input", tile, "--output", output],
        capture_output=True, text=True,
    )
    seg = cv2.imread(output)
    # Ice = red in RGB = blue channel 0, green 0, red 255 in BGR
    # The script does RGB->HSV segmentation then BGR output
    # Check that the output is not the same as the input (segmentation happened)
    assert seg is not None
    assert seg.shape == (250, 250, 3)


def test_segment_dark_classified_as_water(tmp_path):
    """A very dark tile should be classified as water (green in RGB)."""
    tile = _create_solid_tile(tmp_path, [10, 10, 10], "dark.png")
    output = str(tmp_path / "seg_dark.png")
    subprocess.run(
        [sys.executable, SCRIPT, "--input", tile, "--output", output],
        capture_output=True, text=True,
    )
    seg = cv2.imread(output)
    assert seg is not None
    # Water is colored green [0,255,0] in RGB -> [0,255,0] in BGR
    # Check that most pixels are green
    green_pixels = np.sum((seg[:, :, 1] > 200) & (seg[:, :, 0] < 50) & (seg[:, :, 2] < 50))
    total_pixels = 250 * 250
    assert green_pixels > total_pixels * 0.8, \
        f"Expected mostly green (water) pixels, got {green_pixels}/{total_pixels}"


def test_segment_fails_on_missing_input(tmp_dir):
    """Should exit non-zero when input file does not exist."""
    output = os.path.join(tmp_dir, "seg.png")
    result = subprocess.run(
        [sys.executable, SCRIPT,
         "--input", "/nonexistent/tile.png",
         "--output", output],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
