"""Tests for bin/image_split.py."""

import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

BIN_DIR = os.path.join(os.path.dirname(__file__), "..", "bin")
SCRIPT = os.path.join(BIN_DIR, "image_split.py")


def test_split_creates_correct_number_of_tiles(synthetic_image_500, tmp_dir):
    """A 500x500 image with tile_size=250 should produce 4 tiles."""
    prefix = os.path.join(tmp_dir, "tile")
    result = subprocess.run(
        [sys.executable, SCRIPT,
         "--input", synthetic_image_500,
         "--output-prefix", prefix,
         "--tile-size", "250"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    tiles = [f for f in os.listdir(tmp_dir) if f.startswith("tile") and f.endswith(".png")]
    assert len(tiles) == 4


def test_split_tile_dimensions(synthetic_image_500, tmp_dir):
    """Each tile should be exactly tile_size x tile_size."""
    prefix = os.path.join(tmp_dir, "tile")
    subprocess.run(
        [sys.executable, SCRIPT,
         "--input", synthetic_image_500,
         "--output-prefix", prefix,
         "--tile-size", "250"],
        capture_output=True, text=True,
    )
    for f in os.listdir(tmp_dir):
        if f.startswith("tile") and f.endswith(".png"):
            img = cv2.imread(os.path.join(tmp_dir, f))
            assert img.shape[:2] == (250, 250), f"Tile {f} has wrong shape: {img.shape}"


def test_split_filename_pattern(synthetic_image_500, tmp_dir):
    """Tile filenames should follow {prefix}_{row:04d}_{col:04d}.png."""
    prefix = os.path.join(tmp_dir, "tile")
    subprocess.run(
        [sys.executable, SCRIPT,
         "--input", synthetic_image_500,
         "--output-prefix", prefix,
         "--tile-size", "250"],
        capture_output=True, text=True,
    )
    expected_names = {"tile_0000_0000.png", "tile_0000_0250.png",
                      "tile_0250_0000.png", "tile_0250_0250.png"}
    actual_names = {f for f in os.listdir(tmp_dir) if f.startswith("tile") and f.endswith(".png")}
    assert expected_names == actual_names


def test_split_preserves_pixel_content(synthetic_image_500, tmp_dir):
    """Top-left tile should contain the white region of the source image."""
    prefix = os.path.join(tmp_dir, "tile")
    subprocess.run(
        [sys.executable, SCRIPT,
         "--input", synthetic_image_500,
         "--output-prefix", prefix,
         "--tile-size", "250"],
        capture_output=True, text=True,
    )
    tl_tile = cv2.imread(os.path.join(tmp_dir, "tile_0000_0000.png"))
    # The top-left quadrant of the 500x500 image is all white [255,255,255]
    assert np.all(tl_tile == 255)


def test_split_fails_on_missing_input(tmp_dir):
    """Should exit non-zero when the input file does not exist."""
    prefix = os.path.join(tmp_dir, "tile")
    result = subprocess.run(
        [sys.executable, SCRIPT,
         "--input", "/nonexistent/image.png",
         "--output-prefix", prefix,
         "--tile-size", "250"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0


def test_split_pad_non_divisible(tmp_dir):
    """--pad should zero-pad edge tiles when image is not divisible by tile_size.

    A 500x500 image split into 256x256 tiles produces a 2x2 grid where the
    second row/column would normally be 244px. With --pad, all tiles are 256x256.
    """
    img = np.full((500, 500, 3), 128, dtype=np.uint8)
    img_path = os.path.join(tmp_dir, "img500.png")
    cv2.imwrite(img_path, img)

    prefix = os.path.join(tmp_dir, "pad")
    result = subprocess.run(
        [sys.executable, SCRIPT,
         "--input", img_path,
         "--output-prefix", prefix,
         "--tile-size", "256",
         "--pad"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    tiles = [f for f in os.listdir(tmp_dir) if f.startswith("pad") and f.endswith(".png")]
    assert len(tiles) == 4  # ceil(500/256)^2 = 2x2

    for f in tiles:
        tile = cv2.imread(os.path.join(tmp_dir, f))
        assert tile.shape[:2] == (256, 256), f"Tile {f} has wrong shape: {tile.shape}"

    # Bottom-right tile should have zero-padded region
    br_tile = cv2.imread(os.path.join(tmp_dir, "pad_0256_0256.png"))
    assert np.all(br_tile[244:, :, :] == 0), "Padded rows should be zero"
    assert np.all(br_tile[:, 244:, :] == 0), "Padded cols should be zero"
    assert np.all(br_tile[:244, :244, :] == 128), "Original region should be preserved"


def test_split_grayscale_pad(tmp_dir):
    """--grayscale --pad should produce single-channel padded tiles."""
    img = np.full((500, 500), 200, dtype=np.uint8)
    img_path = os.path.join(tmp_dir, "gray500.png")
    cv2.imwrite(img_path, img)

    prefix = os.path.join(tmp_dir, "gp")
    result = subprocess.run(
        [sys.executable, SCRIPT,
         "--input", img_path,
         "--output-prefix", prefix,
         "--tile-size", "256",
         "--grayscale", "--pad"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"

    for f in os.listdir(tmp_dir):
        if f.startswith("gp") and f.endswith(".png"):
            tile = cv2.imread(os.path.join(tmp_dir, f), cv2.IMREAD_GRAYSCALE)
            assert tile.shape == (256, 256), f"Tile {f} has wrong shape: {tile.shape}"
