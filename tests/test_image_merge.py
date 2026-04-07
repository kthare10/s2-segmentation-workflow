"""Tests for bin/image_merge.py."""

import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

BIN_DIR = os.path.join(os.path.dirname(__file__), "..", "bin")
SPLIT_SCRIPT = os.path.join(BIN_DIR, "image_split.py")
MERGE_SCRIPT = os.path.join(BIN_DIR, "image_merge.py")


def _split_image(image_path, tmp_dir, prefix, tile_size):
    """Helper: split an image into tiles and return tile paths."""
    subprocess.run(
        [sys.executable, SPLIT_SCRIPT,
         "--input", image_path,
         "--output-prefix", os.path.join(tmp_dir, prefix),
         "--tile-size", str(tile_size)],
        capture_output=True, text=True, check=True,
    )
    return sorted([
        os.path.join(tmp_dir, f) for f in os.listdir(tmp_dir)
        if f.startswith(prefix) and f.endswith(".png")
    ])


def test_merge_reconstructs_original(synthetic_image_500, tmp_dir):
    """Split then merge should reconstruct the original image."""
    tiles = _split_image(synthetic_image_500, tmp_dir, "tile", 250)
    assert len(tiles) == 4

    merged_path = os.path.join(tmp_dir, "merged.png")
    cmd = [sys.executable, MERGE_SCRIPT]
    for t in tiles:
        cmd.extend(["--input", t])
    cmd.extend(["--output", merged_path, "--tile-size", "250", "--original-size", "500"])

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert os.path.exists(merged_path)

    original = cv2.imread(synthetic_image_500)
    merged = cv2.imread(merged_path)
    assert original.shape == merged.shape
    assert np.array_equal(original, merged), "Merged image should match the original"


def test_merge_output_dimensions(synthetic_image_500, tmp_dir):
    """Merged output should have the specified original-size dimensions."""
    tiles = _split_image(synthetic_image_500, tmp_dir, "tile", 250)
    merged_path = os.path.join(tmp_dir, "merged.png")
    cmd = [sys.executable, MERGE_SCRIPT]
    for t in tiles:
        cmd.extend(["--input", t])
    cmd.extend(["--output", merged_path, "--tile-size", "250", "--original-size", "500"])
    subprocess.run(cmd, capture_output=True, text=True, check=True)

    merged = cv2.imread(merged_path)
    assert merged.shape[:2] == (500, 500)


def test_merge_fails_with_no_valid_tiles(tmp_dir):
    """Should fail when no tiles match the expected filename pattern."""
    bad_tile = os.path.join(tmp_dir, "bad_name.png")
    img = np.zeros((250, 250, 3), dtype=np.uint8)
    cv2.imwrite(bad_tile, img)

    merged_path = os.path.join(tmp_dir, "merged.png")
    result = subprocess.run(
        [sys.executable, MERGE_SCRIPT,
         "--input", bad_tile,
         "--output", merged_path,
         "--tile-size", "250",
         "--original-size", "500"],
        capture_output=True, text=True,
    )
    assert result.returncode != 0


def test_merge_handles_subset_of_tiles(synthetic_image_500, tmp_dir):
    """Merging with fewer tiles should still produce an image (with black gaps)."""
    tiles = _split_image(synthetic_image_500, tmp_dir, "tile", 250)
    # Only use first 2 tiles
    merged_path = os.path.join(tmp_dir, "partial.png")
    cmd = [sys.executable, MERGE_SCRIPT]
    for t in tiles[:2]:
        cmd.extend(["--input", t])
    cmd.extend(["--output", merged_path, "--tile-size", "250", "--original-size", "500"])

    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0
    merged = cv2.imread(merged_path)
    assert merged.shape[:2] == (500, 500)
