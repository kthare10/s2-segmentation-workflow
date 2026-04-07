"""End-to-end integration test: runs the full Stage 1 pipeline and verifies outputs.

This test mirrors run_manual.sh but with pytest assertions. Stage 2 (training)
is tested separately in test_train_unet.py and test_evaluate_model.py since
it requires more time and resources.
"""

import os
import subprocess
import sys

import cv2
import numpy as np
import pytest

BIN_DIR = os.path.join(os.path.dirname(__file__), "..", "bin")


@pytest.fixture
def stage1_pipeline_data(tmp_path):
    """Create a 500x500 test image for the full Stage 1 pipeline."""
    img = np.zeros((500, 500, 3), dtype=np.uint8)
    img[:250, :250] = [255, 255, 255]   # white (ice)
    img[:250, 250:] = [100, 100, 100]   # gray (thin ice)
    img[250:, :250] = [10, 10, 30]      # dark (water)
    img[250:, 250:] = [200, 200, 200]   # light gray
    path = str(tmp_path / "test_image.png")
    cv2.imwrite(path, img)
    return path, tmp_path


def test_full_stage1_pipeline(stage1_pipeline_data):
    """Run split → segment → merge and verify the final merged output."""
    img_path, work_dir = stage1_pipeline_data
    tile_size = 250
    original_size = 500

    # Step 1: Split
    prefix = str(work_dir / "tile")
    result = subprocess.run(
        [sys.executable, os.path.join(BIN_DIR, "image_split.py"),
         "--input", img_path,
         "--output-prefix", prefix,
         "--tile-size", str(tile_size)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"Split failed: {result.stderr}"

    tiles = sorted([
        str(work_dir / f) for f in os.listdir(work_dir)
        if f.startswith("tile") and f.endswith(".png")
    ])
    assert len(tiles) == 4, f"Expected 4 tiles, got {len(tiles)}"

    # Step 2: Segment each tile
    seg_tiles = []
    for tile_path in tiles:
        seg_name = tile_path.replace("tile", "seg")
        seg_tiles.append(seg_name)
        result = subprocess.run(
            [sys.executable, os.path.join(BIN_DIR, "color_segment.py"),
             "--input", tile_path,
             "--output", seg_name],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"Segment failed for {tile_path}: {result.stderr}"

    # Step 3: Merge
    merged_path = str(work_dir / "merged.png")
    cmd = [sys.executable, os.path.join(BIN_DIR, "image_merge.py")]
    for s in seg_tiles:
        cmd.extend(["--input", s])
    cmd.extend([
        "--output", merged_path,
        "--tile-size", str(tile_size),
        "--original-size", str(original_size),
    ])
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Merge failed: {result.stderr}"
    assert os.path.exists(merged_path)

    # Verify merged image dimensions
    merged = cv2.imread(merged_path)
    assert merged.shape[:2] == (original_size, original_size), \
        f"Expected {original_size}x{original_size}, got {merged.shape[:2]}"

    # Verify that segmentation actually changed the image (not pass-through)
    original = cv2.imread(img_path)
    assert not np.array_equal(original, merged), \
        "Merged image should differ from the original (segmentation colors applied)"
