#!/usr/bin/env python3

"""Merge segmented tiles back into a full-size image.

Takes explicit tile file paths (passed as repeated --input arguments)
and reassembles them into the original image dimensions based on the
row/col indices encoded in filenames.
"""

import argparse
import logging
import os
import re
import sys

import numpy as np
from PIL import Image

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Merge segmented tiles into full image")
    parser.add_argument("--input", action="append", required=True,
                        help="Input tile PNG (can be specified multiple times)")
    parser.add_argument("--output", required=True, help="Output merged image PNG")
    parser.add_argument("--tile-size", type=int, default=250,
                        help="Tile dimension in pixels (default: 250)")
    parser.add_argument("--original-size", type=int, default=2000,
                        help="Original image dimension (default: 2000)")
    args = parser.parse_args()

    logger.info(f"Merging {len(args.input)} tiles into {args.output}")
    logger.info(f"Tile size: {args.tile_size}, Original size: {args.original_size}")

    # Parse row/col from filenames and sort
    tiles = {}
    pattern = re.compile(r"_(\d{4})_(\d{4})\.png$")
    for tile_path in args.input:
        match = pattern.search(os.path.basename(tile_path))
        if match:
            row, col = int(match.group(1)), int(match.group(2))
            tiles[(row, col)] = tile_path
        else:
            logger.warning(f"Could not parse row/col from: {tile_path}")

    if not tiles:
        logger.error("No valid tiles found")
        sys.exit(1)

    # Assemble full image
    full_img = np.zeros((args.original_size, args.original_size, 3), dtype=np.uint8)
    for (row, col), path in sorted(tiles.items()):
        tile = np.array(Image.open(path))
        th, tw = tile.shape[:2]
        full_img[row:row + th, col:col + tw, :] = tile

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    Image.fromarray(full_img).save(args.output)
    logger.info(f"Merged image saved: {args.output}")


if __name__ == "__main__":
    main()
