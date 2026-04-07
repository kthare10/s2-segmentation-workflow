#!/usr/bin/env python3

"""Split a Sentinel-2 image into tiles for parallel processing.

Takes a single large PNG image and splits it into a grid of smaller
tile PNGs. Each tile is named {basename}_{row}_{col}.png with
zero-padded row/col indices.
"""

import argparse
import logging
import os
import sys

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Split image into tiles")
    parser.add_argument("--input", required=True, help="Input image PNG")
    parser.add_argument("--output-prefix", required=True,
                        help="Output tile prefix (e.g. 's2_vis_00')")
    parser.add_argument("--tile-size", type=int, default=250,
                        help="Tile dimension in pixels (default: 250)")
    parser.add_argument("--grayscale", action="store_true",
                        help="Read image as grayscale (single channel)")
    parser.add_argument("--pad", action="store_true",
                        help="Zero-pad edge tiles to full tile size "
                             "(use when image dims are not divisible by tile size)")
    args = parser.parse_args()

    logger.info(f"Input: {args.input}")
    logger.info(f"Output prefix: {args.output_prefix}")
    logger.info(f"Tile size: {args.tile_size}")

    read_flag = cv2.IMREAD_GRAYSCALE if args.grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(args.input, read_flag)
    if img is None:
        logger.error(f"Failed to read image: {args.input}")
        sys.exit(1)

    h, w = img.shape[:2]
    logger.info(f"Image size: {w}x{h} (grayscale={args.grayscale})")

    tile_count = 0
    ts = args.tile_size
    for r in range(0, h, ts):
        for c in range(0, w, ts):
            if args.grayscale:
                tile = img[r:r + ts, c:c + ts]
            else:
                tile = img[r:r + ts, c:c + ts, :]

            # Pad undersized edge tiles to full tile_size with zeros
            if args.pad and (tile.shape[0] < ts or tile.shape[1] < ts):
                if args.grayscale:
                    padded = np.zeros((ts, ts), dtype=tile.dtype)
                else:
                    padded = np.zeros((ts, ts, tile.shape[2]), dtype=tile.dtype)
                padded[:tile.shape[0], :tile.shape[1]] = tile
                tile = padded

            filename = f"{args.output_prefix}_{str(r).zfill(4)}_{str(c).zfill(4)}.png"
            cv2.imwrite(filename, tile)
            tile_count += 1

    logger.info(f"Created {tile_count} tiles")


if __name__ == "__main__":
    main()
