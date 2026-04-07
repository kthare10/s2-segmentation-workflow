#!/usr/bin/env python3

"""HSV color segmentation on a single image tile.

Classifies pixels into three classes using HSV thresholds:
  - Ice (bright): colored red
  - Thin ice (mid-range): colored blue
  - Water (dark): colored green
"""

import argparse
import logging
import os
import sys

import cv2

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def color_segmentation(img):
    """Apply HSV threshold-based segmentation to an RGB image."""
    lower_ice = (0, 0, 205)
    upper_ice = (185, 255, 255)

    lower_tice = (0, 0, 31)
    upper_tice = (185, 255, 204)

    lower_water = (0, 0, 0)
    upper_water = (185, 255, 30)

    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    mask_ice = cv2.inRange(hsv_img, lower_ice, upper_ice)
    mask_tice = cv2.inRange(hsv_img, lower_tice, upper_tice)
    mask_water = cv2.inRange(hsv_img, lower_water, upper_water)

    seg_img = img.copy()
    seg_img[mask_ice == 255] = [255, 0, 0]
    seg_img[mask_tice == 255] = [0, 0, 255]
    seg_img[mask_water == 255] = [0, 255, 0]

    return seg_img


def main():
    parser = argparse.ArgumentParser(description="Color segment a single tile")
    parser.add_argument("--input", required=True, help="Input tile PNG")
    parser.add_argument("--output", required=True, help="Output segmented tile PNG")
    args = parser.parse_args()

    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {args.output}")

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        logger.error(f"Failed to read image: {args.input}")
        sys.exit(1)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    seg = color_segmentation(img_rgb)
    seg_bgr = cv2.cvtColor(seg, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output, seg_bgr)

    logger.info(f"Segmentation complete: {args.output}")


if __name__ == "__main__":
    main()
