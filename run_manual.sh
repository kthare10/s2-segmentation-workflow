#!/bin/bash
# run_manual.sh — Test each pipeline step locally before Pegasus submission.
#
# Prerequisites: pip install tensorflow opencv-python-headless scikit-learn Pillow
#
# This script creates synthetic test data and runs each wrapper script
# to validate argument parsing, I/O, and tool compatibility.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEST_DIR="${SCRIPT_DIR}/test_run"

echo "=== Setting up test directory ==="
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR/train_images" "$TEST_DIR/train_masks"
cd "$TEST_DIR"

# --- Generate synthetic test data ---
echo "=== Generating synthetic test image (250x250, will test with tile_size=125) ==="
python3 -c "
import numpy as np, cv2
# Create a 250x250 test image with colored regions
img = np.zeros((250, 250, 3), dtype=np.uint8)
img[:125, :125] = [255, 255, 255]   # white (ice)
img[:125, 125:] = [100, 100, 100]   # gray (thin ice)
img[125:, :] = [10, 10, 30]         # dark (water)
cv2.imwrite('test_image.png', img)

# Create 4 small 256x256 training images and masks
for i in range(4):
    train = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
    cv2.imwrite(f'train_images/img_{i:02d}.png', train)
    mask = np.random.choice([0, 128, 255], size=(256, 256)).astype(np.uint8)
    cv2.imwrite(f'train_masks/mask_{i:02d}.png', mask)

print('Test data created.')
"

# --- Stage 1: Color Segmentation ---
echo ""
echo "=== Step 1: image_split ==="
python3 "$SCRIPT_DIR/bin/image_split.py" \
    --input test_image.png \
    --output-prefix test_tile \
    --tile-size 125

echo ""
echo "=== Step 2: color_segment (per tile) ==="
for tile in test_tile_*.png; do
    seg_name="${tile/test_tile/test_seg}"
    python3 "$SCRIPT_DIR/bin/color_segment.py" \
        --input "$tile" \
        --output "$seg_name"
done

echo ""
echo "=== Step 3: image_merge ==="
SEG_ARGS=""
for seg in test_seg_*.png; do
    SEG_ARGS="$SEG_ARGS --input $seg"
done
python3 "$SCRIPT_DIR/bin/image_merge.py" \
    $SEG_ARGS \
    --output test_merged.png \
    --tile-size 125 \
    --original-size 250

echo ""
echo "=== Step 4: preprocess_data ==="
IMG_ARGS=""
for f in train_images/*.png; do
    IMG_ARGS="$IMG_ARGS --image $f"
done
MASK_ARGS=""
for f in train_masks/*.png; do
    MASK_ARGS="$MASK_ARGS --mask $f"
done
python3 "$SCRIPT_DIR/bin/preprocess_data.py" \
    $IMG_ARGS \
    $MASK_ARGS \
    --x-train X_train.npy \
    --x-test X_test.npy \
    --y-train y_train_cat.npy \
    --y-test y_test_cat.npy \
    --n-classes 3

echo ""
echo "=== Step 5: train_unet (2 epochs for testing) ==="
cp "$SCRIPT_DIR/bin/model.py" .
python3 "$SCRIPT_DIR/bin/train_unet.py" \
    --train-data X_train.npy \
    --train-labels y_train_cat.npy \
    --output-model model.hdf5 \
    --output-history training_history.json \
    --epochs 2 \
    --batch-size 2 \
    --n-classes 3 \
    --mode single-gpu

echo ""
echo "=== Step 6: evaluate_model ==="
python3 "$SCRIPT_DIR/bin/evaluate_model.py" \
    --model model.hdf5 \
    --test-data X_test.npy \
    --test-labels y_test_cat.npy \
    --output evaluation_results.json

echo ""
echo "=== Step 7: generate_plots ==="
mkdir -p plots
python3 "$SCRIPT_DIR/bin/generate_plots.py" \
    --training-history training_history.json \
    --evaluation-results evaluation_results.json \
    --model model.hdf5 \
    --test-data X_test.npy \
    --test-labels y_test_cat.npy \
    --metadata preprocess_metadata.json \
    --output-dir plots \
    --num-samples 3

echo ""
echo "=== All steps completed successfully! ==="
echo "Output files:"
ls -la test_merged.png model.hdf5 training_history.json evaluation_results.json
echo "Plot files:"
ls -la plots/
