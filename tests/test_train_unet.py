"""Tests for bin/train_unet.py."""

import glob
import json
import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("tensorflow", reason="TensorFlow required for train_unet tests")

BIN_DIR = os.path.join(os.path.dirname(__file__), "..", "bin")
TRAIN_SCRIPT = os.path.join(BIN_DIR, "train_unet.py")
PREPROCESS_SCRIPT = os.path.join(BIN_DIR, "preprocess_data.py")


@pytest.fixture
def preprocessed_data(synthetic_training_data, tmp_path):
    """Run preprocess_data to create .npy files for training tests."""
    img_dir, mask_dir = synthetic_training_data
    x_train = str(tmp_path / "X_train.npy")
    x_test = str(tmp_path / "X_test.npy")
    y_train = str(tmp_path / "y_train_cat.npy")
    y_test = str(tmp_path / "y_test_cat.npy")

    cmd = [sys.executable, PREPROCESS_SCRIPT]
    for f in sorted(glob.glob(os.path.join(img_dir, "*.png"))):
        cmd.extend(["--image", f])
    for f in sorted(glob.glob(os.path.join(mask_dir, "*.png"))):
        cmd.extend(["--mask", f])
    cmd.extend([
        "--x-train", x_train, "--x-test", x_test,
        "--y-train", y_train, "--y-test", y_test,
        "--n-classes", "3", "--random-state", "42",
    ])
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    return x_train, x_test, y_train, y_test


def test_train_single_gpu(preprocessed_data, tmp_path):
    """Single-GPU training for 1 epoch should produce model and history."""
    x_train, _, y_train, _ = preprocessed_data
    model_out = str(tmp_path / "model.hdf5")
    history_out = str(tmp_path / "history.json")

    # Copy model.py to working directory (train_unet imports from cwd)
    import shutil
    shutil.copy(os.path.join(BIN_DIR, "model.py"), str(tmp_path / "model.py"))

    result = subprocess.run(
        [sys.executable, TRAIN_SCRIPT,
         "--train-data", x_train,
         "--train-labels", y_train,
         "--output-model", model_out,
         "--output-history", history_out,
         "--epochs", "1",
         "--batch-size", "2",
         "--n-classes", "3",
         "--mode", "single-gpu"],
        capture_output=True, text=True,
        cwd=str(tmp_path),
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert os.path.exists(model_out), "model.hdf5 not created"
    assert os.path.exists(history_out), "training_history.json not created"


def test_train_history_format(preprocessed_data, tmp_path):
    """Training history JSON should contain expected keys."""
    x_train, _, y_train, _ = preprocessed_data
    model_out = str(tmp_path / "model.hdf5")
    history_out = str(tmp_path / "history.json")

    import shutil
    shutil.copy(os.path.join(BIN_DIR, "model.py"), str(tmp_path / "model.py"))

    subprocess.run(
        [sys.executable, TRAIN_SCRIPT,
         "--train-data", x_train, "--train-labels", y_train,
         "--output-model", model_out, "--output-history", history_out,
         "--epochs", "1", "--batch-size", "2", "--n-classes", "3",
         "--mode", "single-gpu"],
        capture_output=True, text=True, check=True,
        cwd=str(tmp_path),
    )

    with open(history_out) as f:
        history = json.load(f)

    assert "loss" in history, "History should contain 'loss'"
    assert "accuracy" in history, "History should contain 'accuracy'"
    assert "training_time_seconds" in history
    assert isinstance(history["loss"], list)
    assert len(history["loss"]) == 1  # 1 epoch
