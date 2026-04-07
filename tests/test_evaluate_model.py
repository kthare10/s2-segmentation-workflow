"""Tests for bin/evaluate_model.py."""

import glob
import json
import os
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("tensorflow", reason="TensorFlow required for evaluate_model tests")

BIN_DIR = os.path.join(os.path.dirname(__file__), "..", "bin")
TRAIN_SCRIPT = os.path.join(BIN_DIR, "train_unet.py")
EVAL_SCRIPT = os.path.join(BIN_DIR, "evaluate_model.py")
PREPROCESS_SCRIPT = os.path.join(BIN_DIR, "preprocess_data.py")


@pytest.fixture
def trained_model(synthetic_training_data, tmp_path):
    """Preprocess data, train for 1 epoch, return paths to model and test data."""
    img_dir, mask_dir = synthetic_training_data
    x_train = str(tmp_path / "X_train.npy")
    x_test = str(tmp_path / "X_test.npy")
    y_train = str(tmp_path / "y_train_cat.npy")
    y_test = str(tmp_path / "y_test_cat.npy")

    # Preprocess
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

    # Train
    import shutil
    shutil.copy(os.path.join(BIN_DIR, "model.py"), str(tmp_path / "model.py"))
    model_out = str(tmp_path / "model.hdf5")
    subprocess.run(
        [sys.executable, TRAIN_SCRIPT,
         "--train-data", x_train, "--train-labels", y_train,
         "--output-model", model_out, "--output-history", str(tmp_path / "hist.json"),
         "--epochs", "1", "--batch-size", "2", "--n-classes", "3",
         "--mode", "single-gpu"],
        capture_output=True, text=True, check=True,
        cwd=str(tmp_path),
    )
    return model_out, x_test, y_test


def test_evaluate_produces_results(trained_model, tmp_path):
    """evaluate_model should create evaluation_results.json."""
    model_path, x_test, y_test = trained_model
    eval_out = str(tmp_path / "eval_results.json")

    result = subprocess.run(
        [sys.executable, EVAL_SCRIPT,
         "--model", model_path,
         "--test-data", x_test,
         "--test-labels", y_test,
         "--output", eval_out],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert os.path.exists(eval_out)


def test_evaluate_result_keys(trained_model, tmp_path):
    """Evaluation JSON should contain all expected metric keys."""
    model_path, x_test, y_test = trained_model
    eval_out = str(tmp_path / "eval_results.json")

    subprocess.run(
        [sys.executable, EVAL_SCRIPT,
         "--model", model_path, "--test-data", x_test,
         "--test-labels", y_test, "--output", eval_out],
        capture_output=True, text=True, check=True,
    )

    with open(eval_out) as f:
        results = json.load(f)

    expected_keys = {"test_loss", "test_accuracy", "f1_score", "precision",
                     "recall", "evaluation_time_seconds"}
    assert expected_keys.issubset(results.keys()), \
        f"Missing keys: {expected_keys - results.keys()}"


def test_evaluate_metrics_are_numeric(trained_model, tmp_path):
    """All metric values should be numeric (float)."""
    model_path, x_test, y_test = trained_model
    eval_out = str(tmp_path / "eval_results.json")

    subprocess.run(
        [sys.executable, EVAL_SCRIPT,
         "--model", model_path, "--test-data", x_test,
         "--test-labels", y_test, "--output", eval_out],
        capture_output=True, text=True, check=True,
    )

    with open(eval_out) as f:
        results = json.load(f)

    for key, value in results.items():
        assert isinstance(value, (int, float)), f"{key} should be numeric, got {type(value)}"
