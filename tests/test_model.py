"""Tests for bin/model.py — U-Net model definition."""

import os
import sys

import numpy as np
import pytest

tf = pytest.importorskip("tensorflow", reason="TensorFlow required for model tests")

# Add bin/ to path so we can import model directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bin"))
from model import multi_unet_model


def test_model_default_input_shape():
    """Default model should accept (256, 256, 1) inputs."""
    model = multi_unet_model()
    assert model.input_shape == (None, 256, 256, 1)


def test_model_default_output_shape():
    """Default model should output (256, 256, 3) — 3-class softmax."""
    model = multi_unet_model()
    assert model.output_shape == (None, 256, 256, 3)


def test_model_custom_classes():
    """Model with n_classes=5 should output 5-channel softmax."""
    model = multi_unet_model(n_classes=5)
    assert model.output_shape == (None, 256, 256, 5)


def test_model_custom_dimensions():
    """Model should accept custom input dimensions."""
    model = multi_unet_model(IMG_HEIGHT=128, IMG_WIDTH=128, IMG_CHANNELS=3)
    assert model.input_shape == (None, 128, 128, 3)
    assert model.output_shape == (None, 128, 128, 3)


def test_model_forward_pass():
    """A forward pass with random data should produce valid probability maps."""
    model = multi_unet_model(n_classes=3, IMG_HEIGHT=64, IMG_WIDTH=64, IMG_CHANNELS=1)
    model.compile(optimizer="adam", loss="categorical_crossentropy")

    dummy = np.random.rand(2, 64, 64, 1).astype(np.float32)
    output = model.predict(dummy, verbose=0)

    assert output.shape == (2, 64, 64, 3)
    # Softmax outputs should sum to ~1 along the class axis
    sums = output.sum(axis=-1)
    np.testing.assert_allclose(sums, 1.0, atol=1e-5)


def test_model_is_uncompiled():
    """multi_unet_model should return an uncompiled model."""
    model = multi_unet_model()
    assert model.optimizer is None
