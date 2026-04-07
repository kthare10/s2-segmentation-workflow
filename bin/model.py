#!/usr/bin/env python3

"""Shared U-Net model definition for Sentinel-2 sea ice segmentation.

6-level U-Net (16→32→64→128→256→512 filters) with Conv2DTranspose
upsampling, dropout (0.1–0.3), and softmax output for 3-class
semantic segmentation (ice, thin-ice, water).

This module is used by both train_unet.py and evaluate_model.py.
It is registered in the Pegasus Replica Catalog and staged into
each job's working directory.
"""

from keras.layers import (
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    MaxPooling2D,
    concatenate,
)
from keras.models import Model


def multi_unet_model(n_classes=3, IMG_HEIGHT=256, IMG_WIDTH=256, IMG_CHANNELS=1):
    """Build a 6-level U-Net for semantic segmentation.

    Args:
        n_classes: Number of output classes.
        IMG_HEIGHT: Input image height.
        IMG_WIDTH: Input image width.
        IMG_CHANNELS: Number of input channels.

    Returns:
        Uncompiled Keras Model.
    """
    inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
    s = inputs

    # Contraction path
    c1 = Conv2D(16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p4)
    c5 = Dropout(0.2)(c5)
    c5 = Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c5)
    p5 = MaxPooling2D(pool_size=(2, 2))(c5)

    c6 = Conv2D(512, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(p5)
    c6 = Dropout(0.3)(c6)
    c6 = Conv2D(512, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c6)

    # Expansive path
    u7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding="same")(c6)
    u7 = concatenate([u7, c5])
    c7 = Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(256, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c7)

    u8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding="same")(c7)
    u8 = concatenate([u8, c4])
    c8 = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u8)
    c8 = Dropout(0.2)(c8)
    c8 = Conv2D(128, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c8)

    u9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(c8)
    u9 = concatenate([u9, c3])
    c9 = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u9)
    c9 = Dropout(0.2)(c9)
    c9 = Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c9)

    u10 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding="same")(c9)
    u10 = concatenate([u10, c2])
    c10 = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u10)
    c10 = Dropout(0.1)(c10)
    c10 = Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c10)

    u11 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding="same")(c10)
    u11 = concatenate([u11, c1], axis=3)
    c11 = Conv2D(16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(u11)
    c11 = Dropout(0.1)(c11)
    c11 = Conv2D(16, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same")(c11)

    outputs = Conv2D(n_classes, (1, 1), activation="softmax")(c11)

    model = Model(inputs=[inputs], outputs=[outputs])
    return model
