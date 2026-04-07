#!/usr/bin/env python3

"""Evaluate a trained U-Net model on the test set.

Loads the trained model and test data, computes loss, accuracy,
F1, precision, and recall, and writes results to JSON.
"""

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import tensorflow as tf
from keras import backend as K

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def main():
    parser = argparse.ArgumentParser(description="Evaluate U-Net model")
    parser.add_argument("--model", required=True, help="Trained model.hdf5")
    parser.add_argument("--test-data", required=True, help="X_test.npy")
    parser.add_argument("--test-labels", required=True, help="y_test_cat.npy")
    parser.add_argument("--output", required=True, help="Output evaluation_results.json")
    args = parser.parse_args()

    logger.info(f"Model: {args.model}")
    logger.info(f"Test data: {args.test_data}")

    X_test = np.load(args.test_data)
    y_test_cat = np.load(args.test_labels)
    logger.info(f"Test data shape: {X_test.shape}, Labels: {y_test_cat.shape}")

    # Load model with custom metrics
    custom_objects = {
        "recall_m": recall_m,
        "precision_m": precision_m,
        "f1_m": f1_m,
    }
    model = tf.keras.models.load_model(args.model, custom_objects=custom_objects)

    begin = time.time()
    test_loss, test_acc, f1_score, precision, recall = model.evaluate(
        X_test, y_test_cat, verbose=1
    )
    end = time.time()

    results = {
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "f1_score": float(f1_score),
        "precision": float(precision),
        "recall": float(recall),
        "evaluation_time_seconds": end - begin,
    }

    logger.info(f"Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
    logger.info(f"F1: {f1_score:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved: {args.output}")


if __name__ == "__main__":
    main()
