from scipy.stats import norm
import tensorflow as tf
import numpy as np
import random as python_random
seed = 42
np.random.seed(seed)
python_random.seed(seed)
tf.random.set_seed(seed)

@tf.function
def compute_dynamic_thresholds(means, logvars, z_score):
    stds = tf.sqrt(tf.exp(logvars))  # Recover standard deviations from log-variances
    thresholds = tf.reduce_sum(stds * z_score, axis=1)  # Compute thresholds
    return thresholds


def classify_with_dynamic_thresholds(features, means, logvars, preds, z_score):
    # Classify samples with dynamic thresholds for outlier detection.
    thresholds = compute_dynamic_thresholds(means, logvars, z_score=z_score)  # Class-level thresholds
    stds = tf.sqrt(tf.exp(logvars))  # Add epsilon to avoid division by zero
    # Compute Mahalanobis-like distance
    distances = tf.reduce_sum(((features - tf.gather(means, preds)) / tf.gather(stds, preds)) ** 2, axis=1)
    # Identify unknown samples
    unknown_mask = distances > tf.gather(thresholds, preds)
    classifications = tf.where(unknown_mask, tf.constant(means.shape[0], dtype=tf.int32), preds)  # Mark unknowns
    return classifications