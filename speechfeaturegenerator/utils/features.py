"""Feature generation utilities for one-hot encoding."""

import numpy as np


def generate_onehot_features(
    labels, onsets, offsets, n_t, all_labels, mode="duration", sr=100
):
    """
    Generates one-hot encoded features from labels with onset and offset times.

    Args:
        labels (np.ndarray): Array of labels (e.g., phoneme labels).
        onsets (np.ndarray): Array of onset times in seconds.
        offsets (np.ndarray): Array of offset times in seconds.
        n_t (int): Number of time points in the output feature matrix.
        all_labels (list): List of all possible labels (defines feature dimensions).
        mode (str, optional): Feature mode. One of "onset", "offset", or "duration". Defaults to "duration".
        sr (int, optional): Sample rate for time discretization. Defaults to 100.

    Returns:
        np.ndarray: One-hot encoded feature matrix of shape (n_t, n_labels).
    """
    n_labels = len(all_labels)
    features = np.zeros((n_t, n_labels), dtype=np.int8)

    label_to_index = {label: idx for idx, label in enumerate(all_labels)}

    dt = 1.0 / sr
    time_points = np.arange(n_t) * dt

    for label, onset, offset in zip(labels, onsets, offsets):
        if label not in label_to_index:
            continue

        label_idx = label_to_index[label]

        if mode == "onset":
            onset_idx = np.argmin(np.abs(time_points - onset))
            if onset_idx < n_t:
                features[onset_idx, label_idx] = 1
        elif mode == "offset":
            offset_idx = np.argmin(np.abs(time_points - offset))
            if offset_idx < n_t:
                features[offset_idx, label_idx] = 1
        else:  # mode == "duration"
            onset_idx = np.argmin(np.abs(time_points - onset))
            offset_idx = np.argmin(np.abs(time_points - offset))
            if onset_idx < n_t:
                start_idx = max(0, onset_idx)
                end_idx = min(n_t, offset_idx + 1)
                features[start_idx:end_idx, label_idx] = 1

    return features

