"""Input/output utilities for saving features and metadata."""

import os
import hdf5storage
import numpy as np


def save_discrete_feature(labels, onsets, offsets, output_dir, wav_name_no_ext, extra_data=None):
    """
    Saves discrete feature data (labels, onsets, offsets) to a .mat file.

    Args:
        labels (np.ndarray): Array of labels.
        onsets (np.ndarray): Array of onset times.
        offsets (np.ndarray): Array of offset times.
        output_dir (str): Directory to save the output file.
        wav_name_no_ext (str): Base name for the output file (without extension).
        extra_data (np.ndarray, optional): Additional data array to save (e.g., word frequencies, embeddings).

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    discrete_data = np.column_stack([labels, onsets, offsets])
    out_path = os.path.join(output_dir, f"{wav_name_no_ext}.mat")
    if extra_data is not None:
        hdf5storage.savemat(out_path, {"discrete": discrete_data, "extra": extra_data})
    else:
        hdf5storage.savemat(out_path, {"discrete": discrete_data})


def write_summary(output_dir, time_window, dimensions, sampling_rate, extra=None):
    """
    Writes a summary text file with feature extraction metadata.

    Args:
        output_dir (str): Directory to save the summary file.
        time_window (str): Description of the time window used.
        dimensions (str): Description of feature dimensions.
        sampling_rate (int): Sampling rate of the features.
        extra (str, optional): Additional information to include. Defaults to None.

    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(output_dir, "summary.txt")

    with open(summary_path, "w") as f:
        f.write(f"Time Window: {time_window}\n")
        f.write(f"Dimensions: {dimensions}\n")
        f.write(f"Sampling Rate: {sampling_rate} Hz\n")
        if extra:
            f.write(f"Additional Info: {extra}\n")

