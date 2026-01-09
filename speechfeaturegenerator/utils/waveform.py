"""Waveform processing utilities for feature extraction."""

import os
import librosa
import numpy as np


def prepare_waveform(
    out_sr, wav_path, output_root, n_t, time_window, feature, variants
):
    """
    Prepares waveform for feature extraction by loading, resampling, and creating time vectors.

    Args:
        out_sr (int): Output sample rate for resampling.
        wav_path (str): Path to the input audio file.
        output_root (str): Root directory for output files.
        n_t (int, optional): Number of time points. If None, calculated from audio duration.
        time_window (list): Time window [start, end] in seconds relative to audio start.
        feature (str): Feature name (e.g., "phoneme").
        variants (list): List of variant names for output directories.

    Returns:
        tuple: Contains:
            - wav_name_no_ext (str): Audio filename without extension
            - waveform (np.ndarray): Loaded waveform
            - sample_rate (int): Original sample rate
            - t_num_new (int): Number of time points in output
            - t_new (np.ndarray): Time vector for output
            - feature_variant_out_dirs (list): List of output directories for each variant
    """
    wav_name_no_ext = os.path.splitext(os.path.basename(wav_path))[0]
    waveform, sample_rate = librosa.load(wav_path, sr=None)

    duration = len(waveform) / sample_rate
    total_duration = duration + abs(time_window[0]) + time_window[1]

    if n_t is None:
        t_num_new = int(total_duration * out_sr)
    else:
        t_num_new = n_t

    t_new = np.linspace(time_window[0], duration + time_window[1], t_num_new)

    feature_variant_out_dirs = []
    for variant in variants:
        variant_dir = os.path.join(output_root, feature, variant)
        os.makedirs(variant_dir, exist_ok=True)
        feature_variant_out_dirs.append(variant_dir)

    return (
        wav_name_no_ext,
        waveform,
        sample_rate,
        t_num_new,
        t_new,
        feature_variant_out_dirs,
    )

