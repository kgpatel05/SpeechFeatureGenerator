"""Phoneme feature extraction from audio files."""

import os
import re

import hdf5storage
import numpy as np

from speechfeaturegenerator.utils.waveform import prepare_waveform
from speechfeaturegenerator.utils.features import generate_onehot_features
from speechfeaturegenerator.utils.io import save_discrete_feature, write_summary
from speechfeaturegenerator.utils.textgrid_reader import (
    load_phoneme_labels_from_textgrid,
)


def phoneme(
    device,
    output_root,
    stim_names,
    wav_dir,
    textgrid_path=None,
    textgrid_dir=None,
    out_sr=100,
    pc=None,
    time_window=[-1, 1],
    pca_weights_from=None,
    compute_original=True,
    meta_only=False,
    **kwargs,
):
    """
    Extract phoneme features from audio files using TextGrid labels.
    
    Args:
        textgrid_path: Path to a single TextGrid file (optional)
        textgrid_dir: Directory containing TextGrid files (optional, will look for {stim_name}.TextGrid)
    """
    variant = kwargs.get("variant", "onehot_duration")

    if compute_original:
        for stim_name in stim_names:
            wav_path = os.path.join(wav_dir, f"{stim_name}.wav")
            
            # Load from TextGrid
            if textgrid_path:
                # Single TextGrid file for all stimuli
                phoneme_labels, phoneme_onsets, phoneme_offsets = (
                    load_phoneme_labels_from_textgrid(textgrid_path)
                )
            elif textgrid_dir:
                # TextGrid file per stimulus
                tg_path = os.path.join(textgrid_dir, f"{stim_name}.TextGrid")
                phoneme_labels, phoneme_onsets, phoneme_offsets = (
                    load_phoneme_labels_from_textgrid(tg_path)
                )
            else:
                raise ValueError(
                    "Must provide either textgrid_path or textgrid_dir"
                )

            generate_phoneme_features(
                output_root,
                wav_path,
                phoneme_labels=phoneme_labels,
                phoneme_onsets=phoneme_onsets,
                phoneme_offsets=phoneme_offsets,
                n_t=None,
                out_sr=out_sr,
                time_window=time_window,
                meta_only=meta_only,
                variant=variant,
            )


def generate_phoneme_features(
    output_root,
    wav_path,
    phoneme_labels,
    phoneme_onsets,
    phoneme_offsets,
    n_t,
    out_sr=100,
    time_window=[-1, 1],
    meta_only=False,
    variant="onehot_duration",
):
    """Generate phoneme one-hot features from labels and timing."""
    feature = "phoneme"
    variants = [variant, "discrete"]
    (
        wav_name_no_ext,
        waveform,
        sample_rate,
        t_num_new,
        t_new,
        feature_variant_out_dirs,
    ) = prepare_waveform(
        out_sr, wav_path, output_root, n_t, time_window, feature, variants
    )
    feature_variant_out_dir = feature_variant_out_dirs[0]
    discrete_feature_variant_out_dir = feature_variant_out_dirs[1]
    
    phoneme_onsets = np.array(phoneme_onsets, dtype=float)
    phoneme_offsets = np.array(phoneme_offsets, dtype=float)
    
    save_discrete_feature(
        phoneme_labels,
        phoneme_onsets,
        phoneme_offsets,
        discrete_feature_variant_out_dir,
        wav_name_no_ext,
    )

    phoneme_onsets = phoneme_onsets + abs(time_window[0])
    phoneme_offsets = phoneme_offsets + abs(time_window[0])

    # Remove "spn" phonemes first
    phoneme_labels, phoneme_onsets, phoneme_offsets = remove_phoneme(
        phoneme_labels, phoneme_onsets, phoneme_offsets, "spn"
    )

    # Clean and normalize phoneme labels
    phoneme_labels = np.array([label.upper() for label in phoneme_labels])
    phoneme_labels = [re.sub(r"\d+", "", phoneme) for phoneme in phoneme_labels]
    phoneme_labels = [str(phoneme).strip() for phoneme in phoneme_labels]
    
    phoneme_onsets = phoneme_onsets.reshape(-1)
    phoneme_offsets = phoneme_offsets.reshape(-1)
    phoneme_labels = np.array(phoneme_labels, dtype="<U3")

    # Extract unique phoneme labels from the data
    all_phoneme_labels = sorted(set(phoneme_labels))

    if variant == "onehot_onset":
        mode = "onset"
    elif variant == "onehot_offset":
        mode = "offset"
    else:
        mode = "duration"

    phoneme_features = generate_onehot_features(
        phoneme_labels,
        phoneme_onsets,
        phoneme_offsets,
        t_num_new,
        all_phoneme_labels,
        mode=mode,
        sr=out_sr,
    )

    out_mat_path = os.path.join(feature_variant_out_dir, f"{wav_name_no_ext}.mat")
    hdf5storage.savemat(out_mat_path, {"features": phoneme_features, "t": t_new})

    write_summary(
        feature_variant_out_dir,
        time_window=f"{-time_window[0]} second before to {time_window[1]} second after",
        dimensions="[time, phoneme]",
        sampling_rate=out_sr,
        extra=f"Each column correspond to one of them: {all_phoneme_labels}",
    )


def remove_phoneme(phoneme_labels, phoneme_onsets, phoneme_offsets, phoneme_remove_label):
    """Remove specified phoneme label and corresponding timing."""
    phoneme_indexes = np.array(
        [i for i, label in enumerate(phoneme_labels) if label != phoneme_remove_label]
    )
    return (
        phoneme_labels[phoneme_indexes],
        phoneme_onsets[phoneme_indexes],
        phoneme_offsets[phoneme_indexes],
    )
