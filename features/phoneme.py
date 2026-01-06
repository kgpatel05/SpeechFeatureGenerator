"""Phoneme feature extraction from audio files."""

import os
import re

import hdf5storage
import numpy as np

from utils.waveform import prepare_waveform
from utils.features import generate_onehot_features
from utils.io import save_discrete_feature, write_summary
from utils.phoneme_reader import (
    load_phoneme_labels_from_csv,
    load_phoneme_label_set,
)


def phoneme(
    device,
    output_root,
    stim_names,
    wav_dir,
    csv_path,
    out_sr=100,
    pc=None,
    time_window=[-1, 1],
    pca_weights_from=None,
    compute_original=True,
    meta_only=False,
    **kwargs,
):
    """Extract phoneme features from audio files using CSV labels."""
    variant = kwargs.get("variant", "onehot_duration")
    phoneme_labels_csv = kwargs.get("phoneme_labels_csv", None)

    if compute_original:
        for stim_name in stim_names:
            wav_path = os.path.join(wav_dir, f"{stim_name}.wav")
            phoneme_labels, phoneme_onsets, phoneme_offsets = (
                load_phoneme_labels_from_csv(csv_path, stim_name)
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
                phoneme_labels_csv=phoneme_labels_csv,
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
    phoneme_labels_csv=None,
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

    if phoneme_labels_csv is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        phoneme_labels_csv = os.path.join(
            os.path.dirname(script_dir), "data", "phoneme_labels_standard.csv"
        )

    all_phoneme_labels = load_phoneme_label_set(phoneme_labels_csv)

    phoneme_labels, phoneme_onsets, phoneme_offsets = remove_phoneme(
        phoneme_labels, phoneme_onsets, phoneme_offsets, "spn"
    )

    phoneme_labels = np.array([label.upper() for label in phoneme_labels])
    phoneme_labels = [re.sub(r"\d+", "", phoneme) for phoneme in phoneme_labels]
    phoneme_labels = [str(phoneme).strip() for phoneme in phoneme_labels]
    
    phoneme_onsets = phoneme_onsets.reshape(-1)
    phoneme_offsets = phoneme_offsets.reshape(-1)
    phoneme_labels = np.array(phoneme_labels, dtype="<U3")

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
