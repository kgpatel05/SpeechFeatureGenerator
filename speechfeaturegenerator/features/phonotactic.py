"""Phonotactic (diphone frequency) feature extraction from audio files.

Computes log10 diphone frequency based on word frequency table and ARPA dictionary.
"""

import os
import pickle
import re

import hdf5storage
import numpy as np
import pandas as pd

from speechfeaturegenerator.utils.waveform import prepare_waveform
from speechfeaturegenerator.utils.io import save_discrete_feature, write_summary
from speechfeaturegenerator.utils.textgrid_reader import (
    load_phoneme_labels_from_textgrid,
    load_word_labels_from_textgrid,
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


def phonotactic(
    device,
    output_root,
    stim_names,
    wav_dir,
    textgrid_path=None,
    textgrid_dir=None,
    out_sr=100,
    pc=None,
    time_window=(-1, 1),
    pca_weights_from=None,
    compute_original=True,
    meta_only=False,
    word_freq_table_path=None,
    arpa_dict_path=None,
    variant="onehot_duration",
    **kwargs,
):
    """
    Extract phonotactic (diphone frequency) features from audio files.

    Computes log10 diphone frequency based on word frequency table and ARPA dictionary.

    Args:
        device: Device (unused; kept for API compatibility).
        output_root: Root directory for output files.
        stim_names: List of stimulus names (no extension).
        wav_dir: Directory containing .wav files.
        textgrid_path: Single TextGrid path (optional).
        textgrid_dir: Directory of per-stimulus TextGrids (optional).
        out_sr: Output sampling rate in Hz.
        time_window: [start_sec, end_sec] relative to audio start.
        word_freq_table_path: Path to word frequency Excel file (.xls).
        arpa_dict_path: Path to ARPA dictionary file (.dict).
        variant: "onehot_duration" or "onehot_onset".
    """
    if word_freq_table_path is None:
        raise ValueError("word_freq_table_path is required for phonotactic features")
    if arpa_dict_path is None:
        raise ValueError("arpa_dict_path is required for phonotactic features")

    # Load word frequency table
    word_freq_df = pd.read_excel(word_freq_table_path)
    word_freq_dict = dict(zip(word_freq_df["Word"], word_freq_df["FREQcount"]))

    # Build diphone frequency dictionary
    diphone_freq_dict = extract_diphone_frequencies(
        output_root, arpa_dict_path, word_freq_dict
    )

    if compute_original:
        for stim_name in stim_names:
            wav_path = os.path.join(wav_dir, f"{stim_name}.wav")

            # Load from TextGrid
            if textgrid_path:
                phoneme_labels, phoneme_onsets, phoneme_offsets = (
                    load_phoneme_labels_from_textgrid(textgrid_path)
                )
                word_labels, word_onsets, word_offsets = (
                    load_word_labels_from_textgrid(textgrid_path)
                )
            elif textgrid_dir:
                tg_path = os.path.join(textgrid_dir, f"{stim_name}.TextGrid")
                phoneme_labels, phoneme_onsets, phoneme_offsets = (
                    load_phoneme_labels_from_textgrid(tg_path)
                )
                word_labels, word_onsets, word_offsets = (
                    load_word_labels_from_textgrid(tg_path)
                )
            else:
                raise ValueError("Must provide either textgrid_path or textgrid_dir")

            generate_phonotactic_features(
                output_root=output_root,
                wav_path=wav_path,
                word_labels=word_labels,
                word_onsets=word_onsets,
                word_offsets=word_offsets,
                phoneme_labels=phoneme_labels,
                phoneme_onsets=phoneme_onsets,
                phoneme_offsets=phoneme_offsets,
                diphone_freq_dict=diphone_freq_dict,
                n_t=None,
                out_sr=out_sr,
                time_window=list(time_window),
                meta_only=meta_only,
                variant=variant,
            )


def extract_diphone_frequencies(output_root, arpa_dict_path, word_freq_dict):
    """
    Build diphone frequency dictionary from ARPA dict and word frequency table.

    Returns dict mapping diphone label (e.g., "AA.B") to frequency count.
    """
    # Check for cached version
    cache_path = os.path.join(output_root, "diphone_freq_dict.pkl")
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    # Load ARPA dictionary
    arpa_dict = {}
    with open(arpa_dict_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            word = parts[0].lower()
            # Skip float numbers, get only phonemes
            phonemes = " ".join([p for p in parts if ("." not in p) and p != word])
            if word not in arpa_dict:
                arpa_dict[word] = phonemes

    diphone_freq_dict = {}

    # Words that can have 't added (contractions)
    can_add_contraction = [
        "can", "ain", "aren", "couldn", "didn", "doesn", "don", "hadn",
        "hasn", "haven", "isn", "mustn", "shouldn", "wasn", "weren", "wouldn", "won",
    ]

    for word, freq in word_freq_dict.items():
        word = str(word).lower()

        if word not in arpa_dict:
            # Try adding 't for contractions
            if word.endswith("n") and word in can_add_contraction:
                word = word + "'t"
            if word not in arpa_dict:
                continue

        phonemes = arpa_dict[word].split(" ")
        # Remove stress markers (digits)
        phonemes = [re.sub(r"\d+", "", p) for p in phonemes]

        # Build diphones
        diphones = []
        if len(phonemes) == 1:
            diphones.append(phonemes[0] + ". ")
        else:
            for i in range(len(phonemes) - 1):
                diphones.append(phonemes[i] + "." + phonemes[i + 1])

        for diphone in diphones:
            diphone_freq_dict[diphone] = diphone_freq_dict.get(diphone, 0) + freq

    # Cache the dictionary
    os.makedirs(output_root, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(diphone_freq_dict, f)

    return diphone_freq_dict


def generate_phonotactic_features(
    output_root,
    wav_path,
    word_labels,
    word_onsets,
    word_offsets,
    phoneme_labels,
    phoneme_onsets,
    phoneme_offsets,
    diphone_freq_dict,
    n_t,
    out_sr=100,
    time_window=(-1, 1),
    meta_only=False,
    variant="onehot_duration",
):
    """Generate phonotactic (diphone frequency) features for one file."""
    feature = "phonotactickp"
    variants = [variant, "discrete"]
    (
        wav_name_no_ext,
        _,
        _,
        _,
        t_new,
        feature_variant_out_dirs,
    ) = prepare_waveform(
        out_sr, wav_path, output_root, n_t, time_window, feature, variants
    )
    feature_variant_out_dir = feature_variant_out_dirs[0]
    discrete_dir = feature_variant_out_dirs[1]

    # Convert to arrays
    phoneme_onsets = np.array(phoneme_onsets, dtype=float)
    phoneme_offsets = np.array(phoneme_offsets, dtype=float)
    word_onsets = np.array(word_onsets, dtype=float)
    word_offsets = np.array(word_offsets, dtype=float)

    # Remove "spn" phonemes
    phoneme_labels, phoneme_onsets, phoneme_offsets = remove_phoneme(
        phoneme_labels, phoneme_onsets, phoneme_offsets, "spn"
    )

    # Normalize phoneme labels
    phoneme_labels = np.array([label.upper() for label in phoneme_labels])
    phoneme_labels = [re.sub(r"\d+", "", p) for p in phoneme_labels]
    phoneme_labels = [str(p).strip().rstrip(".") for p in phoneme_labels]

    phoneme_onsets = phoneme_onsets.reshape(-1)
    phoneme_offsets = phoneme_offsets.reshape(-1)
    phoneme_durations = phoneme_offsets - phoneme_onsets
    tolerance = min(np.min(phoneme_durations) / 2, 1e-2) if len(phoneme_durations) > 0 else 1e-2
    phoneme_labels = np.array(phoneme_labels, dtype="<U3")

    # Remove "SP" phonemes
    sp_mask = phoneme_labels != "SP"
    phoneme_labels = phoneme_labels[sp_mask]
    phoneme_onsets = phoneme_onsets[sp_mask]
    phoneme_offsets = phoneme_offsets[sp_mask]

    # Build diphone labels, onsets, offsets
    diphone_labels = []
    diphone_onsets = []
    diphone_offsets = []

    for i in range(len(word_labels)):
        word_start = word_onsets[i]
        word_end = word_offsets[i]

        # Find phonemes within this word
        phoneme_indices = np.where(
            (phoneme_onsets >= word_start - tolerance)
            & (phoneme_offsets <= word_end + tolerance)
        )[0]

        if len(phoneme_indices) == 1:
            phoneme = phoneme_labels[phoneme_indices[0]]
            diphone = f"{phoneme}. "
            diphone_labels.append(diphone)
            diphone_onsets.append(phoneme_onsets[phoneme_indices[0]])
            diphone_offsets.append(phoneme_offsets[phoneme_indices[0]])
        elif len(phoneme_indices) == 0:
            continue
        else:
            for j in range(len(phoneme_indices) - 1):
                p1 = phoneme_labels[phoneme_indices[j]]
                p2 = phoneme_labels[phoneme_indices[j + 1]]
                diphone = f"{p1}.{p2}"
                diphone_labels.append(diphone)
                diphone_onsets.append(phoneme_onsets[phoneme_indices[j]])
                diphone_offsets.append(phoneme_offsets[phoneme_indices[j + 1]])

    diphone_labels = np.array(diphone_labels)
    diphone_onsets = np.array(diphone_onsets)
    diphone_offsets = np.array(diphone_offsets)

    # Get frequencies for each diphone
    diphone_frequencies = np.array(
        [diphone_freq_dict.get(d, 1) for d in diphone_labels],  # default 1 to avoid log10(0)
        dtype=float,
    )

    # Create time-aligned feature (log10 frequency)
    features_ts = np.zeros(len(t_new), dtype=float)
    for i, (onset, offset) in enumerate(zip(diphone_onsets, diphone_offsets)):
        log_freq = np.log10(diphone_frequencies[i]) if diphone_frequencies[i] > 0 else 0.0
        if variant == "onehot_onset":
            # Place value at onset only
            onset_idx = np.searchsorted(t_new, onset, side="left")
            if onset_idx < len(features_ts):
                features_ts[onset_idx] = log_freq
        else:  # onehot_duration
            # Spread value over duration
            start_idx = np.searchsorted(t_new, onset, side="left")
            end_idx = np.searchsorted(t_new, offset, side="right")
            features_ts[start_idx:end_idx] = log_freq

    features_ts = features_ts.reshape(-1, 1)

    if not meta_only:
        out_path = os.path.join(feature_variant_out_dir, f"{wav_name_no_ext}.mat")
        hdf5storage.savemat(out_path, {"features": features_ts, "t": t_new})

        save_discrete_feature(
            diphone_labels,
            diphone_onsets,
            diphone_offsets,
            discrete_dir,
            wav_name_no_ext,
            np.log10(np.maximum(diphone_frequencies, 1)),
        )

    write_summary(
        feature_variant_out_dir,
        time_window=f"{-time_window[0]} second before to {time_window[1]} second after",
        dimensions="[time, 1]",
        sampling_rate=out_sr,
        extra="Log10(diphone frequency) from ARPA dict + word frequency table.",
    )
