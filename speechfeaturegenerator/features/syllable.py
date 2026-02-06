"""Syllable feature extraction (vowel_position, syllable_type) from phoneme TextGrids."""

import os
import re

import hdf5storage
import numpy as np

from speechfeaturegenerator.utils.waveform import prepare_waveform
from speechfeaturegenerator.utils.io import save_discrete_feature, write_summary
from speechfeaturegenerator.utils.textgrid_reader import (
    load_phoneme_labels_from_textgrid,
)


def is_vowel(phoneme: str) -> bool:
    """Return True if this phoneme label is one of the standard ARPABET vowels."""
    # Common ARPABET vowel labels (without stress markers) - matching feature_extraction
    vowels = {
        'AA', 'AE', 'AH', 'AO', 'AW', 'AY',
        'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW'
    }
    return phoneme.upper() in vowels


def classify_syllable(
    phoneme_labels: list, syllable_start: int, syllable_end: int
) -> int:
    """
    Classify syllable type from phoneme slice [syllable_start..syllable_end].
    Returns 0=Closed, 1=Open, 2=Vowel-r, 3=Vowel Team, 4=Consonant-le, 5=Other.
    """
    # Convert the slice into a plain Python list of strings - matching feature_extraction
    seq = list(phoneme_labels[syllable_start : syllable_end + 1])
    # Find the nucleus: the first vowel in this slice - matching feature_extraction
    nucleus_idx = None
    for idx, ph in enumerate(seq):
        if is_vowel(ph):
            nucleus_idx = idx
            break
    if nucleus_idx is None:
        # If no vowel at all, treat as "Other."
        return 5

    nucleus = seq[nucleus_idx].upper()
    coda = seq[nucleus_idx + 1 :]  # phonemes after the nucleus, as a list

    # (A) Vowel-r: if nucleus is "ER"
    if nucleus == 'ER':
        return 2

    # (B) Vowel Team: ARPABET diphthongs
    if nucleus in {'AW', 'AY', 'OY', 'OW'}:
        return 3

    # (C) Consonant-le (C-le): if the final two phonemes look like [consonant, 'AH']
    if len(seq) >= 2:
        last = seq[-1].upper()
        second_last = seq[-2].upper()
        if last == 'AH' and second_last.endswith('L'):
            return 4

    # (D) Closed vs Open:
    #    • Closed: coda is nonempty and its first item is a consonant
    #    • Open: coda is empty or begins with a vowel
    if len(coda) > 0:
        if not is_vowel(coda[0]):
            return 0  # Closed
        else:
            return 1  # Open
    else:
        return 1  # Open (no coda at all)

    # (E) Anything else → "Other"
    return 5


def syllable(
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
    variants=None,
    window_size=3,
    **kwargs,
):
    """
    Extract syllable features (vowel_position, syllable_type) from TextGrid phoneme tier.
    variants: list of "vowel_position", "syllable_type" (default: both).
    """
    if variants is None:
        variants = ["vowel_position", "syllable_type"]
    if compute_original:
        for stim_name in stim_names:
            wav_path = os.path.join(wav_dir, f"{stim_name}.wav")
            if textgrid_path:
                phoneme_labels, phoneme_onsets, phoneme_offsets = (
                    load_phoneme_labels_from_textgrid(textgrid_path)
                )
            elif textgrid_dir:
                tg_path = os.path.join(textgrid_dir, f"{stim_name}.TextGrid")
                phoneme_labels, phoneme_onsets, phoneme_offsets = (
                    load_phoneme_labels_from_textgrid(tg_path)
                )
            else:
                raise ValueError(
                    "Must provide either textgrid_path or textgrid_dir"
                )
            generate_syllable_features(
                output_root=output_root,
                wav_path=wav_path,
                phoneme_labels=phoneme_labels,
                phoneme_onsets=phoneme_onsets,
                phoneme_offsets=phoneme_offsets,
                n_t=None,
                out_sr=out_sr,
                time_window=list(time_window),
                meta_only=meta_only,
                variants=variants,
                window_size=window_size,
            )


def generate_syllable_features(
    output_root,
    wav_path,
    phoneme_labels,
    phoneme_onsets,
    phoneme_offsets,
    n_t,
    out_sr=100,
    time_window=(-1, 1),
    meta_only=False,
    variants=None,
    window_size=3,
):
    """Generate vowel_position and/or syllable_type features for one file."""
    if variants is None:
        variants = ["vowel_position", "syllable_type"]
    feature = "syllable"
    all_variants = list(variants) + ["discrete"]
    (
        wav_name_no_ext,
        _,
        _,
        t_num_new,
        t_new,
        feature_variant_out_dirs,
    ) = prepare_waveform(
        out_sr, wav_path, output_root, n_t, time_window, feature, all_variants
    )
    discrete_dir = feature_variant_out_dirs[-1]
    save_discrete_feature(
        phoneme_labels,
        phoneme_onsets,
        phoneme_offsets,
        discrete_dir,
        wav_name_no_ext,
    )
    # Remove "SP" (silent pause) phonemes before processing
    phoneme_labels = np.array(phoneme_labels)
    phoneme_onsets = np.array(phoneme_onsets, dtype=float)
    phoneme_offsets = np.array(phoneme_offsets, dtype=float)
    
    # Filter out SP phonemes
    sp_mask = np.array([str(label).upper().strip() != "SP" for label in phoneme_labels])
    phoneme_labels = phoneme_labels[sp_mask]
    phoneme_onsets = phoneme_onsets[sp_mask]
    phoneme_offsets = phoneme_offsets[sp_mask]
    
    # Note: Do NOT add time_window offset here because we compare directly to t_new
    # (which starts at time_window[0]). The offset is already in t_new's coordinate system.
    phoneme_labels = list(phoneme_labels)

    vowel_indices = [i for i, ph in enumerate(phoneme_labels) if is_vowel(ph)]
    if not vowel_indices:
        vowel_indices = [0]
    syllable_bounds = []
    for idx in range(len(vowel_indices)):
        start = vowel_indices[idx - 1] + 1 if idx > 0 else 0
        end = (
            vowel_indices[idx + 1] - 1
            if idx < len(vowel_indices) - 1
            else len(phoneme_labels) - 1
        )
        syllable_bounds.append((start, end))
    syllable_types = [
        classify_syllable(phoneme_labels, start, end)
        for (start, end) in syllable_bounds
    ]
    phoneme_to_syllable = np.zeros(len(phoneme_labels), dtype=int)
    for syll_idx, (start, end) in enumerate(syllable_bounds):
        phoneme_to_syllable[start : end + 1] = syll_idx

    for variant, feature_variant_out_dir in zip(
        variants, feature_variant_out_dirs[:-1]
    ):
        if variant == "vowel_position":
            features = np.zeros((t_num_new, window_size), dtype=np.float32)
            for t_idx in range(t_num_new):
                current_time = t_new[t_idx]
                active = [
                    (i, phoneme_labels[i])
                    for i, (on, off) in enumerate(
                        zip(phoneme_onsets, phoneme_offsets)
                    )
                    if on <= current_time <= off
                ]
                if active:
                    current_idx = active[0][0]
                    half = window_size // 2
                    window_phonemes = []
                    for i_idx in range(
                        current_idx - half, current_idx + half + 1
                    ):
                        if 0 <= i_idx < len(phoneme_labels):
                            window_phonemes.append(phoneme_labels[i_idx])
                        else:
                            window_phonemes.append("PAD")
                    features[t_idx, :] = [
                        1.0 if is_vowel(p) else 0.0 for p in window_phonemes
                    ]
            desc = (
                f"Sliding window vowel-position (window_size={window_size}), "
                f"SR={out_sr} Hz"
            )
        elif variant == "syllable_type":
            n_types = 6
            features = np.zeros((t_num_new, n_types), dtype=np.float32)
            for t_idx in range(t_num_new):
                current_time = t_new[t_idx]
                active = [
                    (i,)
                    for i, (on, off) in enumerate(
                        zip(phoneme_onsets, phoneme_offsets)
                    )
                    if on <= current_time <= off
                ]
                if active:
                    syl_idx = phoneme_to_syllable[active[0][0]]
                    type_idx = syllable_types[syl_idx]
                    features[t_idx, type_idx] = 1.0
                else:
                    features[t_idx, 5] = 1.0
            desc = (
                "Syllable type one-hot (Closed, Open, Vowel-r, V-Team, C-le, "
                f"Other), SR={out_sr} Hz"
            )
        else:
            raise ValueError(f"Unknown variant: {variant}")

        out_mat_path = os.path.join(
            feature_variant_out_dir, f"{wav_name_no_ext}.mat"
        )
        hdf5storage.savemat(out_mat_path, {"features": features, "t": t_new})
        write_summary(
            feature_variant_out_dir,
            time_window=f"{-time_window[0]} second before to {time_window[1]} second after",
            dimensions=f"[time, {features.shape[1]}]",
            sampling_rate=out_sr,
            extra=desc,
        )
