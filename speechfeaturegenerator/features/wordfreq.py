"""Word frequency features (SUBTLWF, Lg10WF, SUBTLCD, Lg10CD) from word tier + table."""

import os

import hdf5storage
import numpy as np
import pandas as pd

from speechfeaturegenerator.utils.waveform import prepare_waveform
from speechfeaturegenerator.utils.io import save_discrete_feature, write_summary
from speechfeaturegenerator.utils.textgrid_reader import load_word_labels_from_textgrid


def load_word_freq_table(path):
    """
    Load Excel or CSV with Word, SUBTLWF, Lg10WF, SUBTLCD, Lg10CD.
    Returns dict of dicts: SUBTLWF, Lg10WF, SUBTLCD, Lg10CD each word -> value.
    """
    if path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path)
    df["Word"] = df["Word"].astype(str).str.lower()
    return {
        "SUBTLWF": dict(zip(df["Word"], df["SUBTLWF"])),
        "Lg10WF": dict(zip(df["Word"], df["Lg10WF"])),
        "SUBTLCD": dict(zip(df["Word"], df["SUBTLCD"])),
        "Lg10CD": dict(zip(df["Word"], df["Lg10CD"])),
    }


def wordfreq(
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
    variant="onehot_duration",
    fill_zeros=False,
    **kwargs,
):
    """
    Extract word frequency features from word tier + table.
    Requires word_freq_table_path (Excel/CSV with Word, SUBTLWF, Lg10WF, SUBTLCD, Lg10CD).
    
    Args:
        variant: "onehot_duration" (values over word duration) or "onehot_onset" (values at onset only).
        fill_zeros: If True, use 0.0 for missing words; if False, use NaN.
    """
    if word_freq_table_path is None:
        raise ValueError(
            "word_freq_table_path (Excel/CSV) is required for wordfreq features"
        )
    word_freq_dicts = load_word_freq_table(word_freq_table_path)
    fill = 0.0 if fill_zeros else np.nan

    if compute_original:
        for stim_name in stim_names:
            wav_path = os.path.join(wav_dir, f"{stim_name}.wav")
            if textgrid_path:
                word_labels, word_onsets, word_offsets = (
                    load_word_labels_from_textgrid(textgrid_path)
                )
            elif textgrid_dir:
                tg_path = os.path.join(textgrid_dir, f"{stim_name}.TextGrid")
                word_labels, word_onsets, word_offsets = (
                    load_word_labels_from_textgrid(tg_path)
                )
            else:
                raise ValueError(
                    "Must provide either textgrid_path or textgrid_dir"
                )
            generate_wordfreq_features(
                output_root=output_root,
                wav_path=wav_path,
                word_labels=word_labels,
                word_onsets=word_onsets,
                word_offsets=word_offsets,
                word_freq_dicts=word_freq_dicts,
                fill_missing=fill,
                n_t=None,
                out_sr=out_sr,
                time_window=list(time_window),
                meta_only=meta_only,
                variant=variant,
            )


def generate_wordfreq_features(
    output_root,
    wav_path,
    word_labels,
    word_onsets,
    word_offsets,
    word_freq_dicts,
    fill_missing,
    n_t,
    out_sr=100,
    time_window=(-1, 1),
    meta_only=False,
    variant="onehot_duration",
):
    """Generate time-aligned word frequency features for one file."""
    feature = "wordfreq"
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

    word_onsets = np.array(word_onsets, dtype=float)
    word_offsets = np.array(word_offsets, dtype=float)
    word_labels = list(word_labels)

    v1 = [word_freq_dicts["SUBTLWF"].get(w, fill_missing) for w in word_labels]
    v2 = [word_freq_dicts["Lg10WF"].get(w, fill_missing) for w in word_labels]
    v3 = [word_freq_dicts["SUBTLCD"].get(w, fill_missing) for w in word_labels]
    v4 = [word_freq_dicts["Lg10CD"].get(w, fill_missing) for w in word_labels]
    word_frequencies = np.column_stack([v1, v2, v3, v4]).astype(float)
    n_cols = word_frequencies.shape[1]

    features_ts = np.zeros((len(t_new), n_cols), dtype=float)
    for i, (onset, offset) in enumerate(zip(word_onsets, word_offsets)):
        if variant == "onehot_onset":
            # Place values at onset only
            onset_idx = np.searchsorted(t_new, onset, side="left")
            if onset_idx < len(features_ts):
                features_ts[onset_idx, :] = word_frequencies[i, :]
        else:  # onehot_duration
            # Spread values over duration
            start_idx = np.searchsorted(t_new, onset, side="left")
            end_idx = np.searchsorted(t_new, offset, side="right")
            features_ts[start_idx:end_idx, :] = word_frequencies[i, :]

    # Save discrete features with word frequencies array - matching feature_extraction
    # Note: word_onsets/offsets are saved without adjustment, matching feature_extraction
    save_discrete_feature(
        np.array(word_labels),
        word_onsets,
        word_offsets,
        discrete_dir,
        wav_name_no_ext,
        word_frequencies,
    )
    out_path = os.path.join(
        feature_variant_out_dir, f"{wav_name_no_ext}.mat"
    )
    hdf5storage.savemat(out_path, {"features": features_ts, "t": t_new})
    missing = "0.0" if fill_missing == 0.0 else "NaN"
    write_summary(
        feature_variant_out_dir,
        time_window=f"{-time_window[0]}s to {time_window[1]}s",
        dimensions=f"[time, {n_cols}]",
        sampling_rate=out_sr,
        extra=f"Columns: SUBTLWF, Lg10WF, SUBTLCD, Lg10CD. Missing={missing}. Variant: {variant}.",
    )
