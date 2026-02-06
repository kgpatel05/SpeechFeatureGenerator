"""Lexical semantics features (log frequency, semantic density) from word tier + CSV table."""

import os

import hdf5storage
import numpy as np
import pandas as pd

from speechfeaturegenerator.utils.waveform import prepare_waveform
from speechfeaturegenerator.utils.io import save_discrete_feature, write_summary
from speechfeaturegenerator.utils.textgrid_reader import load_word_labels_from_textgrid


def load_lexical_table(csv_path):
    """
    Load CSV with columns Word, Log_Freq_HAL, Semantic_Neighborhood_Density.
    Returns dict mapping word (lowercase) -> {Log_Freq_HAL, Semantic_Neighborhood_Density}.
    """
    df = pd.read_csv(csv_path)
    required = ["Word", "Log_Freq_HAL", "Semantic_Neighborhood_Density"]
    if not all(c in df.columns for c in required):
        raise ValueError(f"CSV must have columns: {required}")
    for c in ["Log_Freq_HAL", "Semantic_Neighborhood_Density"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    df["Word"] = df["Word"].astype(str).str.lower()
    # Drop duplicates, keeping first occurrence
    df = df.drop_duplicates(subset=["Word"], keep="first")
    return df.set_index("Word")[["Log_Freq_HAL", "Semantic_Neighborhood_Density"]].to_dict("index")


def lexical(
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
    lexical_table_path=None,
    variant="onehot_duration",
    feature_type="semantic_density",
    **kwargs,
):
    """
    Extract lexical semantics (log freq, semantic density) from word tier + CSV.
    Requires lexical_table_path (CSV with Word, Log_Freq_HAL, Semantic_Neighborhood_Density).
    
    Args:
        variant: "onehot_duration" (value over word duration) or "onehot_onset" (value at onset only).
        feature_type: "semantic_density" or "log_freq" (which column to use from CSV).
    """
    if lexical_table_path is None:
        raise ValueError("lexical_table_path (CSV path) is required for lexical features")
    lexical_dict = load_lexical_table(lexical_table_path)
    default = {"Log_Freq_HAL": 0.0, "Semantic_Neighborhood_Density": 0.0}

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
            generate_lexical_features(
                output_root=output_root,
                wav_path=wav_path,
                word_labels=word_labels,
                word_onsets=word_onsets,
                word_offsets=word_offsets,
                lexical_dict=lexical_dict,
                default=default,
                n_t=None,
                out_sr=out_sr,
                time_window=list(time_window),
                meta_only=meta_only,
                variant=variant,
                feature_type=feature_type,
            )


def generate_lexical_features(
    output_root,
    wav_path,
    word_labels,
    word_onsets,
    word_offsets,
    lexical_dict,
    default,
    n_t,
    out_sr=100,
    time_window=(-1, 1),
    meta_only=False,
    variant="onehot_duration",
    feature_type="semantic_density",
):
    """Generate time-aligned lexical feature for one file."""
    feature = "lexicalkp"  # Matching feature_extraction feature name
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

    # Note: Do NOT add time_window offset here because we use t_new (which starts at time_window[0])
    # directly with searchsorted. The offset is already in t_new's coordinate system.
    word_onsets = np.array(word_onsets, dtype=float)
    word_offsets = np.array(word_offsets, dtype=float)
    word_labels = np.array([str(x).strip() for x in word_labels])

    features_ts = np.zeros_like(t_new, dtype=float)
    semantic_features = []  # For discrete feature saving - matching feature_extraction
    for word, onset, offset in zip(word_labels, word_onsets, word_offsets):
        row = lexical_dict.get(word.lower(), default)
        if feature_type == "semantic_density":
            val = float(row.get("Semantic_Neighborhood_Density", 0.0))
        else:
            val = float(row.get("Log_Freq_HAL", 0.0))
        
        if variant == "onehot_onset":
            # Place value at onset only
            onset_idx = np.searchsorted(t_new, onset, side="left")
            if onset_idx < len(features_ts):
                features_ts[onset_idx] = val
        else:  # onehot_duration
            # Spread value over duration
            start_idx = np.searchsorted(t_new, onset, side="left")
            end_idx = np.searchsorted(t_new, offset, side="right")
            if start_idx < end_idx:
                features_ts[start_idx:end_idx] = val
        semantic_features.append(val)

    if not meta_only:
        semantic_features_arr = np.array(semantic_features, dtype=float)
        # Save discrete features with semantic_features array - matching feature_extraction
        save_discrete_feature(
            word_labels, word_onsets, word_offsets, discrete_dir, wav_name_no_ext, semantic_features_arr
        )
        out_path = os.path.join(
            feature_variant_out_dir, f"{wav_name_no_ext}.mat"
        )
        # Use format 7.3 matching feature_extraction
        hdf5storage.savemat(
            out_path,
            {"features": features_ts.reshape(-1, 1), "t": t_new},
            format="7.3",
            store_python_metadata=True,
            matlab_compatible=True,
        )
    else:
        save_discrete_feature(
            word_labels, word_onsets, word_offsets, discrete_dir, wav_name_no_ext
        )
    write_summary(
        feature_variant_out_dir,
        time_window=f"{-time_window[0]} second before to {time_window[1]} second after",
        dimensions="[time, 1]",
        sampling_rate=out_sr,
        extra=f"Lexical feature: {feature_type}, variant: {variant}. From CSV table.",
    )
