"""Word one-hot feature extraction from audio files using TextGrid labels."""

import os

import hdf5storage
import numpy as np

from speechfeaturegenerator.utils.waveform import prepare_waveform
from speechfeaturegenerator.utils.features import generate_onehot_features
from speechfeaturegenerator.utils.io import save_discrete_feature, write_summary
from speechfeaturegenerator.utils.textgrid_reader import load_word_labels_from_textgrid


def word(
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
    variant="onehot_duration",
    **kwargs,
):
    """
    Extract word one-hot features from audio files using TextGrid labels.

    Args:
        device: Device (unused; kept for API compatibility).
        output_root: Root directory for output files.
        stim_names: List of stimulus names (no extension).
        wav_dir: Directory containing .wav files.
        textgrid_path: Single TextGrid path (optional).
        textgrid_dir: Directory of per-stimulus TextGrids (optional).
        out_sr: Output sampling rate in Hz.
        time_window: [start_sec, end_sec] relative to audio start.
        variant: "onehot_duration" or "onehot_onset".
    """
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
            generate_word_features(
                output_root=output_root,
                wav_path=wav_path,
                word_labels=word_labels,
                word_onsets=word_onsets,
                word_offsets=word_offsets,
                n_t=None,
                out_sr=out_sr,
                time_window=list(time_window),
                meta_only=meta_only,
                variant=variant,
            )


def generate_word_features(
    output_root,
    wav_path,
    word_labels,
    word_onsets,
    word_offsets,
    n_t,
    out_sr=100,
    time_window=(-1, 1),
    meta_only=False,
    variant="onehot_duration",
):
    """Generate word one-hot features from word labels and timing."""
    feature = "word"
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

    word_onsets = np.array(word_onsets, dtype=float)
    word_offsets = np.array(word_offsets, dtype=float)
    word_labels = np.array([str(x).strip() for x in word_labels])
    non_empty = np.array([len(x) > 0 for x in word_labels], dtype=bool)
    if not np.any(non_empty):
        raise ValueError(f"No non-empty word labels in TextGrid for {wav_path}")
    word_labels = word_labels[non_empty]
    word_onsets = word_onsets[non_empty]
    word_offsets = word_offsets[non_empty]

    # Save discrete features (before adding time offset)
    save_discrete_feature(
        word_labels,
        word_onsets,
        word_offsets,
        discrete_feature_variant_out_dir,
        wav_name_no_ext,
    )

    # Add time window offset for one-hot generation
    word_onsets = word_onsets + abs(time_window[0])
    word_offsets = word_offsets + abs(time_window[0])
    all_word_labels = sorted(set(word_labels))
    n_words = len(all_word_labels)

    # Determine mode from variant
    if variant == "onehot_onset":
        mode = "onset"
    else:  # onehot_duration
        mode = "duration"

    features = generate_onehot_features(
        word_labels,
        word_onsets,
        word_offsets,
        t_num_new,
        all_word_labels,
        mode=mode,
        sr=out_sr,
    )

    out_mat_path = os.path.join(feature_variant_out_dir, f"{wav_name_no_ext}.mat")
    hdf5storage.savemat(out_mat_path, {"features": features, "t": t_new})
    write_summary(
        feature_variant_out_dir,
        time_window=f"{-time_window[0]} second before to {time_window[1]} second after",
        dimensions=f"[time, {n_words}]",
        sampling_rate=out_sr,
        extra=f"Each column corresponds to one of: {all_word_labels}",
    )
