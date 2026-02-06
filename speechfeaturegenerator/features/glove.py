"""GloVe word embedding features from word tier + GloVe text file."""

import os

import hdf5storage
import numpy as np

from speechfeaturegenerator.utils.waveform import prepare_waveform
from speechfeaturegenerator.utils.io import save_discrete_feature, write_summary
from speechfeaturegenerator.utils.textgrid_reader import load_word_labels_from_textgrid
from speechfeaturegenerator.utils.embeddings import load_glove_embeddings, get_word_embedding


def glove(
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
    glove_file_path=None,
    variant="onehot_duration",
    **kwargs,
):
    """
    Extract GloVe word embedding features from word tier.
    Requires glove_file_path (GloVe .txt: word + space-separated vector per line).
    
    Args:
        variant: "onehot_duration" (embedding over word duration) or "onehot_onset" (embedding at onset only).
    """
    if glove_file_path is None:
        raise ValueError(
            "glove_file_path (GloVe .txt path) is required for glove features"
        )
    glove_model = load_glove_embeddings(glove_path=glove_file_path)

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
            generate_glove_features(
                output_root=output_root,
                wav_path=wav_path,
                word_labels=word_labels,
                word_onsets=word_onsets,
                word_offsets=word_offsets,
                glove_model=glove_model,
                n_t=None,
                out_sr=out_sr,
                time_window=list(time_window),
                meta_only=meta_only,
                variant=variant,
            )


def generate_glove_features(
    output_root,
    wav_path,
    word_labels,
    word_onsets,
    word_offsets,
    glove_model,
    n_t,
    out_sr=100,
    time_window=(-1, 1),
    meta_only=False,
    variant="onehot_duration",
):
    """Generate time-aligned GloVe embedding features for one file."""
    feature = "glovekp"  # Matching feature_extraction feature name
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

    sample = next(iter(glove_model.values()))
    dim = sample.shape[0]
    default_vec = np.zeros(dim, dtype=np.float32)
    features_ts = np.zeros((len(t_new), dim), dtype=np.float32)

    glove_discrete_features = []
    for i, (onset, offset) in enumerate(zip(word_onsets, word_offsets)):
        emb = get_word_embedding(
            word_labels[i], glove_model, lowercase=True, default=default_vec
        )
        if variant == "onehot_onset":
            # Place embedding at onset only
            onset_idx = np.searchsorted(t_new, onset, side="left")
            if onset_idx < len(features_ts):
                features_ts[onset_idx, :] = emb
        else:  # onehot_duration
            # Spread embedding over duration
            start_idx = np.searchsorted(t_new, onset, side="left")
            end_idx = np.searchsorted(t_new, offset, side="right")
            features_ts[start_idx:end_idx, :] = emb
        glove_discrete_features.append(emb)

    # Save discrete features with glove_discrete_features array - matching feature_extraction
    glove_discrete_features = np.array(glove_discrete_features)
    save_discrete_feature(
        word_labels, word_onsets, word_offsets, discrete_dir, wav_name_no_ext, glove_discrete_features
    )
    out_path = os.path.join(
        feature_variant_out_dir, f"{wav_name_no_ext}.mat"
    )
    hdf5storage.savemat(out_path, {"features": features_ts, "t": t_new})
    write_summary(
        feature_variant_out_dir,
        time_window=f"{-time_window[0]} second before to {time_window[1]} second after",
        dimensions=f"[time, {dim}]",
        sampling_rate=out_sr,
        extra="GloVe word embeddings per time point.",
    )
