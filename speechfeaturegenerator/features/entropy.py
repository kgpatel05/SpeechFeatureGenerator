"""GPT-2 word entropy feature extraction from audio files."""

import os

import hdf5storage
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from speechfeaturegenerator.utils.waveform import prepare_waveform
from speechfeaturegenerator.utils.io import save_discrete_feature, write_summary
from speechfeaturegenerator.utils.textgrid_reader import load_word_labels_from_textgrid


def entropy(
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
    model_name="gpt2",
    log_base="e",
    **kwargs,
):
    """
    Extract GPT-2 word entropy features from audio files using TextGrid labels.
    
    Args:
        device: Device to run model on (e.g., "cuda", "mps", "cpu")
        output_root: Root directory for output files
        stim_names: List of stimulus names (without extension)
        wav_dir: Directory containing .wav files
        textgrid_path: Path to a single TextGrid file (optional)
        textgrid_dir: Directory containing TextGrid files (optional, will look for {stim_name}.TextGrid)
        out_sr: Output sampling rate in Hz (default: 100)
        time_window: Time window [start, end] in seconds relative to audio start
        compute_original: Whether to compute original features
        meta_only: If True, only generate metadata
        model_name: GPT-2 model name (default: "gpt2")
        log_base: Base for logarithm ("e", "2", or "10", default: "e")
        **kwargs: Additional arguments
    """
    variant = kwargs.get("variant", "onehot_duration")

    if compute_original:
        for stim_name in stim_names:
            wav_path = os.path.join(wav_dir, f"{stim_name}.wav")
            
            # Load from TextGrid
            if textgrid_path:
                # Single TextGrid file for all stimuli
                word_labels, word_onsets, word_offsets = (
                    load_word_labels_from_textgrid(textgrid_path)
                )
            elif textgrid_dir:
                # TextGrid file per stimulus
                tg_path = os.path.join(textgrid_dir, f"{stim_name}.TextGrid")
                word_labels, word_onsets, word_offsets = (
                    load_word_labels_from_textgrid(tg_path)
                )
            else:
                raise ValueError(
                    "Must provide either textgrid_path or textgrid_dir"
                )

            generate_entropy_features(
                output_root,
                wav_path,
                word_labels=word_labels,
                word_onsets=word_onsets,
                word_offsets=word_offsets,
                n_t=None,
                out_sr=out_sr,
                time_window=time_window,
                meta_only=meta_only,
                variant=variant,
                device=device,
                model_name=model_name,
                log_base=log_base,
            )


def generate_entropy_features(
    output_root,
    wav_path,
    word_labels,
    word_onsets,
    word_offsets,
    n_t,
    out_sr=100,
    time_window=[-1, 1],
    meta_only=False,
    variant="onehot_duration",
    device="cpu",
    model_name="gpt2",
    log_base="e",
):
    """
    Generate GPT-2 word entropy features from word labels and timing.
    
    Args:
        output_root: Root directory for output files
        wav_path: Path to audio file
        word_labels: Array of word labels
        word_onsets: Array of word onset times
        word_offsets: Array of word offset times
        n_t: Number of time points (if None, calculated from audio)
        out_sr: Output sampling rate in Hz
        time_window: Time window [start, end] in seconds
        meta_only: If True, only generate metadata
        variant: "onehot_duration" (value over word duration) or "onehot_onset" (value at onset only)
        device: Device to run model on
        model_name: GPT-2 model name
        log_base: Base for logarithm ("e", "2", or "10")
    """
    feature = "entropy"
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
    word_labels = np.array(word_labels)
    
    # Filter out empty words
    non_empty_mask = np.array([len(str(label).strip()) > 0 for label in word_labels])
    spoken_words = word_labels[non_empty_mask]
    spoken_onsets = word_onsets[non_empty_mask]
    spoken_offsets = word_offsets[non_empty_mask]
    
    if len(spoken_words) == 0:
        raise ValueError(f"No spoken words found in TextGrid for {wav_path}")
    
    # Compute GPT-2 entropy
    word_entropy_values = compute_word_entropy(
        spoken_words.tolist(),
        device=device,
        model_name=model_name,
        log_base=log_base,
    )
    
    # Create full array with NaN for empty words
    word_entropy_full = np.full(len(word_labels), np.nan, dtype=float)
    word_entropy_full[non_empty_mask] = word_entropy_values
    
    # Save discrete feature (word labels with timing)
    save_discrete_feature(
        word_labels,
        word_onsets,
        word_offsets,
        discrete_feature_variant_out_dir,
        wav_name_no_ext,
    )
    
    # Create time-aligned feature
    # Note: Use t_new directly with searchsorted (no offset needed - t_new starts at time_window[0])
    entropy_ts = np.full(t_num_new, np.nan, dtype=float)
    
    for i in range(len(word_labels)):
        if np.isnan(word_entropy_full[i]):
            continue
        onset = word_onsets[i]
        offset = word_offsets[i]
        if variant == "onehot_onset":
            # Place value at onset only
            onset_idx = np.searchsorted(t_new, onset, side="left")
            if onset_idx < len(entropy_ts):
                entropy_ts[onset_idx] = word_entropy_full[i]
        else:  # onehot_duration
            # Spread value over duration
            start_idx = np.searchsorted(t_new, onset, side="left")
            end_idx = np.searchsorted(t_new, offset, side="right")
            if start_idx < end_idx:
                entropy_ts[start_idx:end_idx] = word_entropy_full[i]
    
    # Reshape for output (time x 1)
    entropy_features = entropy_ts.reshape(-1, 1)
    
    if not meta_only:
        out_mat_path = os.path.join(feature_variant_out_dir, f"{wav_name_no_ext}.mat")
        hdf5storage.savemat(
            out_mat_path,
            {"features": entropy_features, "t": t_new}
        )
    
    log_unit = "nats" if log_base == "e" else ("bits" if log_base == "2" else "dits")
    write_summary(
        feature_variant_out_dir,
        time_window=f"{-time_window[0]} second before to {time_window[1]} second after",
        dimensions="[time, 1]",
        sampling_rate=out_sr,
        extra=f"GPT-2 word entropy (first token) in {log_unit}. Model: {model_name}",
    )


def compute_word_entropy(words, device="cpu", model_name="gpt2", log_base="e"):
    """
    Compute GPT-2 entropy for a list of words.
    
    Args:
        words: List of word strings
        device: Device to run model on
        model_name: GPT-2 model name
        log_base: Base for logarithm ("e", "2", or "10")
    
    Returns:
        Array of word entropy values (entropy of first token of each word)
    """
    # Load model and tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
    model.eval()
    
    # Create sentence from words
    sentence = " ".join(words)
    
    # Tokenize
    enc = tokenizer(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=True,
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"][0]
    offsets = enc["offset_mapping"][0].tolist()
    
    # Compute token-level entropy
    token_entropy = compute_token_entropy(
        model, tokenizer, input_ids, device=device, log_base=log_base
    )
    
    # Align tokens to words
    word_char_spans = compute_word_char_spans(words)
    word_to_tokens = token_to_word_alignment(offsets, word_char_spans)
    
    # Aggregate to word level (use first token entropy)
    word_entropy = np.full(len(words), np.nan, dtype=float)
    for i, toks in enumerate(word_to_tokens):
        if toks:
            word_entropy[i] = token_entropy[toks[0]]
    
    return word_entropy


@torch.no_grad()
def compute_token_entropy(model, tokenizer, input_ids, device="cpu", log_base="e"):
    """
    Compute token-level entropy using GPT-2.
    
    Args:
        model: GPT-2 model
        tokenizer: GPT-2 tokenizer
        input_ids: Token IDs tensor
        device: Device to run on
        log_base: Base for logarithm ("e", "2", or "10")
    
    Returns:
        Array of token entropy values
    """
    # Add EOS as BOS-like prefix
    bos = torch.tensor([tokenizer.eos_token_id], dtype=torch.long)
    ids2 = torch.cat([bos, input_ids.cpu()]).unsqueeze(0).to(device)
    
    outputs = model(ids2)
    logits = outputs.logits[0]
    
    # We want distributions that predict ids2[1:] (the sentence tokens)
    pred_logits = logits[:-1]
    
    log_probs = torch.log_softmax(pred_logits, dim=-1)
    probs = torch.softmax(pred_logits, dim=-1)
    
    # Compute entropy: -sum(p * log(p))
    token_entropy = -(probs * log_probs).sum(dim=-1)
    
    # Convert log base if needed
    if log_base == "2":
        token_entropy = token_entropy / np.log(2)
    elif log_base == "10":
        token_entropy = token_entropy / np.log(10)
    # log_base == "e" uses natural log (no conversion needed)
    
    return token_entropy.cpu().numpy()


def compute_word_char_spans(words):
    """
    Compute character spans for words in a sentence.
    
    Args:
        words: List of word strings
    
    Returns:
        List of (start, end) character position tuples
    """
    spans = []
    pos = 0
    for i, w in enumerate(words):
        if i > 0:
            pos += 1  # Space between words
        start = pos
        end = start + len(w)
        spans.append((start, end))
        pos = end
    return spans


def token_to_word_alignment(offsets, word_spans):
    """
    Align token character offsets to word character spans.
    
    Args:
        offsets: List of (tok_start, tok_end) character offsets
        word_spans: List of (w_start, w_end) character spans
    
    Returns:
        List of lists, each contains token indices for that word
    """
    W = len(word_spans)
    word_to_tokens = [[] for _ in range(W)]
    
    w = 0
    for t, (ts, te) in enumerate(offsets):
        # Skip empty spans
        if te <= ts:
            continue
        
        # Advance word pointer until potential overlap
        while w < W and word_spans[w][1] <= ts:
            w += 1
        if w >= W:
            break
        
        # Token may overlap multiple words in pathological cases
        for w2 in range(w, W):
            ws, we = word_spans[w2]
            if ws >= te:
                break
            # Overlap condition
            if (ts < we) and (te > ws):
                word_to_tokens[w2].append(t)
    
    return word_to_tokens
