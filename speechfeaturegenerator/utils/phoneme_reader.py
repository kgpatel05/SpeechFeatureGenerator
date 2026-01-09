"""Phoneme label reader from CSV files."""

import re
import polars as pl
import numpy as np


def parse_merge_format(label):
    """Parse {a,b,c} format and return first label."""
    merge_pattern = r"\{([^}]+)\}"
    match = re.search(merge_pattern, label)
    if match:
        labels = [l.strip() for l in match.group(1).split(",")]
        return labels[0] if labels else label
    return label


def build_merge_mapping(csv_path):
    """Build mapping from merged labels to canonical (first) label."""
    df = pl.read_csv(csv_path)
    merge_mapping = {}

    if "label" in df.columns:
        for label in df["label"].unique():
            merge_pattern = r"\{([^}]+)\}"
            match = re.search(merge_pattern, str(label))
            if match:
                labels = [l.strip().upper() for l in match.group(1).split(",")]
                canonical = labels[0] if labels else str(label).upper()
                for lbl in labels:
                    merge_mapping[lbl] = canonical

    return merge_mapping


def get_merge_sets(csv_path):
    """Extract canonical labels from merge groups."""
    df = pl.read_csv(csv_path)
    merge_sets = set()

    if "label" in df.columns:
        for label in df["label"].unique():
            merge_pattern = r"\{([^}]+)\}"
            match = re.search(merge_pattern, str(label))
            if match:
                labels = [l.strip().upper() for l in match.group(1).split(",")]
                if labels:
                    merge_sets.add(labels[0])

    return merge_sets


def load_phoneme_labels_from_csv(csv_path, stim_name):
    """Load phoneme labels, onsets, and offsets from CSV for a stimulus."""
    df = pl.read_csv(csv_path)

    if "stim_name" not in df.columns:
        raise ValueError("CSV file must contain a 'stim_name' column")

    stim_df = df.filter(pl.col("stim_name") == stim_name)

    if "type" in df.columns:
        stim_df = stim_df.filter(pl.col("type") == "phoneme")
    elif "label_type" in df.columns:
        stim_df = stim_df.filter(pl.col("label_type") == "phoneme")

    if len(stim_df) == 0:
        raise ValueError(f"No phoneme data found for stimulus '{stim_name}' in CSV file")

    merge_mapping = build_merge_mapping(csv_path)

    labels = []
    onsets = []
    offsets = []

    for row in stim_df.iter_rows(named=True):
        label = str(row["label"]).strip()
        label = parse_merge_format(label)
        label = label.upper()

        if label in merge_mapping:
            label = merge_mapping[label]

        labels.append(label)
        onsets.append(float(row["onset"]))
        offsets.append(float(row["offset"]))

    return (
        np.array(labels),
        np.array(onsets, dtype=float),
        np.array(offsets, dtype=float),
    )


def load_phoneme_label_set(csv_path):
    """Load phoneme label set from CSV."""
    df = pl.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError("CSV file must contain a 'label' column")

    labels = [str(label).strip().upper() for label in df["label"].to_list()]
    return sorted(set(labels))
