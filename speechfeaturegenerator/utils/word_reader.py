"""Word label reader from CSV files."""

import polars as pl
import numpy as np


def load_word_labels_from_csv(csv_path, stim_name):
    """Load word labels, onsets, and offsets from CSV for a stimulus."""
    df = pl.read_csv(csv_path)

    if "stim_name" not in df.columns:
        raise ValueError("CSV file must contain a 'stim_name' column")

    stim_df = df.filter(pl.col("stim_name") == stim_name)

    if "type" in df.columns:
        stim_df = stim_df.filter(pl.col("type") == "word")
    elif "label_type" in df.columns:
        stim_df = stim_df.filter(pl.col("label_type") == "word")

    if len(stim_df) == 0:
        raise ValueError(
            f"No word data found for stimulus '{stim_name}' in CSV file"
        )

    labels = []
    onsets = []
    offsets = []

    for row in stim_df.iter_rows(named=True):
        label = str(row["label"]).strip()
        labels.append(label)
        onsets.append(float(row["onset"]))
        offsets.append(float(row["offset"]))

    return (
        np.array(labels),
        np.array(onsets, dtype=float),
        np.array(offsets, dtype=float),
    )
