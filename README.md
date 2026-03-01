# SpeechFeatureGenerator

A Python package for extracting **linguistic and acoustic speech features** from aligned audio and annotations. It supports phoneme- and word-level analyses, surprisal and entropy measures, lexical statistics, GloVe embeddings, phonotactics, and related utilities for working with waveforms and TextGrids.

## Requirements

- Python 3.8+

## Installation

From the repository root, using [uv](https://github.com/astral-sh/uv):

```bash
uv sync
```

Install the package in editable mode for development:

```bash
uv pip install -e .
```

Optional development tools:

```bash
uv sync --extra dev
```

## Package layout

- **`speechfeaturegenerator.features`** — Feature generators (e.g. phoneme, diphone, word, syllable, surprisal, entropy, lexical, word frequency, GloVe, phonotactic). Each area typically exposes a low-level helper and a `generate_*_features` function for batch-style use.
- **`speechfeaturegenerator.utils`** — I/O, embeddings, waveform helpers, TextGrid reading, phoneme inventories, and shared feature utilities (e.g. one-hot encoding).
- **`data/`** — Reference label files such as phoneme inventories used by the feature pipelines.

## Dependencies

Core dependencies include NumPy, Polars, Librosa, TextGrid, Pandas, OpenPyXL, and hdf5storage. See `pyproject.toml` for exact version constraints.

## Author

Krish Patel — [kpatel46@u.rochester.edu](mailto:kpatel46@u.rochester.edu)
