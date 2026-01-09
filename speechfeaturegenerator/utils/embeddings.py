"""Utilities for loading word embeddings (GLOVE)."""

import os
from pathlib import Path
import numpy as np


def get_glove_cache_dir():
    """
    Returns the cache directory for GLOVE embeddings.

    Uses ~/.cache/speechfeaturegenerator/glove/ by default.
    Can be overridden with SPEECHFEATUREGENERATOR_CACHE environment variable.

    Returns:
        Path: Path to the GLOVE cache directory.
    """
    cache_base = os.environ.get(
        "SPEECHFEATUREGENERATOR_CACHE",
        os.path.join(Path.home(), ".cache", "speechfeaturegenerator"),
    )
    glove_cache = os.path.join(cache_base, "glove")
    Path(glove_cache).mkdir(parents=True, exist_ok=True)
    return Path(glove_cache)


def load_glove_embeddings(glove_path=None, dim=50, variant="6B"):
    """
    Loads GLOVE word embeddings from a text file.

    Args:
        glove_path (str, optional): Path to GLOVE embeddings file. If None, looks in cache directory.
        dim (int, optional): Embedding dimension (50, 100, 200, or 300). Defaults to 50.
            Only used if glove_path is None to construct default filename.
        variant (str, optional): GLOVE variant (e.g., "6B", "840B", "twitter.27B"). Defaults to "6B".
            Only used if glove_path is None to construct default filename.

    Returns:
        dict: Dictionary mapping words to numpy arrays of embeddings.
    """
    if glove_path is None:
        cache_dir = get_glove_cache_dir()
        glove_path = cache_dir / f"glove.{variant}.{dim}d.txt"

    glove_path = Path(glove_path)

    if not glove_path.exists():
        raise FileNotFoundError(
            f"GLOVE embeddings file not found: {glove_path}\n"
            f"Please download GLOVE embeddings from https://nlp.stanford.edu/projects/glove/\n"
            f"and place them in: {glove_path}\n"
            f"Or set the glove_path parameter to point to your embeddings file."
        )

    embeddings = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            embeddings[word] = vector

    return embeddings


def get_word_embedding(word, embeddings, lowercase=True, default=None):
    """
    Retrieves embedding for a word, handling case variations and missing words.

    Args:
        word (str): Word to get embedding for.
        embeddings (dict): Dictionary of word embeddings.
        lowercase (bool, optional): Whether to try lowercase version if not found. Defaults to True.
        default (np.ndarray, optional): Default embedding to return if word not found. 
            If None, returns zero vector with same dimension as embeddings.

    Returns:
        np.ndarray: Word embedding vector.
    """
    word_clean = word.strip()

    if word_clean in embeddings:
        return embeddings[word_clean]

    if lowercase:
        word_lower = word_clean.lower()
        if word_lower in embeddings:
            return embeddings[word_lower]

    if default is not None:
        return default

    if len(embeddings) > 0:
        sample_embedding = next(iter(embeddings.values()))
        return np.zeros_like(sample_embedding)

    raise ValueError("No embeddings available and no default provided")

