"""TextGrid file reader for loading phoneme and word labels."""

import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def load_phoneme_labels_from_textgrid(
    textgrid_path: str,
    tier_name: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load phoneme labels, onsets, and offsets from a TextGrid file.
    
    Args:
        textgrid_path: Path to the TextGrid file
        tier_name: Name of the tier to read (default: "phones" or "phonemes")
    
    Returns:
        Tuple of (labels, onsets, offsets) as numpy arrays
    """
    try:
        from textgrid import TextGrid
    except ImportError:
        raise ImportError(
            "textgrid package required. Install with: pip install textgrid"
        )
    
    textgrid_path = Path(textgrid_path)
    if not textgrid_path.exists():
        raise FileNotFoundError(f"TextGrid file not found: {textgrid_path}")
    
    # Load TextGrid
    tg = TextGrid.fromFile(str(textgrid_path))
    
    # Find phoneme tier (try common names)
    if tier_name is None:
        tier_names = ["phones", "phonemes", "phone", "phoneme"]
        tier = None
        for name in tier_names:
            try:
                tier = tg.getFirst(name)
                break
            except:
                continue
        if tier is None:
            available = [tier.name for tier in tg.tiers]
            raise ValueError(
                f"No phoneme tier found. Available tiers: {available}. "
                f"Expected one of: {tier_names}"
            )
    else:
        tier = tg.getFirst(tier_name)
        if tier is None:
            available = [tier.name for tier in tg.tiers]
            raise ValueError(
                f"Tier '{tier_name}' not found. Available tiers: {available}"
            )
    
    # Extract labels, onsets, and offsets
    labels = []
    onsets = []
    offsets = []
    
    for interval in tier:
        # Skip empty intervals
        if interval.mark and interval.mark.strip():
            labels.append(interval.mark.strip())
            onsets.append(interval.minTime)
            offsets.append(interval.maxTime)
    
    return (
        np.array(labels),
        np.array(onsets, dtype=float),
        np.array(offsets, dtype=float),
    )


def load_word_labels_from_textgrid(
    textgrid_path: str,
    tier_name: Optional[str] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load word labels, onsets, and offsets from a TextGrid file.
    
    Args:
        textgrid_path: Path to the TextGrid file
        tier_name: Name of the tier to read (default: "words" or "word")
    
    Returns:
        Tuple of (labels, onsets, offsets) as numpy arrays
    """
    try:
        from textgrid import TextGrid
    except ImportError:
        raise ImportError(
            "textgrid package required. Install with: pip install textgrid"
        )
    
    textgrid_path = Path(textgrid_path)
    if not textgrid_path.exists():
        raise FileNotFoundError(f"TextGrid file not found: {textgrid_path}")
    
    # Load TextGrid
    tg = TextGrid.fromFile(str(textgrid_path))
    
    # Find word tier (try common names)
    if tier_name is None:
        tier_names = ["words", "word", "Words", "Word"]
        tier = None
        for name in tier_names:
            try:
                tier = tg.getFirst(name)
                break
            except:
                continue
        if tier is None:
            available = [tier.name for tier in tg.tiers]
            raise ValueError(
                f"No word tier found. Available tiers: {available}. "
                f"Expected one of: {tier_names}"
            )
    else:
        tier = tg.getFirst(tier_name)
        if tier is None:
            available = [tier.name for tier in tg.tiers]
            raise ValueError(
                f"Tier '{tier_name}' not found. Available tiers: {available}"
            )
    
    # Extract labels, onsets, and offsets
    labels = []
    onsets = []
    offsets = []
    
    for interval in tier:
        # Skip empty intervals
        if interval.mark and interval.mark.strip():
            labels.append(interval.mark.strip())
            onsets.append(interval.minTime)
            offsets.append(interval.maxTime)
    
    return (
        np.array(labels),
        np.array(onsets, dtype=float),
        np.array(offsets, dtype=float),
    )
