import os
from pathlib import Path
from speechfeaturegenerator.features import phoneme

# Get the directory where this script is located
script_dir = Path(__file__).parent

phoneme(
    device=None,
    output_root=str(script_dir / "out"),
    stim_names=["natural-normal"],
    wav_dir=str(script_dir),
    textgrid_path=str(script_dir / "natural-normal.TextGrid"),
    out_sr=100,
    time_window=[-1, 1],
    variant="onehot_duration"
)