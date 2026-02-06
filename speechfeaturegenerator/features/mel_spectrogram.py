"""Mel spectrogram feature extraction from audio files."""

import os

import hdf5storage
import numpy as np
from scipy.interpolate import interp1d

from speechfeaturegenerator.utils.waveform import prepare_waveform
from speechfeaturegenerator.utils.io import write_summary
from speechfeaturegenerator.utils.mel_spectrogram import get_mel_spectrogram


def mel_spectrogram(
    device,
    output_root,
    stim_names,
    wav_dir,
    out_sr=100,
    n_t=None,
    time_window=[-1, 1],
    compute_original=True,
    meta_only=False,
    **kwargs,
):
    """
    Extract mel spectrogram features from audio files.
    
    Parameters:
    -----------
    device : None
        Device parameter (for compatibility with other features, not used)
    output_root : str
        Root directory for output files
    stim_names : list
        List of stimulus names (without .wav extension)
    wav_dir : str
        Directory containing audio files
    out_sr : int, optional
        Output sampling rate in Hz (default: 100)
    n_t : int, optional
        Number of time points. If None, calculated from audio duration
    time_window : list, optional
        Time window [start, end] in seconds relative to audio start (default: [-1, 1])
    compute_original : bool, optional
        Whether to compute features (default: True)
    meta_only : bool, optional
        Whether to only generate metadata (default: False)
    **kwargs : dict
        Additional parameters:
            - variant : str, optional
                Feature variant name (default: "standard")
            - nfilts : int, optional
                Number of mel bands (default: 80)
            - wintime : float, optional
                Window size in seconds (default: 0.025)
            - minfreq : int, optional
                Minimum frequency in Hz (default: 0)
            - maxfreq : int or None, optional
                Maximum frequency in Hz. If None, uses fs/2 (default: None)
    """
    variant = kwargs.get("variant", "standard")
    nfilts = kwargs.get("nfilts", 80)
    wintime = kwargs.get("wintime", 0.025)
    minfreq = kwargs.get("minfreq", 0)
    maxfreq = kwargs.get("maxfreq", None)

    if compute_original:
        for stim_name in stim_names:
            wav_path = os.path.join(wav_dir, f"{stim_name}.wav")
            
            generate_mel_spectrogram_features(
                output_root,
                wav_path,
                n_t=n_t,
                out_sr=out_sr,
                time_window=time_window,
                meta_only=meta_only,
                variant=variant,
                nfilts=nfilts,
                wintime=wintime,
                minfreq=minfreq,
                maxfreq=maxfreq,
            )


def generate_mel_spectrogram_features(
    output_root,
    wav_path,
    n_t,
    out_sr=100,
    time_window=[-1, 1],
    meta_only=False,
    variant="standard",
    nfilts=80,
    wintime=0.025,
    minfreq=0,
    maxfreq=None,
):
    """
    Generate mel spectrogram features from audio file.
    
    Parameters:
    -----------
    output_root : str
        Root directory for output files
    wav_path : str
        Path to audio file
    n_t : int, optional
        Number of time points. If None, calculated from audio duration
    out_sr : int, optional
        Output sampling rate in Hz (default: 100)
    time_window : list, optional
        Time window [start, end] in seconds relative to audio start (default: [-1, 1])
    meta_only : bool, optional
        Whether to only generate metadata (default: False)
    variant : str, optional
        Feature variant name (default: "standard")
    nfilts : int, optional
        Number of mel bands (default: 80)
    wintime : float, optional
        Window size in seconds (default: 0.025)
    minfreq : int, optional
        Minimum frequency in Hz (default: 0)
    maxfreq : int or None, optional
        Maximum frequency in Hz. If None, uses fs/2 (default: None)
    """
    feature = "mel_spectrogram"
    variants = [variant]
    
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
    
    if meta_only:
        write_summary(
            feature_variant_out_dir,
            time_window=f"{-time_window[0]} second before to {time_window[1]} second after",
            dimensions=f"[time, {nfilts} mel bands]",
            sampling_rate=out_sr,
            extra=f"Mel spectrogram with {nfilts} bands, window={wintime}s, minfreq={minfreq}Hz, maxfreq={maxfreq if maxfreq else 'fs/2'}Hz",
        )
        return
    
    # Compute mel spectrogram
    steptime = 1.0 / out_sr
    mel_spec, freqs = get_mel_spectrogram(
        waveform,
        sample_rate,
        wintime=wintime,
        steptime=steptime,
        nfilts=nfilts,
        minfreq=minfreq,
        maxfreq=maxfreq,
    )
    
    # Transpose to match expected format: [time, features]
    # mel_spec is [n_mel, n_time], need [n_time, n_mel]
    mel_spec = mel_spec.T
    
    # Ensure time dimension matches
    if mel_spec.shape[0] != len(t_new):
        # Interpolate or trim to match expected time points
        mel_spec_time = np.arange(mel_spec.shape[0]) / out_sr
        f_interp = interp1d(
            mel_spec_time,
            mel_spec,
            axis=0,
            kind="linear",
            fill_value="extrapolate",
        )
        mel_spec = f_interp(t_new)
    
    # Save features
    out_mat_path = os.path.join(feature_variant_out_dir, f"{wav_name_no_ext}.mat")
    hdf5storage.savemat(
        out_mat_path,
        {
            "features": mel_spec,
            "t": t_new,
            "freqs": freqs,
            "nfilts": nfilts,
            "sample_rate": sample_rate,
        },
    )
    
    write_summary(
        feature_variant_out_dir,
        time_window=f"{-time_window[0]} second before to {time_window[1]} second after",
        dimensions=f"[time, {nfilts} mel bands]",
        sampling_rate=out_sr,
        extra=f"Mel spectrogram with {nfilts} bands, window={wintime}s, minfreq={minfreq}Hz, maxfreq={maxfreq if maxfreq else 'fs/2'}Hz. Frequency bins: {len(freqs)}",
    )
