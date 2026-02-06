"""Mel spectrogram computation utilities."""

import numpy as np
from scipy.signal import spectrogram
from scipy.signal.windows import hann as hanning


def hz2mel(f, htk=False):
    """Convert frequencies in Hz to mel scale."""
    if htk:
        return 2595 * np.log10(1 + f / 700)
    else:
        f_0 = 0.0
        f_sp = 200.0 / 3
        brkfrq = 1000.0
        brkpt = (brkfrq - f_0) / f_sp
        logstep = np.exp(np.log(6.4) / 27.0)
        
        linpts = f < brkfrq
        z = np.zeros_like(f)
        
        if np.isscalar(f):
            if linpts:
                z = (f - f_0) / f_sp
            else:
                z = brkpt + np.log(f / brkfrq) / np.log(logstep)
        else:
            z[linpts] = (f[linpts] - f_0) / f_sp
            z[~linpts] = brkpt + np.log(f[~linpts] / brkfrq) / np.log(logstep)
        return z


def mel2hz(z, htk=False):
    """Convert mel scale frequencies to Hz."""
    if htk:
        return 700.0 * (10 ** (z / 2595.0) - 1)
    else:
        f_0 = 0.0
        f_sp = 200.0 / 3
        brkfrq = 1000.0
        brkpt = (brkfrq - f_0) / f_sp
        logstep = np.exp(np.log(6.4) / 27.0)
        
        linpts = z < brkpt
        f = np.zeros_like(z)
        
        if np.isscalar(z):
            if linpts:
                f = f_0 + f_sp * z
            else:
                f = brkfrq * np.exp(np.log(logstep) * (z - brkpt))
        else:
            f[linpts] = f_0 + f_sp * z[linpts]
            f[~linpts] = brkfrq * np.exp(np.log(logstep) * (z[~linpts] - brkpt))
        return f


def fft2melmx(nfft, sr=8000, nfilts=0, bwidth=1.0, minfreq=0, maxfreq=4000, htkmel=False, constamp=0):
    """Generate mel filterbank matrix."""
    if nfilts == 0:
        nfilts = int(np.ceil(hz2mel(maxfreq, htkmel) / 2))
    
    wts = np.zeros((nfilts, nfft))
    fftfrqs = np.arange(0, nfft / 2) / nfft * sr
    
    minmel = hz2mel(minfreq, htkmel)
    maxmel = hz2mel(maxfreq, htkmel)
    binfrqs = mel2hz(minmel + np.arange(0, nfilts + 2) / (nfilts + 2) * (maxmel - minmel), htkmel)
    
    for i in range(nfilts):
        fs = binfrqs[i + np.array([0, 1, 2])]
        fs = fs[1] + bwidth * (fs - fs[1])
        
        loslope = (fftfrqs - fs[0]) / (fs[1] - fs[0])
        hislope = (fs[2] - fftfrqs) / (fs[2] - fs[1])
        w = np.minimum(loslope, hislope)
        w[w < 0] = 0
        wts[i, 0:int(nfft / 2)] = w
    
    if constamp == 0:
        wts = np.dot(np.diag(2.0 / (binfrqs[2 + np.arange(nfilts)] - binfrqs[np.arange(nfilts)])), wts)
    
    wts[:, int(nfft / 2 + 2):int(nfft)] = 0
    return wts, binfrqs


def powspec(x, sr=8000, wintime=0.025, steptime=0.010, dither=1):
    """Compute power spectrogram."""
    winpts = int(np.round(wintime * sr))
    steppts = int(np.round(steptime * sr))
    
    NFFT = int(2 ** np.ceil(np.log2(winpts)))
    WINDOW = hanning(winpts)
    NOVERLAP = winpts - steppts
    
    f, t, Sxx = spectrogram(x * 32768, nfft=NFFT, fs=sr, nperseg=len(WINDOW), 
                            window=WINDOW, noverlap=NOVERLAP)
    y = np.abs(Sxx) ** 2
    
    if dither:
        y = y + winpts
    
    e = np.log(np.sum(y))
    return y, e


def audspec(pspectrum, sr=16000, nfilts=80, fbtype='mel', minfreq=0, maxfreq=8000, 
            sumpower=True, bwidth=1.0):
    """Perform critical band analysis (mel scaling)."""
    nfreqs, nframes = pspectrum.shape
    nfft = int((nfreqs - 1) * 2)
    
    if fbtype == 'mel':
        wts, freqs = fft2melmx(nfft=nfft, sr=sr, nfilts=nfilts, bwidth=bwidth, 
                              minfreq=minfreq, maxfreq=maxfreq)
    else:
        raise ValueError(f"fbtype '{fbtype}' not recognized")
    
    wts = wts[:, 0:nfreqs]
    
    if sumpower:
        aspectrum = np.dot(wts, pspectrum)
    else:
        aspectrum = np.dot(wts, np.sqrt(pspectrum)) ** 2
    
    return aspectrum, wts, freqs


def get_mel_spectrogram(w, fs, wintime=0.025, steptime=0.010, nfilts=80, 
                        minfreq=0, maxfreq=None):
    """
    Compute mel-band spectrogram.
    
    Parameters:
    -----------
    w : array-like
        Audio signal vector
    fs : int
        Sampling rate of audio signal
    wintime : float
        Window size in seconds (default: 0.025)
    steptime : float
        Step size in seconds (default: 0.010)
    nfilts : int
        Number of mel-band filters (default: 80)
    minfreq : int
        Minimum frequency to analyze in Hz (default: 0)
    maxfreq : int or None
        Maximum frequency to analyze in Hz. If None, defaults to fs/2
    
    Returns:
    --------
    mel_spectrogram : array
        Mel-band spectrogram [n_mel, n_time]
    freqs : array
        Array of frequency bin edges
    """
    if maxfreq is None:
        maxfreq = int(fs / 2)
    
    pspec, e = powspec(w, sr=fs, wintime=wintime, steptime=steptime, dither=1)
    aspectrum, wts, freqs = audspec(pspec, sr=fs, nfilts=nfilts, fbtype='mel', 
                                    minfreq=minfreq, maxfreq=maxfreq, 
                                    sumpower=True, bwidth=1.0)
    mel_spectrogram = aspectrum ** 0.001
    
    return mel_spectrogram, freqs
