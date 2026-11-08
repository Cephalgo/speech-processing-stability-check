"""Synthetic speech-like test audio (no recordings, no text-to-speech).

A classic source-filter model: a Rosenberg glottal pulse train with a slowly
varying F0 plus small cycle-to-cycle jitter and shimmer, shaped by a cascade
of formant resonators for a few vowels, arranged into "syllables" separated by
short gaps and longer pauses, with optional background noise at a set SNR.
The clips contain no human voice and carry no licence restrictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from scipy import signal as sps

VOWELS = {  # approximate formant frequencies (Hz) and bandwidths
    "a": ([730, 1090, 2440, 3400], [90, 110, 160, 250]),
    "i": ([270, 2290, 3010, 3700], [60, 100, 170, 250]),
    "u": ([300, 870, 2240, 3300], [60, 90, 150, 250]),
    "e": ([530, 1840, 2480, 3500], [70, 100, 160, 250]),
    "o": ([570, 840, 2410, 3400], [80, 100, 160, 250]),
}


def _rosenberg(n: int, open_q: float = 0.6, close_q: float = 0.15) -> np.ndarray:
    t1 = max(1, int(open_q * n))
    t2 = max(1, int(close_q * n))
    g = np.zeros(n)
    k = np.arange(t1)
    g[:t1] = 0.5 * (1 - np.cos(np.pi * k / t1))
    k = np.arange(min(t2, n - t1))
    g[t1:t1 + len(k)] = np.cos(0.5 * np.pi * k / t2)
    return g


def glottal_source(duration_s: float, sr: int, f0: float, rng: np.random.Generator,
                   jitter: float = 0.005, shimmer: float = 0.03,
                   f0_wobble: float = 0.06) -> np.ndarray:
    n_total = int(duration_s * sr)
    out = np.zeros(n_total + sr // 20)
    pos = 0
    t = 0.0
    phase = rng.uniform(0, 2 * np.pi)
    while pos < n_total:
        f = f0 * (1 + f0_wobble * np.sin(2 * np.pi * 0.7 * t + phase))
        f *= 1 + jitter * rng.standard_normal()
        period = max(2, int(round(sr / f)))
        amp = 1 + shimmer * rng.standard_normal()
        pulse = _rosenberg(period)
        end = min(pos + period, len(out))
        out[pos:end] += amp * pulse[:end - pos]
        pos += period
        t += period / sr
    src = np.diff(out[:n_total + 1])  # radiated flow derivative
    return src


def formant_filter(x: np.ndarray, sr: int, vowel: str) -> np.ndarray:
    freqs, bws = VOWELS[vowel]
    y = x
    for f, bw in zip(freqs, bws):
        if f >= sr / 2:
            continue
        r = np.exp(-np.pi * bw / sr)
        theta = 2 * np.pi * f / sr
        a = [1.0, -2 * r * np.cos(theta), r * r]
        b = [1 - r]
        y = sps.lfilter(b, a, y)
    return y


def synth_utterance(sr: int = 16000, duration_s: float = 6.0, f0: float = 120.0,
                    snr_db: Optional[float] = 20.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(duration_s * sr)
    y = np.zeros(n)
    pos = int(0.3 * sr)
    vowels = list(VOWELS)
    while pos < n - int(0.4 * sr):
        dur = rng.uniform(0.15, 0.35)
        m = int(dur * sr)
        v = vowels[rng.integers(len(vowels))]
        seg = formant_filter(glottal_source(dur, sr, f0 * rng.uniform(0.95, 1.05), rng), sr, v)
        seg = seg[:m]
        env = np.ones(len(seg))
        ramp = min(len(seg) // 4, int(0.03 * sr))
        if ramp > 0:
            env[:ramp] = np.linspace(0, 1, ramp)
            env[-ramp:] = np.linspace(1, 0, ramp)
        seg = seg * env * rng.uniform(0.6, 1.0)
        end = min(pos + len(seg), n)
        y[pos:end] += seg[:end - pos]
        gap = rng.uniform(0.6, 1.0) if rng.random() < 0.18 else rng.uniform(0.04, 0.10)
        pos = end + int(gap * sr)
    y /= np.max(np.abs(y)) + 1e-12
    y *= 0.5
    if snr_db is not None:
        # pink-ish background noise scaled to the requested SNR over voiced parts
        white = rng.standard_normal(n)
        pink = sps.lfilter([0.049922035, -0.095993537, 0.050612699, -0.004408786],
                           [1, -2.494956002, 2.017265875, -0.522189400], white)
        active = np.abs(y) > 1e-4
        p_sig = np.mean(y[active] ** 2) if active.any() else np.mean(y ** 2)
        p_noise = np.mean(pink ** 2)
        pink *= np.sqrt(p_sig / (p_noise * 10 ** (snr_db / 10)))
        y = y + pink
    peak = np.max(np.abs(y))
    if peak > 0.95:
        y *= 0.95 / peak
    return y.astype(np.float32)


def write_demo_set(out_dir: str | Path, n_files: int = 8, sr: int = 16000,
                   seed: int = 1) -> List[Path]:
    import soundfile as sf

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    paths = []
    for i in range(n_files):
        f0 = float(rng.choice([rng.uniform(95, 140), rng.uniform(180, 240)]))
        snr = float(rng.uniform(10, 30))
        y = synth_utterance(sr=sr, duration_s=float(rng.uniform(5, 8)), f0=f0,
                            snr_db=snr, seed=seed * 1000 + i)
        p = out_dir / f"synthetic_{i:02d}_f0-{f0:.0f}_snr-{snr:.0f}.wav"
        sf.write(p, y, sr, subtype="PCM_16")
        paths.append(p)
    return paths
