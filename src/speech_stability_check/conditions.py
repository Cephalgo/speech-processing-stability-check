"""Processing conditions that simulate what phones, apps and software updates
can do to a recording before it reaches an analysis pipeline.

Every condition takes a mono float32 signal in [-1, 1] and its sample rate and
returns a signal at the same sample rate (so features are always computed on
comparable inputs). Conditions whose optional dependency is missing raise
``ConditionUnavailable`` and are skipped by the runner.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from math import gcd
from typing import Callable, Dict, List

import numpy as np
from scipy import signal as sps


class ConditionUnavailable(RuntimeError):
    """Raised when a condition cannot run on this machine."""


@dataclass(frozen=True)
class Condition:
    name: str
    description: str
    func: Callable[[np.ndarray, int], np.ndarray]


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    g = gcd(sr_in, sr_out)
    return sps.resample_poly(x, sr_out // g, sr_in // g).astype(np.float32)


def _match_length(y: np.ndarray, n: int) -> np.ndarray:
    if len(y) >= n:
        return y[:n]
    return np.pad(y, (0, n - len(y)))


def _clip(x: np.ndarray) -> np.ndarray:
    return np.clip(x, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------- conditions

def original(x: np.ndarray, sr: int) -> np.ndarray:
    return x.astype(np.float32, copy=True)


def _noisereduce(x: np.ndarray, sr: int, prop_decrease: float) -> np.ndarray:
    try:
        import noisereduce as nr
    except ImportError as exc:  # pragma: no cover
        raise ConditionUnavailable("noisereduce is not installed") from exc
    y = nr.reduce_noise(y=x, sr=sr, stationary=False, prop_decrease=prop_decrease)
    return _clip(_match_length(np.asarray(y, dtype=np.float32), len(x)))


def spectral_gate_mild(x: np.ndarray, sr: int) -> np.ndarray:
    return _noisereduce(x, sr, prop_decrease=0.5)


def spectral_gate_strong(x: np.ndarray, sr: int) -> np.ndarray:
    return _noisereduce(x, sr, prop_decrease=1.0)


def hpf_agc(x: np.ndarray, sr: int) -> np.ndarray:
    """Approximation of a WebRTC-style voice front end: 80 Hz high-pass
    filter followed by a frame-wise automatic gain control and a soft limiter.
    This is a simple open approximation, not the WebRTC code itself."""
    sos = sps.butter(2, 80.0, btype="highpass", fs=sr, output="sos")
    y = sps.sosfilt(sos, x).astype(np.float64)

    frame = max(1, int(0.010 * sr))
    n_frames = int(np.ceil(len(y) / frame))
    rms = np.array([np.sqrt(np.mean(y[i * frame:(i + 1) * frame] ** 2))
                    for i in range(n_frames)])
    rms_db = 20 * np.log10(rms + 1e-12)
    # crude level-based activity gate (WebRTC's AGC adapts only on detected voice)
    active = rms_db > np.percentile(rms_db, 95) - 25.0
    target_rms = 10 ** (-18.0 / 20.0)  # -18 dBFS
    max_gain = 10 ** (20.0 / 20.0)  # +20 dB
    attack, release = 0.3, 0.02  # smoothing per 10 ms frame
    gain = 1.0
    gains = np.empty(n_frames)
    for i in range(n_frames):
        if active[i]:
            desired = min(max_gain, target_rms / rms[i])
            coef = attack if desired < gain else release
            gain = gain + coef * (desired - gain)
        gains[i] = gain  # hold gain outside detected voice
    # interpolate the gain curve sample by sample to avoid steps
    centres = (np.arange(n_frames) + 0.5) * frame
    g = np.interp(np.arange(len(y)), centres, gains)
    y = np.tanh(y * g)  # soft limiter
    return _clip(y)


def telephone_8k(x: np.ndarray, sr: int) -> np.ndarray:
    """Narrow-band telephone channel: resample to 8 kHz, 300-3400 Hz band-pass,
    resample back to the original rate."""
    y = _resample(x, sr, 8000)
    sos = sps.butter(4, [300.0, 3400.0], btype="bandpass", fs=8000, output="sos")
    y = sps.sosfilt(sos, y)
    return _clip(_match_length(_resample(y, 8000, sr), len(x)))


def _ffmpeg_roundtrip(x: np.ndarray, sr: int, encode_args: List[str], ext: str,
                      encoder: str) -> np.ndarray:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise ConditionUnavailable("ffmpeg not found on PATH")
    if not _ffmpeg_has_encoder(ffmpeg, encoder):
        raise ConditionUnavailable(f"ffmpeg has no '{encoder}' encoder")
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "in.wav")
        enc = os.path.join(tmp, "enc." + ext)
        dec = os.path.join(tmp, "dec.wav")
        sf.write(src, x, sr, subtype="PCM_16")
        base = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
        subprocess.run(base + ["-i", src] + encode_args + [enc], check=True)
        subprocess.run(base + ["-i", enc, "-ac", "1", "-ar", str(sr), dec], check=True)
        y, _ = sf.read(dec, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    return _clip(_match_length(y, len(x)))


_ENCODER_CACHE: Dict[str, bool] = {}


def _ffmpeg_has_encoder(ffmpeg: str, encoder: str) -> bool:
    if encoder not in _ENCODER_CACHE:
        out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                             capture_output=True, text=True).stdout
        _ENCODER_CACHE[encoder] = any(
            line.split()[1:2] == [encoder] for line in out.splitlines() if line.strip()
        )
    return _ENCODER_CACHE[encoder]


def opus_12k(x: np.ndarray, sr: int) -> np.ndarray:
    """Low-bitrate Opus (12 kbit/s, VoIP mode) round trip via ffmpeg."""
    return _ffmpeg_roundtrip(
        x, sr,
        ["-c:a", "libopus", "-b:a", "12k", "-application", "voip", "-ar", "16000"],
        "ogg", "libopus",
    )


def amr_nb(x: np.ndarray, sr: int) -> np.ndarray:
    """AMR-NB (12.2 kbit/s) round trip via ffmpeg, if ffmpeg was built with
    libopencore-amrnb."""
    return _ffmpeg_roundtrip(
        x, sr,
        ["-c:a", "libopencore_amrnb", "-b:a", "12.2k", "-ar", "8000", "-ac", "1"],
        "amr", "libopencore_amrnb",
    )


def rnnoise(x: np.ndarray, sr: int) -> np.ndarray:
    """RNNoise recurrent-network noise suppression (via the pyrnnoise package)."""
    try:
        from pyrnnoise import RNNoise
    except Exception as exc:
        raise ConditionUnavailable("pyrnnoise is not installed") from exc
    y48 = _resample(x, sr, 48000)
    pcm = (np.clip(y48, -1, 1) * 32767).astype(np.int16)
    den = RNNoise(sample_rate=48000)
    frames = [frame for _, frame in den.denoise_chunk(pcm, partial=True)]
    if not frames:
        raise ConditionUnavailable("pyrnnoise returned no audio")
    out = np.concatenate([np.atleast_2d(f)[0] for f in frames]).astype(np.float32) / 32768.0
    return _clip(_match_length(_resample(out, 48000, sr), len(x)))


CONDITIONS: Dict[str, Condition] = {
    c.name: c for c in [
        Condition("original", "Unprocessed input", original),
        Condition("spectral_gate_mild", "noisereduce spectral gating, 50% reduction", spectral_gate_mild),
        Condition("spectral_gate_strong", "noisereduce spectral gating, 100% reduction", spectral_gate_strong),
        Condition("rnnoise", "RNNoise neural noise suppression (pyrnnoise)", rnnoise),
        Condition("hpf_agc", "80 Hz high-pass + automatic gain control (WebRTC-style approximation)", hpf_agc),
        Condition("opus_12k", "Opus 12 kbit/s VoIP round trip (ffmpeg)", opus_12k),
        Condition("amr_nb", "AMR-NB 12.2 kbit/s round trip (ffmpeg with libopencore-amrnb)", amr_nb),
        Condition("telephone_8k", "8 kHz telephone band, 300-3400 Hz", telephone_8k),
    ]
}

DEFAULT_CONDITIONS = list(CONDITIONS)
