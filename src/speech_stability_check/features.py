"""Open, standard speech features.

* Praat (via parselmouth): F0 mean and SD, local jitter, local shimmer, HNR.
* Timing from a simple energy voice-activity detector: speech ratio, pause
  count, mean pause duration, speech-segment rate.
* Optional: openSMILE eGeMAPS v02 functionals (88 features).
"""

from __future__ import annotations

from typing import Dict

import numpy as np

PITCH_FLOOR = 75.0
PITCH_CEILING = 500.0


def praat_features(x: np.ndarray, sr: int) -> Dict[str, float]:
    import parselmouth
    from parselmouth.praat import call

    snd = parselmouth.Sound(x.astype(np.float64), sampling_frequency=sr)
    pitch = snd.to_pitch(time_step=0.01, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    f0 = pitch.selected_array["frequency"]
    f0 = f0[f0 > 0]
    out = {
        "f0_mean_hz": float(np.mean(f0)) if f0.size else np.nan,
        "f0_sd_hz": float(np.std(f0, ddof=1)) if f0.size > 1 else np.nan,
    }
    try:
        pp = call(snd, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)
        out["jitter_local_pct"] = 100 * call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        out["shimmer_local_pct"] = 100 * call(
            [snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except Exception:
        out["jitter_local_pct"] = np.nan
        out["shimmer_local_pct"] = np.nan
    try:
        harm = call(snd, "To Harmonicity (cc)", 0.01, PITCH_FLOOR, 0.1, 1.0)
        out["hnr_db"] = call(harm, "Get mean", 0, 0)
    except Exception:
        out["hnr_db"] = np.nan
    return {k: (float(v) if v is not None else np.nan) for k, v in out.items()}


def vad_mask(x: np.ndarray, sr: int, frame_s: float = 0.025, hop_s: float = 0.010) -> np.ndarray:
    """Boolean speech mask per hop, from frame energy in dB.

    The threshold sits 35% of the way between a noise floor (10th percentile)
    and a speech level (95th percentile), and never less than 6 dB above the
    floor. Simple and transparent by design."""
    frame = int(frame_s * sr)
    hop = int(hop_s * sr)
    if len(x) < frame:
        return np.zeros(0, dtype=bool)
    n = 1 + (len(x) - frame) // hop
    idx = np.arange(frame)[None, :] + hop * np.arange(n)[:, None]
    energy = np.mean(x[idx].astype(np.float64) ** 2, axis=1)
    db = 10 * np.log10(energy + 1e-12)
    floor, peak = np.percentile(db, 10), np.percentile(db, 95)
    thr = max(floor + 0.35 * (peak - floor), floor + 6.0)
    return db > thr


def _runs(mask: np.ndarray, value: bool):
    """Yield (start, length) of runs equal to value."""
    if mask.size == 0:
        return
    padded = np.concatenate([[not value], mask, [not value]])
    edges = np.flatnonzero(np.diff((padded == value).astype(int)))
    for s, e in zip(edges[::2], edges[1::2]):
        yield s, e - s


def timing_features(x: np.ndarray, sr: int, hop_s: float = 0.010,
                    min_pause_s: float = 0.15, min_speech_s: float = 0.05) -> Dict[str, float]:
    mask = vad_mask(x, sr, hop_s=hop_s)
    if mask.size == 0:
        return {k: np.nan for k in
                ("speech_ratio", "pause_count", "mean_pause_s", "speech_segment_rate_per_s")}
    # drop tiny speech blips, then fill short gaps
    for s, length in list(_runs(mask, True)):
        if length * hop_s < min_speech_s:
            mask[s:s + length] = False
    for s, length in list(_runs(mask, False)):
        if 0 < s and s + length < mask.size and length * hop_s < min_pause_s:
            mask[s:s + length] = True
    speech = list(_runs(mask, True))
    pauses = [length * hop_s for s, length in _runs(mask, False)
              if s > 0 and s + length < mask.size]  # internal pauses only
    duration = len(x) / sr
    return {
        "speech_ratio": float(mask.mean()),
        "pause_count": float(len(pauses)),
        "mean_pause_s": float(np.mean(pauses)) if pauses else 0.0,
        "speech_segment_rate_per_s": float(len(speech) / duration) if duration else np.nan,
    }


_SMILE = None


def egemaps_available() -> bool:
    try:
        import opensmile  # noqa: F401
        return True
    except Exception:
        return False


def egemaps_features(x: np.ndarray, sr: int) -> Dict[str, float]:
    global _SMILE
    import opensmile

    if _SMILE is None:
        _SMILE = opensmile.Smile(feature_set=opensmile.FeatureSet.eGeMAPSv02,
                                 feature_level=opensmile.FeatureLevel.Functionals)
    df = _SMILE.process_signal(x.astype(np.float32), sr)
    return {f"egemaps_{k}": float(v) for k, v in df.iloc[0].items()}


def extract_all(x: np.ndarray, sr: int, egemaps: bool = False) -> Dict[str, float]:
    feats: Dict[str, float] = {}
    feats.update(praat_features(x, sr))
    feats.update(timing_features(x, sr))
    if egemaps:
        feats.update(egemaps_features(x, sr))
    return feats
