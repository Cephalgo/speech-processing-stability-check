import numpy as np
import pandas as pd
import pytest

from speech_stability_check import cli
from speech_stability_check.conditions import CONDITIONS, ConditionUnavailable
from speech_stability_check.features import extract_all, timing_features
from speech_stability_check.stats import icc_2_1, summarise
from speech_stability_check.synth import synth_utterance, write_demo_set

SR = 16000


@pytest.fixture(scope="module")
def clip():
    return synth_utterance(sr=SR, duration_s=3.0, f0=150.0, snr_db=25.0, seed=3)


# ------------------------------------------------------------------ stats

def test_icc_perfect_agreement():
    rng = np.random.default_rng(0)
    col = rng.normal(size=20)
    assert icc_2_1(np.column_stack([col, col, col])) == pytest.approx(1.0)


def test_icc_shrout_fleiss_example():
    # Shrout & Fleiss (1979) Table 2 data; published ICC(2,1) = 0.29
    data = np.array([[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8],
                     [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]], dtype=float)
    assert icc_2_1(data) == pytest.approx(0.29, abs=0.01)


def test_icc_offset_lowers_absolute_agreement():
    rng = np.random.default_rng(1)
    col = rng.normal(size=30)
    assert icc_2_1(np.column_stack([col, col + 2.0])) < 0.5


def test_summarise_threshold_and_shift():
    df = pd.DataFrame({
        "file": ["a", "b", "a", "b"],
        "condition": ["original", "original", "x", "x"],
        "feat": [100.0, 200.0, 105.0, 260.0],
    })
    s = summarise(df, threshold_pct=10).set_index(["feature", "condition"])
    row = s.loc[("feat", "x")]
    assert row["mean_shift"] == pytest.approx(32.5)
    assert row["pct_files_over_threshold"] == pytest.approx(50.0)
    assert np.isfinite(s.loc[("feat", "ALL"), "icc21_all_conditions"])


# --------------------------------------------------------------- synthesis

def test_synthetic_clip_is_bounded(clip):
    assert clip.dtype == np.float32
    assert len(clip) == 3 * SR
    assert np.max(np.abs(clip)) <= 0.95 + 1e-6


# ---------------------------------------------------------------- features

def test_praat_f0_recovers_synthetic_pitch(clip):
    feats = extract_all(clip, SR)
    assert feats["f0_mean_hz"] == pytest.approx(150.0, rel=0.1)
    for key in ("jitter_local_pct", "shimmer_local_pct", "hnr_db"):
        assert np.isfinite(feats[key])


def test_timing_features_find_pauses():
    x = np.zeros(SR * 3, dtype=np.float32)
    t = np.arange(int(0.5 * SR)) / SR
    tone = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    x[int(0.2 * SR):int(0.7 * SR)] = tone
    x[int(1.5 * SR):int(2.0 * SR)] = tone
    x += 1e-4 * np.random.default_rng(0).standard_normal(len(x)).astype(np.float32)
    f = timing_features(x, SR)
    assert f["pause_count"] == 1
    assert f["mean_pause_s"] == pytest.approx(0.8, abs=0.05)
    assert f["speech_ratio"] == pytest.approx(1.0 / 3.0, abs=0.05)


# -------------------------------------------------------------- conditions

@pytest.mark.parametrize("name", list(CONDITIONS))
def test_condition_keeps_length_and_range(name, clip):
    try:
        y = CONDITIONS[name].func(clip, SR)
    except ConditionUnavailable as exc:
        pytest.skip(str(exc))
    assert len(y) == len(clip)
    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) <= 1.0


def test_telephone_removes_high_band(clip):
    y = CONDITIONS["telephone_8k"].func(clip, SR)
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1 / SR)
    assert spec[freqs > 4500].sum() < 1e-3 * spec.sum()


# --------------------------------------------------------------------- CLI

def test_cli_end_to_end(tmp_path):
    audio = tmp_path / "audio"
    write_demo_set(audio, n_files=3, seed=7)
    out = tmp_path / "out"
    rc = cli.main(["run", str(audio), "-o", str(out), "-c", "spectral_gate_strong",
                   "telephone_8k", "hpf_agc", "--plot", "-t", "5"])
    assert rc == 0
    feats = pd.read_csv(out / "features.csv")
    summary = pd.read_csv(out / "summary.csv")
    assert set(feats["condition"]) == {"original", "spectral_gate_strong", "telephone_8k", "hpf_agc"}
    assert len(feats) == 3 * 4
    assert {"f0_mean_hz", "hnr_db", "speech_ratio"} <= set(summary["feature"])
    assert (out / "stability_heatmap.png").stat().st_size > 1000
