"""Command-line interface.

    speech-stability-check run INPUT_DIR -o OUTPUT_DIR [options]
    speech-stability-check make-synthetic OUTPUT_DIR [--n 8]
    speech-stability-check list-conditions
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import __version__
from .conditions import CONDITIONS, DEFAULT_CONDITIONS, ConditionUnavailable
from .features import egemaps_available, extract_all
from .stats import summarise

log = logging.getLogger("speech_stability_check")


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    return x, int(sr)


def run(input_dir: Path, output_dir: Path, conditions: List[str],
        threshold_pct: float = 10.0, egemaps: bool = False, plot: bool = False,
        save_audio: bool = False) -> Dict[str, Path]:
    files = sorted(p for p in Path(input_dir).rglob("*") if p.suffix.lower() == ".wav")
    if not files:
        raise SystemExit(f"No .wav files found in {input_dir}")
    if "original" not in conditions:
        conditions = ["original"] + conditions
    output_dir.mkdir(parents=True, exist_ok=True)
    if egemaps and not egemaps_available():
        log.warning("opensmile is not installed; skipping eGeMAPS features")
        egemaps = False

    skipped: Dict[str, str] = {}
    rows = []
    for path in files:
        x, sr = load_wav(path)
        rel = str(path.relative_to(input_dir))
        log.info("%s (%.1f s, %d Hz)", rel, len(x) / sr, sr)
        for name in conditions:
            if name in skipped:
                continue
            cond = CONDITIONS[name]
            try:
                y = cond.func(x, sr)
            except ConditionUnavailable as exc:
                skipped[name] = str(exc)
                log.warning("Skipping condition '%s': %s", name, exc)
                continue
            if save_audio:
                import soundfile as sf

                dest = output_dir / "audio" / name / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(dest), y, sr, subtype="PCM_16")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                feats = extract_all(y, sr, egemaps=egemaps)
            rows.append({"file": rel, "condition": name, **feats})

    features = pd.DataFrame(rows)
    summary = summarise(features, threshold_pct=threshold_pct)
    out = {
        "features": output_dir / "features.csv",
        "summary": output_dir / "summary.csv",
    }
    features.to_csv(out["features"], index=False)
    summary.to_csv(out["summary"], index=False)
    if plot:
        from .plot import plot_summary

        out["plot"] = plot_summary(summary, output_dir / "stability_heatmap.png",
                                   threshold_pct=threshold_pct)
    if skipped:
        pd.Series(skipped, name="reason").rename_axis("condition").to_csv(
            output_dir / "skipped_conditions.csv")
    _print_table(summary, threshold_pct, len(files))
    return out


def _print_table(summary: pd.DataFrame, threshold_pct: float, n_files: int) -> None:
    core = summary[~summary["feature"].str.startswith("egemaps_")]
    per = core[core["condition"] != "ALL"]
    allc = core[core["condition"] == "ALL"].set_index("feature")["icc21_all_conditions"]
    tab = per.pivot(index="feature", columns="condition", values="median_abs_change_pct")
    cond_order = [c for c in per["condition"].unique() if c in tab.columns]
    tab = tab.loc[[f for f in core["feature"].unique() if f in tab.index], cond_order]
    tab["ICC(2,1) all"] = allc
    with pd.option_context("display.width", 200, "display.max_columns", 20,
                           "display.float_format", "{:.2f}".format):
        print(f"\nMedian absolute change vs original (%), {n_files} files; "
              f"last column = ICC(2,1) across all conditions")
        print(tab.to_string())
        print(f"\nFull table (incl. % files over the {threshold_pct:g}% threshold): summary.csv\n")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="speech-stability-check",
        description="Measure how much standard speech features shift when the same "
                    "recordings pass through different processing conditions.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="Process a folder of WAV files and write CSV results")
    r.add_argument("input_dir", type=Path)
    r.add_argument("-o", "--output-dir", type=Path, default=Path("stability_results"))
    r.add_argument("-c", "--conditions", nargs="+", default=DEFAULT_CONDITIONS,
                   choices=list(CONDITIONS), metavar="NAME",
                   help="Conditions to apply (default: all available). "
                        "See 'list-conditions'.")
    r.add_argument("-t", "--threshold", type=float, default=10.0,
                   help="Relative change (%%) counted as a meaningful shift (default 10)")
    r.add_argument("--egemaps", action="store_true",
                   help="Also extract openSMILE eGeMAPS v02 functionals (needs opensmile)")
    r.add_argument("--plot", action="store_true", help="Write stability_heatmap.png")
    r.add_argument("--save-audio", action="store_true",
                   help="Also write the processed audio for listening checks")

    s = sub.add_parser("make-synthetic", help="Write synthetic speech-like test clips")
    s.add_argument("output_dir", type=Path)
    s.add_argument("--n", type=int, default=8)
    s.add_argument("--seed", type=int, default=1)

    sub.add_parser("list-conditions", help="Show the available processing conditions")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")
    if args.command == "run":
        out = run(args.input_dir, args.output_dir, list(args.conditions), args.threshold,
                  args.egemaps, args.plot, args.save_audio)
        for k, v in out.items():
            print(f"{k}: {v}")
    elif args.command == "make-synthetic":
        from .synth import write_demo_set

        for pth in write_demo_set(args.output_dir, n_files=args.n, seed=args.seed):
            print(pth)
    elif args.command == "list-conditions":
        for c in CONDITIONS.values():
            print(f"{c.name:22s} {c.description}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
