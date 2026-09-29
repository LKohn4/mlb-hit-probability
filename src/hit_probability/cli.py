"""Command-line entry point: ``hit-probability`` or ``python -m hit_probability``."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from hit_probability.config import Config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    defaults = Config()
    p = argparse.ArgumentParser(
        prog="hit-probability",
        description="Train and compare MLB hit-probability models on Statcast data.",
    )
    p.add_argument("--start", default=defaults.start_date, help="Start date, YYYY-MM-DD")
    p.add_argument("--end", default=defaults.end_date, help="End date, YYYY-MM-DD")
    p.add_argument("--data-dir", type=Path, default=defaults.data_dir)
    p.add_argument("--results-dir", type=Path, default=defaults.results_dir)
    p.add_argument("--refresh-data", action="store_true", help="Re-download instead of using cache")
    p.add_argument("--sample-frac", type=float, default=None,
                   help="Use a random fraction of pitches for fast iteration, e.g. 0.1")
    p.add_argument("--epochs", type=int, default=defaults.epochs)
    p.add_argument("--batch-size", type=int, default=defaults.batch_size)
    p.add_argument("--n-estimators", type=int, default=defaults.n_estimators)
    p.add_argument("--skip-nn", action="store_true", help="Train random forests only")
    p.add_argument("--skip-leak-free", action="store_true", help="Skip the leak-free experiment")
    p.add_argument("--skip-dashboard", action="store_true", help="Don't build dashboard figures")
    p.add_argument("--credit", default=defaults.credit,
                   help='Text for every figure footer, e.g. "Leo Kohn"')
    p.add_argument("--min-bip", type=int, default=defaults.min_bip_leaderboard,
                   help="Minimum balls in play for the hits-above-expected leaderboard")
    p.add_argument("--no-name-lookup", action="store_true",
                   help="Show player IDs instead of looking up names online")
    p.add_argument("-v", "--verbose", action="store_true", help="Debug logging")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")  # quiet TensorFlow start-up noise
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    config = Config(
        start_date=args.start,
        end_date=args.end,
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        refresh_data=args.refresh_data,
        sample_frac=args.sample_frac,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_estimators=args.n_estimators,
        run_nn=not args.skip_nn,
        run_leak_free=not args.skip_leak_free,
        make_dashboard=not args.skip_dashboard,
        credit=args.credit,
        min_bip_leaderboard=args.min_bip,
        lookup_names=not args.no_name_lookup,
    )

    from hit_probability import __version__
    from hit_probability.pipeline import run  # heavy imports after logging is configured

    logger = logging.getLogger("hit_probability")
    logger.info("hit_probability %s | season %s to %s", __version__, config.start_date,
                config.end_date)  # fmt: skip
    logger.info("Code: %s", Path(__file__).resolve().parent)
    logger.info("Data: %s | Results: %s", config.data_dir.resolve(), config.results_dir.resolve())

    comparison = run(config)
    print("\n" + comparison.round(4).to_string(index=False))
    return 0
