"""Nightly break-risk retrain job.

Retrains the Stage-2 break-risk model on the last 30 days of FS100-broken
labels vs good plates, then exits. Intended to be scheduled (Windows Task
Scheduler / cron). The dashboard's "Retrain now" button calls the same
``metrology.retrain_break_model`` function.

Run from the ``dashboard`` folder so ``config``/``metrology`` import cleanly::

    conda activate fs50defect
    cd dashboard
    python retrain_nightly.py            # live labels
    python retrain_nightly.py --days 60  # wider history

Schedule (Task Scheduler, daily 02:00) the equivalent of::

    <conda python> C:\\Users\\FS134918\\FS50CornerMetrology\\dashboard\\retrain_nightly.py
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

import metrology


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain the break-risk model.")
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="How many days of history to use for labels/features (default 30).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use synthetic data (offline smoke test).",
    )
    args = parser.parse_args()

    end = datetime.now()
    start = end - timedelta(days=args.days)
    summary = metrology.retrain_break_model(
        start, end, mock=args.mock or None, min_history_days=args.days
    )
    print(f"[{datetime.now():%Y-%m-%d %H:%M}] {summary}")


if __name__ == "__main__":
    main()
