"""Headless 24/7 alert watcher.

Runs the SAME Teams-alert dispatch the dashboard performs, but WITHOUT needing a
browser tab open. Intended to run continuously (Windows Task Scheduler at logon,
"restart if it stops"). Because it is a single long-lived process, the in-memory
cooldown in :mod:`alerts` (``_last_fired``) works correctly, so alerts respect
their cooldown and never spam.

Each cycle it:
  1. (optionally) ingests + scores any new corner images (``scan_and_process``),
  2. builds the run-length edge-grind defect alerts over a rolling window and
     fires the FS50 defect-burst Teams alert (>=3 defects / 15 min per grinder),
  3. builds the FS100-broken backtrack over a rolling window and fires the
     FS100-broken Teams alert (once per plate).

Run from the ``dashboard`` folder so ``config``/``metrology`` import cleanly::

    conda activate fs50defect
    cd dashboard
    python watch_alerts.py                       # live, 5-min cycle, 30-min window
    python watch_alerts.py --interval 300 --window 30
    python watch_alerts.py --once                # single pass (for testing)
    python watch_alerts.py --mock --once         # offline smoke test
    python watch_alerts.py --no-images           # skip image ingestion
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime, timedelta

import alerts
import config
import metrology

try:  # image ingestion is optional (needs ultralytics/torch)
    import ingestion
except Exception:  # pragma: no cover - keep the watcher alive without the model
    ingestion = None


def _run_once(window_minutes: int, do_images: bool, mock: bool) -> None:
    """One dispatch cycle: ingest images, then fire FS50 + FS100 Teams alerts."""
    now = datetime.now()
    start = now - timedelta(minutes=window_minutes)
    stamp = f"{now:%Y-%m-%d %H:%M:%S}"

    if do_images and ingestion is not None:
        try:
            summaries = ingestion.scan_and_process(conf=config.CONF_THRESHOLD)
            if summaries:
                print(f"[{stamp}] ingested {len(summaries)} new image(s)")
        except Exception as exc:  # never let image ingestion kill the loop
            print(f"[{stamp}] image ingestion skipped: {exc}")

    fired: list[dict] = []
    try:
        adf = metrology.build_alerts(start=start, end=now, mock=mock or None)
        fired += alerts.send_defect_burst_alerts(
            adf, window_minutes=15, min_count=3, cooldown_minutes=15
        )
    except Exception as exc:
        print(f"[{stamp}] defect-alert scan failed: {exc}")

    try:
        broken = metrology.broken_with_attribution(
            start=start, end=now, mock=mock or None
        )
        fired += alerts.send_broken_alerts(broken, cooldown_minutes=15)
    except Exception as exc:
        print(f"[{stamp}] broken-plate scan failed: {exc}")

    if fired:
        for f in fired:
            mark = "delivered" if f.get("delivered") else "FAILED"
            print(f"[{stamp}] Teams alert {mark}: {f['rule']}")
    else:
        print(f"[{stamp}] no alerts (window {window_minutes} min)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless Teams alert watcher.")
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between dispatch cycles (default 300 = 5 min).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=30,
        help="Rolling look-back window in minutes to scan each cycle (default 30).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single pass and exit (for testing / Task Scheduler per-tick).",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Skip corner-image ingestion (alerts-only mode).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use synthetic data (offline smoke test).",
    )
    args = parser.parse_args()

    do_images = not args.no_images
    if args.once:
        _run_once(args.window, do_images, args.mock)
        return

    print(
        f"[{datetime.now():%Y-%m-%d %H:%M:%S}] alert watcher started "
        f"(every {args.interval}s, {args.window}-min window). Ctrl+C to stop."
    )
    while True:
        try:
            _run_once(args.window, do_images, args.mock)
        except Exception as exc:  # keep the loop alive no matter what
            print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] cycle error: {exc}")
        time.sleep(max(30, args.interval))


if __name__ == "__main__":
    main()
