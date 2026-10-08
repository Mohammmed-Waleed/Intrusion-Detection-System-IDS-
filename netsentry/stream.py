"""Score traffic record-by-record as if it were arriving live, printing alerts as they fire."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .datasets import NORMAL, Records
from .detector import Detector


def replay(detector: Detector, rec: Records, *, rate: float = 200.0, batch: int = 50,
           limit: int | None = None, alert_log: str | Path | None = None, quiet: bool = False,
           out=sys.stdout) -> dict:
    """Feed `rec` through the detector in small batches at ~`rate` records/second.

    Scoring is batched (as a real sensor would micro-batch flows); pacing only exists
    to make the demo watchable. Use rate=0 to go as fast as possible.
    """
    n = len(rec) if limit is None else min(limit, len(rec))
    if alert_log:
        Path(alert_log).parent.mkdir(parents=True, exist_ok=True)
    log = open(alert_log, "a", encoding="utf-8") if alert_log else None
    stats = Counter()
    families = Counter()
    t0 = time.perf_counter()
    try:
        for start in range(0, n, batch):
            stop = min(start + batch, n)
            X = rec.X.iloc[start:stop]
            pred = detector.predict(X)
            for i, row in zip(range(start, stop), pred.itertuples(index=False)):
                truth = None if rec.attack is None else rec.attack.iloc[i]
                truth_attack = None if rec.category is None else rec.category.iloc[i] != NORMAL
                stats["seen"] += 1
                if truth_attack is not None:
                    stats["tp" if row.is_attack and truth_attack else
                          "fp" if row.is_attack else
                          "fn" if truth_attack else "tn"] += 1
                if not row.is_attack:
                    continue
                stats["alerts"] += 1
                families[row.category] += 1
                event = {"ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                         "record": i, "category": row.category,
                         "attack_score": float(row.attack_score), "family_score": float(row.family_score)}
                if truth is not None:
                    event["truth"] = truth
                if log:
                    log.write(json.dumps(event) + "\n")
                if not quiet:
                    mark = "" if truth_attack is None else ("  ok" if truth_attack else "  FALSE ALARM")
                    label = "" if truth is None else f"  (label: {truth})"
                    print(f"[{i:>7}] ALERT {row.category:<12} score {row.attack_score:.3f}{label}{mark}", file=out)
            if rate > 0:
                behind = stop / rate - (time.perf_counter() - t0)
                if behind > 0:
                    time.sleep(behind)
            if not quiet and (stop // batch) % 20 == 0:
                print(_status(stats, families, time.perf_counter() - t0), file=out)
    except KeyboardInterrupt:
        print("\ninterrupted", file=out)
    finally:
        if log:
            log.close()

    summary = {"records": stats["seen"], "alerts": stats["alerts"], "by_family": dict(families),
               "seconds": round(time.perf_counter() - t0, 2)}
    if stats["tp"] + stats["fn"] + stats["fp"] + stats["tn"]:
        summary["detection_rate"] = stats["tp"] / max(stats["tp"] + stats["fn"], 1)
        summary["false_alarm_rate"] = stats["fp"] / max(stats["fp"] + stats["tn"], 1)
    return summary


def _status(stats: Counter, families: Counter, secs: float) -> str:
    fam = ", ".join(f"{k} {v}" for k, v in families.most_common())
    line = f"--- {stats['seen']:,} records, {stats['alerts']:,} alerts ({fam or 'none'}), {stats['seen'] / max(secs, 1e-9):,.0f} rec/s"
    if stats["tp"] + stats["fn"]:
        line += f", detection {stats['tp'] / (stats['tp'] + stats['fn']):.1%}"
    if stats["fp"] + stats["tn"]:
        line += f", false alarms {stats['fp'] / (stats['fp'] + stats['tn']):.1%}"
    return line + " ---"


def score_file(detector: Detector, rec: Records) -> pd.DataFrame:
    """Batch-score everything; returns predictions joined with any labels."""
    pred = detector.predict(rec.X)
    if rec.attack is not None:
        pred.insert(0, "label", rec.attack.values)
    return pred
