from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .datasets import DATASETS
from .models import MODELS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="netsentry", description="ML intrusion detection on network flow records.")
    p.add_argument("--version", action="version", version=f"netsentry {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("download", help="download a public dataset (NSL-KDD)")
    d.add_argument("dataset", choices=["nsl-kdd"])
    d.add_argument("--out", default="data/nsl-kdd")
    d.add_argument("--force", action="store_true", help="re-download even if files exist")

    t = sub.add_parser("train", help="train a detector, calibrate its threshold and evaluate it")
    t.add_argument("--dataset", choices=sorted(DATASETS), default="nsl-kdd")
    t.add_argument("--data", default=None, help="dataset folder (or CSV for cicids2017)")
    t.add_argument("--model", choices=sorted(MODELS), default="rf",
                   help="; ".join(f"{k}: {v}" for k, v in MODELS.items()))
    t.add_argument("--no-balance", action="store_true", help="disable class reweighting")
    t.add_argument("--target-far", type=float, default=0.01,
                   help="false-alarm rate to calibrate the alert threshold for, on validation data (default 0.01)")
    t.add_argument("--threshold", type=float, default=None, help="fixed alert threshold instead of calibrating")
    t.add_argument("--max-rows", type=int, default=None, help="subsample training data (rare classes are kept)")
    t.add_argument("--test-size", type=float, default=0.2, help="test share for datasets without an official split")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default=None, help="model bundle path (default models/<dataset>-<model>.joblib)")
    t.add_argument("--report", default=None, help="HTML report path (default reports/<dataset>-<model>.html)")

    e = sub.add_parser("evaluate", help="re-evaluate a saved model on its dataset's test split")
    e.add_argument("--model", required=True, help="model bundle (.joblib)")
    e.add_argument("--data", default=None, help="dataset location (default: the one used for training)")
    e.add_argument("--threshold", type=float, default=None, help="override the stored threshold")
    e.add_argument("--report", default=None, help="also write an HTML report here")

    x = sub.add_parser("detect", help="score a file of flow records; --replay streams it like live traffic")
    x.add_argument("--model", required=True, help="model bundle (.joblib)")
    x.add_argument("--input", required=True, help="records in the training dataset's format (labels optional)")
    x.add_argument("--out", default=None, help="write per-record predictions to this CSV")
    x.add_argument("--threshold", type=float, default=None, help="override the stored threshold")
    x.add_argument("--replay", action="store_true", help="stream records and print alerts as they fire")
    x.add_argument("--rate", type=float, default=200, help="replay speed in records/s (0 = unthrottled)")
    x.add_argument("--limit", type=int, default=None, help="only process the first N records")
    x.add_argument("--shuffle", action="store_true", help="shuffle records before replay")
    x.add_argument("--alert-log", default=None, help="append alerts to this JSONL file")
    x.add_argument("--quiet", action="store_true", help="replay without printing each alert")
    return p


def _print_summary(m: dict) -> None:
    b, mc = m["binary"], m["multiclass"]
    print(f"\ntest records        {m['n']:,} ({m['n_attack']:,} attacks)")
    print(f"detection rate      {b['detection_rate']:.2%}")
    print(f"false-alarm rate    {b['false_alarm_rate']:.2%}")
    print(f"precision / F1      {b['precision']:.2%} / {b['f1']:.4f}")
    if b["roc_auc"] is not None:
        print(f"ROC-AUC             {b['roc_auc']:.4f}")
    print(f"family accuracy     {mc['accuracy']:.2%} (macro-F1 {mc['macro_f1']:.4f})")
    snv = m.get("seen_vs_novel")
    if snv and snv["novel"]["n"]:
        print(f"seen attack types   {snv['seen']['detection_rate']:.2%} detected ({snv['seen']['n']:,} records)")
        print(f"novel attack types  {snv['novel']['detection_rate']:.2%} detected "
              f"({snv['novel']['attack_types']} types, {snv['novel']['n']:,} records)")
    b5 = m.get("binary_at_0.5")
    if b5:
        print(f"(at threshold 0.5:  detection {b5['detection_rate']:.2%}, false alarms {b5['false_alarm_rate']:.2%})")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "download":
        from .datasets import download_nsl_kdd

        for path in download_nsl_kdd(args.out, force=args.force):
            print(f"ok  {path}")

    elif args.command == "train":
        from .report import write_metrics, write_report
        from .train import train

        det, metrics = train(args.dataset, args.data, args.model, target_far=args.target_far,
                             threshold=args.threshold, max_rows=args.max_rows, test_size=args.test_size,
                             balanced=not args.no_balance, seed=args.seed)
        stem = f"{args.dataset}-{args.model}"
        out = det.save(args.out or f"models/{stem}.joblib")
        report = write_report(metrics, args.report or f"reports/{stem}.html")
        mpath = write_metrics(metrics, report.with_suffix(".json"))
        _print_summary(metrics)
        print(f"\nmodel:   {out}\nreport:  {report}\nmetrics: {mpath}")

    elif args.command == "evaluate":
        from .datasets import load_dataset
        from .detector import Detector
        from .metrics import evaluate
        from .report import write_report

        det = Detector.load(args.model)
        meta = det.meta
        print(det.describe())
        _, test = load_dataset(meta["dataset"], args.data or meta["data_path"], seed=meta["seed"],
                               test_size=meta["test_size"], max_rows=meta.get("max_rows"))
        m = evaluate(det, test, seen_attacks=set(meta.get("train_attacks", [])) or None, threshold=args.threshold)
        _print_summary(m)
        if args.report:
            from .detector import feature_importance

            m["meta"] = meta
            m["feature_importance"] = feature_importance(det, top=20)
            print(f"report: {write_report(m, args.report)}")

    elif args.command == "detect":
        from .datasets import read_records
        from .detector import Detector

        det = Detector.load(args.model)
        if args.threshold is not None:
            det.threshold = args.threshold
        print(det.describe(), file=sys.stderr)
        rec = read_records(det.meta["dataset"], args.input)
        if args.shuffle:
            import numpy as np
            rec = rec.subset(np.random.default_rng(0).permutation(len(rec)))
        if args.limit:
            rec = rec.subset(range(min(args.limit, len(rec))))

        if args.replay:
            from .stream import replay

            summary = replay(det, rec, rate=args.rate, alert_log=args.alert_log, quiet=args.quiet)
            print(json.dumps(summary, indent=2))
        else:
            from .stream import score_file

            pred = score_file(det, rec)
            for p in (args.out, args.alert_log):
                if p:
                    Path(p).parent.mkdir(parents=True, exist_ok=True)
            if args.out:
                pred.to_csv(args.out, index_label="record")
                print(f"predictions: {args.out}", file=sys.stderr)
            if args.alert_log:
                alerts = pred[pred["is_attack"]]
                alerts.reset_index(names="record").to_json(args.alert_log, orient="records", lines=True)
            n_alert = int(pred["is_attack"].sum())
            print(f"{len(pred):,} records, {n_alert:,} alerts ({n_alert / max(len(pred), 1):.1%})")
            print(pred.loc[pred["is_attack"], "category"].value_counts().to_string())
            if rec.category is not None:
                truth = rec.is_attack
                hit = pred["is_attack"].to_numpy()
                print(f"vs labels: detection {hit[truth].mean() if truth.any() else 0:.2%}, "
                      f"false alarms {hit[~truth].mean() if (~truth).any() else 0:.2%}")
    return 0
