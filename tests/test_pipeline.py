import json

import numpy as np
import pytest

from netsentry.cli import main
from netsentry.datasets import read_nsl_kdd
from netsentry.detector import Detector
from netsentry.metrics import binary_metrics, calibrate_threshold
from netsentry.report import render_report
from netsentry.train import train


def test_calibrate_threshold_respects_target_with_ties():
    scores = np.array([0.0] * 90 + [0.2] * 5 + [0.5] * 5)
    t = calibrate_threshold(scores, 0.05)
    assert np.mean(scores >= t) <= 0.05
    assert np.mean(scores >= calibrate_threshold(scores, 0.10)) <= 0.10
    assert calibrate_threshold(scores, 0.0) > 0.5          # zero false alarms allowed


def test_binary_metrics():
    m = binary_metrics([1, 1, 1, 0, 0], [0.9, 0.8, 0.1, 0.7, 0.2], 0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (2, 1, 1, 1)
    assert m["detection_rate"] == pytest.approx(2 / 3)
    assert m["false_alarm_rate"] == pytest.approx(0.5)


def test_train_save_load_predict(nsl_kdd_dir, tmp_path):
    det, m = train("nsl-kdd", nsl_kdd_dir, "rf", target_far=0.02, log=lambda *_: None)
    assert m["binary"]["detection_rate"] > 0.9
    assert m["binary"]["false_alarm_rate"] < 0.1
    assert m["seen_vs_novel"]["novel"]["attack_types"] == 1        # mscan only appears in test
    assert {r["attack"] for r in m["per_attack"]} == {"neptune", "mscan", "guess_passwd"}
    assert m["feature_importance"] and m["feature_importance"][0][1] > 0

    path = det.save(tmp_path / "model.joblib")
    again = Detector.load(path)
    test = read_nsl_kdd(nsl_kdd_dir / "KDDTest+.txt")
    a, b = det.predict(test.X), again.predict(test.X)
    assert (a["category"] == b["category"]).all()
    assert again.threshold == det.threshold
    assert set(a.columns) == {"attack_score", "is_attack", "category", "family_score"}

    html = render_report(m)
    assert "<svg" in html and "Detection rate by attack type" in html


@pytest.mark.parametrize("model", ["et", "hgb", "logreg", "mlp"])
def test_other_models_train(nsl_kdd_dir, model):
    _, m = train("nsl-kdd", nsl_kdd_dir, model, log=lambda *_: None)
    assert m["binary"]["roc_auc"] > 0.8


def test_missing_columns_are_reported(nsl_kdd_dir):
    det, _ = train("nsl-kdd", nsl_kdd_dir, "logreg", log=lambda *_: None)
    X = read_nsl_kdd(nsl_kdd_dir / "KDDTest+.txt").X.drop(columns=["src_bytes", "flag"])
    with pytest.raises(ValueError, match="src_bytes"):
        det.predict(X)


def test_cicids2017_end_to_end(cicids_dir):
    det, m = train("cicids2017", cicids_dir, "rf", log=lambda *_: None)
    assert det.meta["categorical"] == []
    assert "Fwd Header Length.1" not in det.features
    assert m["binary"]["detection_rate"] > 0.9
    assert set(m["multiclass"]["labels"]) >= {"normal", "DDoS", "PortScan", "WebAttack"}


def test_cli_train_evaluate_detect(nsl_kdd_dir, tmp_path, capsys):
    model, report = tmp_path / "m.joblib", tmp_path / "r.html"
    assert main(["train", "--data", str(nsl_kdd_dir), "--out", str(model), "--report", str(report)]) == 0
    assert model.exists() and report.exists() and report.with_suffix(".json").exists()

    assert main(["evaluate", "--model", str(model), "--data", str(nsl_kdd_dir)]) == 0
    assert "detection rate" in capsys.readouterr().out

    preds, alerts = tmp_path / "p.csv", tmp_path / "a.jsonl"
    test_file = str(nsl_kdd_dir / "KDDTest+.txt")
    assert main(["detect", "--model", str(model), "--input", test_file, "--out", str(preds)]) == 0
    assert preds.read_text().count("\n") == 371   # header + 370 records
    assert "vs labels: detection" in capsys.readouterr().out

    assert main(["detect", "--model", str(model), "--input", test_file, "--replay", "--rate", "0",
                 "--limit", "100", "--alert-log", str(alerts), "--quiet"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["records"] == 100
    lines = alerts.read_text().splitlines()
    assert len(lines) == summary["alerts"] and all(json.loads(x)["category"] != "normal" for x in lines)
