# NetSentry

A machine-learning intrusion detection system (IDS) that classifies network connections as normal or malicious, and names the attack family. Train it on **NSL-KDD** or **CICIDS2017**.

NetSentry is built on scikit-learn and includes everything around the model that an IDS needs:

- **Dataset loaders** that handle each dataset's quirks: NSL-KDD's headerless files and test-only attack types, and CICIDS2017's padded column names, `inf` values, duplicated columns, identifier columns, cp1252 labels and duplicate rows
- **One-command download** of NSL-KDD, with SHA-256 checks against two mirrors
- **Multiclass detection**: `normal` vs `DoS` / `Probe` / `R2L` / `U2R` (NSL-KDD), or `DDoS` / `PortScan` / `BruteForce` / `WebAttack` / `Botnet` / ... (CICIDS2017)
- **Alert-threshold calibration**: the alert cut-off is tuned to a target false-alarm rate on held-out validation data, not left at an arbitrary 0.5
- **IDS-focused evaluation**: detection rate, false-alarm rate, ROC-AUC, per-family confusion, per-attack-type detection, and detection on **attack types the model never saw in training**
- **A self-contained HTML report** (ROC curve, confusion matrix, per-attack bars, feature importance; light and dark mode)
- **Detection on new traffic**: batch-score a file, or replay it as a live stream that prints alerts and writes a JSONL alert log
- Five model types: random forest (default), extra trees, gradient boosting, logistic regression, MLP

## Results on NSL-KDD

Trained on `KDDTrain+` and tested on the official `KDDTest+`, which contains **17 attack types that never appear in training**. Default random forest, CPU only, fit in about 4 s. The full evaluation report is in [`reports/nsl-kdd-rf.html`](reports/nsl-kdd-rf.html) (download it and open it in a browser; GitHub shows only the HTML source).

| metric | value |
| --- | --- |
| Detection rate (attacks flagged) | **89.3%** |
| False-alarm rate (normal traffic flagged) | **10.1%** |
| ROC-AUC | 0.959 |
| Detection on the 17 **unseen** attack types | **87.6%** (3,750 records) |
| Attack-family accuracy (5 classes) | 82.7% (macro-F1 0.708) |
| Scoring throughput | ~100k records/s (batch) |

**Why the threshold matters.** At the textbook 0.5 cut-off, the same model catches only **58.3%** of attacks (2.6% false alarms; 75% accuracy, in line with what most NSL-KDD papers report for KDDTest+). Calibrating the threshold for 1% false alarms on validation data lowers it to 0.065, which raises detection to 89.3% and accuracy to 89.6%.

**The honest caveat.** That threshold produces **0.9%** false alarms on validation data drawn from the training distribution, but **10.1%** on KDDTest+. The test set was deliberately built to differ from the training set, and false-alarm rates estimated in-distribution are optimistic. Most of the extra alarms are normal traffic misread as `Probe`. The report shows this gap explicitly. On a real network, recalibrate the threshold on your own benign traffic (`--threshold`).

### Model comparison (same split, `--target-far 0.01`)

| model | detection | false alarms | ROC-AUC | family macro-F1 | unseen attacks detected |
| --- | --- | --- | --- | --- | --- |
| `rf` random forest, balanced (**default**) | 89.3% | 10.1% | 0.959 | **0.708** | 87.6% |
| `rf` random forest, `--no-balance` | 88.7% | 9.9% | **0.961** | 0.666 | 86.5% |
| `et` extra trees, `--no-balance` | 89.4% | 10.0% | 0.954 | 0.700 | **92.1%** |
| `hgb` gradient boosting, balanced | 80.0% | 8.7% | 0.913 | 0.633 | 63.9% |
| `logreg` logistic regression, balanced | 60.7% | 7.3% | 0.922 | 0.530 | 26.4% |
| `mlp` neural net 128-64 (no class weights) | 76.5% | **3.6%** | 0.903 | 0.640 | 54.6% |

Tree ensembles generalise much better than boosting, linear or neural models to the attack types missing from training. The MLP has the fewest false alarms but misses almost half the unseen attacks. Class balancing mostly helps the rare `R2L` and `U2R` families (macro-F1 0.67 to 0.71).

## Install

```bash
git clone https://github.com/Mohammmed-Waleed/Intrusion-Detection-System-IDS-.git
cd Intrusion-Detection-System-IDS-
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Quick start (NSL-KDD)

```bash
# 1. download KDDTrain+ / KDDTest+ (~22 MB, hash-verified)
netsentry download nsl-kdd

# 2. train + calibrate + evaluate; writes models/nsl-kdd-rf.joblib and reports/nsl-kdd-rf.html
netsentry train --dataset nsl-kdd --model rf
#    ...or skip training and grab the pretrained model from the v0.1.0 release
curl -L --create-dirs -o models/nsl-kdd-rf.joblib \
  https://github.com/Mohammmed-Waleed/Intrusion-Detection-System-IDS-/releases/download/v0.1.0/nsl-kdd-rf.joblib

# 3. watch it work as a live sensor: alerts stream to the terminal and to a JSONL log
netsentry detect --model models/nsl-kdd-rf.joblib --input data/nsl-kdd/KDDTest+.txt \
    --replay --shuffle --rate 200 --alert-log runs/alerts.jsonl

# 4. or batch-score a file into a CSV of predictions
netsentry detect --model models/nsl-kdd-rf.joblib --input data/nsl-kdd/KDDTest+.txt --out runs/preds.csv
```

Replay output, excerpt from an unthrottled run (`--rate 0 --limit 1000`). The `ok` / `FALSE ALARM` markers appear only when the input has labels:

```
[      3] ALERT DoS          score 1.000  (label: neptune)  ok
[      5] ALERT Probe        score 0.225  (label: httptunnel)  ok
[      9] ALERT Probe        score 1.000  (label: satan)  ok
--- 1,000 records, 537 alerts (DoS 294, Probe 172, R2L 65, U2R 6), 1,512 rec/s, detection 89.1%, false alarms 10.6% ---
```

Each alert in the log is one JSON line:

```json
{"ts": "2026-10-08T18:44:22.760+00:00", "record": 3, "category": "DoS", "attack_score": 1.0, "family_score": 0.995, "truth": "neptune"}
```

Other commands:

```bash
netsentry train --model et --target-far 0.005        # stricter alerting
netsentry train --threshold 0.5                      # fixed threshold, no calibration
netsentry evaluate --model models/nsl-kdd-rf.joblib --report reports/eval.html
netsentry --help
```

## Training on CICIDS2017

CICIDS2017 is about 2.8M labelled flows (roughly 1 GB of CSV), so it isn't downloaded automatically.

1. Request it from the [Canadian Institute for Cybersecurity](https://www.unb.ca/cic/datasets/ids-2017.html) and download `MachineLearningCSV.zip` (the `TrafficLabelling` CSVs also work; NetSentry drops their IP, port, flow-id and timestamp columns).
2. Unzip the 8 daily CSVs into `data/cicids2017/`.
3. Train:

```bash
# --max-rows subsamples for speed; rare classes (Heartbleed, Infiltration, SQLi) are kept whole
netsentry train --dataset cicids2017 --data data/cicids2017 --max-rows 500000
```

CICIDS2017 has no official split. NetSentry removes exact duplicate rows first (otherwise duplicated flows leak from train into test and inflate every metric), then makes a stratified 80/20 split (`--test-size`, `--seed`). Labels are grouped into families: `DoS`, `DDoS`, `PortScan`, `BruteForce` (FTP/SSH-Patator), `WebAttack`, `Botnet`, `Infiltration`, `Heartbleed`.

To score your own traffic with a CICIDS2017 model, extract flows with [CICFlowMeter](https://github.com/ahlashkari/CICFlowMeter) and pass the CSV to `netsentry detect`. The column names must match the training schema; NetSentry lists any missing columns rather than guessing.

## How it works

```
records ──► signed log1p + standardise (numeric)  ──►  classifier  ──►  P(normal), P(DoS), P(Probe)...
            one-hot (protocol, service, flag)                                   │
                                                     attack_score = 1 - P(normal)
                                                     alert if attack_score >= threshold
                                                     family = most likely attack class
```

- **One multiclass model** produces both outputs. The binary alert comes from `1 - P(normal)`, so the alert threshold can be tuned without retraining.
- **Threshold calibration**: 10% of the training data is held out. The threshold is the lowest value that keeps false alarms on that hold-out at or below `--target-far`. The test set never influences any choice.
- **Class balancing** (on by default; `--no-balance` turns it off) reweights classes so the 52 `U2R` training rows aren't drowned out by 67k normal ones.

## Python API

```python
from netsentry import Detector
from netsentry.datasets import read_nsl_kdd

det = Detector.load("models/nsl-kdd-rf.joblib")
records = read_nsl_kdd("data/nsl-kdd/KDDTest+.txt")   # labels optional
pred = det.predict(records.X)                         # attack_score, is_attack, category, family_score
alerts = pred[pred.is_attack]

det.predict(records.X, threshold=0.3)                 # try another operating point
```

## Project layout

```
netsentry/
  datasets.py   NSL-KDD / CICIDS2017 loading, cleaning, download, splits
  features.py   preprocessing (signed log, scaling, one-hot)
  models.py     model zoo (rf, et, hgb, logreg, mlp)
  train.py      train -> calibrate threshold -> evaluate
  metrics.py    detection / false-alarm rates, per-attack and unseen-attack analysis
  detector.py   Detector bundle: predict, save/load, feature importance
  stream.py     live replay + alert log, batch scoring
  report.py     self-contained HTML report
  cli.py        `netsentry` command
tests/          pytest suite on synthetic NSL-KDD / CICIDS2017 lookalikes (no download needed)
```

## Tests

```bash
pytest
```

## Limitations

- **NSL-KDD is old.** It derives from 1998 DARPA traffic. It is a good benchmark, and its unseen-attack test is a useful generalisation check, but a model trained on it should not guard a modern network. CICIDS2017 is more realistic but is still lab traffic.
- **No packet capture.** NetSentry classifies connection or flow records, not raw packets. For live use, put a flow exporter in front of it (CICFlowMeter for CICIDS2017-style features).
- **Supervised models detect known behaviour best.** Detection on unseen NSL-KDD attacks is good overall (87.6%), but attacks that look like normal sessions still slip through. `snmpgetattack` is detected 0% of the time (its records are essentially indistinguishable from normal SNMP traffic), `mailbomb` 34%, and even the *seen* `guess_passwd` only 43%, because its KDDTest+ records differ from the training ones. The per-attack chart in the report shows all of these.
- **False-alarm rates shift with the network.** Recalibrate the threshold on benign traffic from where the model will run.
- **Only load model bundles you trust.** They are joblib/pickle files and can execute code when loaded. The [v0.1.0 release](https://github.com/Mohammmed-Waleed/Intrusion-Detection-System-IDS-/releases/tag/v0.1.0) publishes a SHA-256 for its model. Pickled models are also tied to the scikit-learn version they were built with (1.7.1); retrain if loading fails.

## License

MIT
