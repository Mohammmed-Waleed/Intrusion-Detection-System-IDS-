"""Loaders for NSL-KDD and CICIDS2017, normalised into a common `Records` shape.

Every loader returns features plus two label columns:

- ``attack``:   the dataset's fine-grained label (``neptune``, ``DoS Hulk``, ``normal``...)
- ``category``: a coarse family the model is trained on (``normal``, ``DoS``, ``Probe``...)
"""

from __future__ import annotations

import hashlib
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NORMAL = "normal"

# ---------------------------------------------------------------------------
# NSL-KDD
# ---------------------------------------------------------------------------

NSL_KDD_FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login", "count",
    "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
NSL_KDD_CATEGORICAL = ["protocol_type", "service", "flag"]

# The standard attack -> family mapping. KDDTest+ contains 17 attack types that never
# appear in KDDTrain+ (apache2, mscan, snmpguess...), which is what makes it a useful
# test of generalisation to unseen attacks.
NSL_KDD_FAMILIES = {
    "DoS": ["apache2", "back", "land", "mailbomb", "neptune", "pod", "processtable",
            "smurf", "teardrop", "udpstorm", "worm"],
    "Probe": ["ipsweep", "mscan", "nmap", "portsweep", "saint", "satan"],
    "R2L": ["ftp_write", "guess_passwd", "httptunnel", "imap", "multihop", "named", "phf",
            "sendmail", "snmpgetattack", "snmpguess", "spy", "warezclient", "warezmaster",
            "xlock", "xsnoop"],
    "U2R": ["buffer_overflow", "loadmodule", "perl", "ps", "rootkit", "sqlattack", "xterm"],
}
NSL_KDD_ATTACK_FAMILY = {a: fam for fam, attacks in NSL_KDD_FAMILIES.items() for a in attacks}

# Two independent GitHub mirrors of the UNB release; files are checked against these hashes.
NSL_KDD_MIRRORS = [
    "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/",
    "https://raw.githubusercontent.com/jmnwong/NSL-KDD-Dataset/master/",
]
NSL_KDD_FILES = {
    "KDDTrain+.txt": "1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95",
    "KDDTest+.txt": "fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84",
}

# ---------------------------------------------------------------------------
# CICIDS2017
# ---------------------------------------------------------------------------

# Present in the "GeneratedLabelledFlows" variant. They identify a flow rather than
# describe it, and let a model memorise hosts/times instead of learning behaviour.
CICIDS_ID_COLUMNS = {"flow id", "source ip", "src ip", "source port", "src port",
                     "destination ip", "dst ip", "timestamp"}


@dataclass
class Records:
    """Feature rows plus optional labels (labels are None for unlabelled traffic)."""

    X: pd.DataFrame
    attack: pd.Series | None = None
    category: pd.Series | None = None

    def __len__(self) -> int:
        return len(self.X)

    @property
    def is_attack(self) -> np.ndarray:
        return (self.category != NORMAL).to_numpy()

    def subset(self, idx) -> "Records":
        take = lambda s: None if s is None else s.iloc[idx].reset_index(drop=True)  # noqa: E731
        return Records(self.X.iloc[idx].reset_index(drop=True), take(self.attack), take(self.category))


def nsl_kdd_family(label: str) -> str:
    label = label.strip().rstrip(".").lower()
    if label == NORMAL:
        return NORMAL
    return NSL_KDD_ATTACK_FAMILY.get(label, "Unknown")


def read_nsl_kdd(path: str | Path) -> Records:
    """Read an NSL-KDD / KDD'99 file: 41 features, optionally + label (+ difficulty).

    Accepts the original headerless ``.txt`` files as well as a CSV with a header row.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        first = f.readline()
    has_header = first.split(",")[0].strip().lower() == "duration"
    df = pd.read_csv(path, header=0 if has_header else None)
    if not has_header:
        n = df.shape[1]
        if n not in (41, 42, 43):
            raise ValueError(f"{path}: expected 41-43 columns for NSL-KDD, got {n}")
        df.columns = NSL_KDD_FEATURES + ["label", "difficulty"][: n - 41]

    missing = [c for c in NSL_KDD_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing NSL-KDD columns {missing}")
    X = df[NSL_KDD_FEATURES].copy()
    for c in NSL_KDD_CATEGORICAL:
        X[c] = X[c].astype(str)

    if "label" not in df.columns:
        return Records(X)
    attack = df["label"].astype(str).str.strip().str.rstrip(".").str.lower()
    return Records(X, attack.rename("attack"), attack.map(nsl_kdd_family).rename("category"))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_nsl_kdd(out_dir: str | Path = "data/nsl-kdd", force: bool = False) -> list[Path]:
    """Fetch KDDTrain+ / KDDTest+ (about 22 MB together), verifying SHA-256."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    done = []
    for name, digest in NSL_KDD_FILES.items():
        dest = out / name
        if dest.exists() and not force and (digest is None or _sha256(dest) == digest):
            done.append(dest)
            continue
        errors = []
        for mirror in NSL_KDD_MIRRORS:
            url = mirror + name.replace("+", "%2B")
            tmp = dest.with_suffix(dest.suffix + ".part")
            try:
                with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
                if digest is not None and _sha256(tmp) != digest:
                    raise ValueError("SHA-256 mismatch")
                tmp.replace(dest)
                break
            except Exception as e:  # try the next mirror
                tmp.unlink(missing_ok=True)
                errors.append(f"{url}: {e}")
        else:
            raise RuntimeError(f"could not download {name}:\n  " + "\n  ".join(errors))
        done.append(dest)
    return done


def load_nsl_kdd(data_dir: str | Path = "data/nsl-kdd") -> tuple[Records, Records]:
    """The official split: KDDTrain+ for training, KDDTest+ for testing."""
    d = Path(data_dir)
    train, test = d / "KDDTrain+.txt", d / "KDDTest+.txt"
    for p in (train, test):
        if not p.exists():
            raise FileNotFoundError(f"{p} not found - run `netsentry download nsl-kdd --out {d}` first")
    return read_nsl_kdd(train), read_nsl_kdd(test)


# ---------------------------------------------------------------------------
# CICIDS2017
# ---------------------------------------------------------------------------

def cicids_family(label: str) -> str:
    l = label.lower()
    if l == "benign":
        return NORMAL
    if l.startswith("ddos"):
        return "DDoS"
    if l.startswith("dos"):
        return "DoS"
    if "portscan" in l:
        return "PortScan"
    if "patator" in l:
        return "BruteForce"
    if l.startswith("web attack"):
        return "WebAttack"
    if l.startswith("bot"):
        return "Botnet"
    if "infiltration" in l:
        return "Infiltration"
    if "heartbleed" in l:
        return "Heartbleed"
    return "Unknown"


def _clean_label(s: pd.Series) -> pd.Series:
    # "Web Attack \x96 Brute Force" in the raw files: the dash is a cp1252 byte.
    return (s.astype(str)
             .map(lambda v: re.sub(r"[^\x20-\x7e]+", "-", v))
             .str.replace(r"\s+", " ", regex=True)
             .str.strip())


def clean_cicids(df: pd.DataFrame, require_label: bool = True) -> Records:
    """Normalise a raw CICIDS2017 frame (MachineLearningCVE or GeneratedLabelledFlows).

    - strips the stray whitespace in column names (" Flow Duration")
    - drops identifier columns and the duplicated "Fwd Header Length.1"
    - coerces features to numbers; rows containing NaN / +-inf (Flow Bytes/s) are dropped
    - drops blank padding rows (the Thursday web-attack file has ~288k of them)
    """
    df = df.copy()
    df.columns = [re.sub(r"\s+", " ", str(c)).strip() for c in df.columns]
    df = df.loc[:, ~pd.Index(df.columns).duplicated()]
    drop = [c for c in df.columns
            if c.lower() in CICIDS_ID_COLUMNS
            or (re.search(r"\.\d+$", c) and re.sub(r"\.\d+$", "", c) in df.columns)]
    df = df.drop(columns=drop)

    label_col = next((c for c in df.columns if c.lower() == "label"), None)
    if label_col is None and require_label:
        raise ValueError("no 'Label' column found in CICIDS2017 data")

    feats = [c for c in df.columns if c != label_col]
    X = df[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    keep = X.notna().all(axis=1)
    if label_col is not None:
        keep &= df[label_col].notna()
    X = X[keep].reset_index(drop=True).astype("float64")
    if label_col is None:
        return Records(X)
    attack = _clean_label(df.loc[keep, label_col]).reset_index(drop=True).rename("attack")
    return Records(X, attack, attack.map(cicids_family).rename("category"))


def read_cicids(path: str | Path, require_label: bool = True) -> Records:
    """Read one CSV or every CSV in a directory (e.g. MachineLearningCVE/)."""
    path = Path(path)
    files = sorted(path.glob("*.csv")) if path.is_dir() else [path]
    if not files:
        raise FileNotFoundError(f"no CSV files found in {path}")
    frames = []
    for f in files:
        frame = pd.read_csv(f, encoding="latin-1", low_memory=False)
        frame.columns = [str(c).strip() for c in frame.columns]
        frames.append(frame)
    return clean_cicids(pd.concat(frames, ignore_index=True), require_label=require_label)


def _downsample(rec: Records, max_rows: int, seed: int, min_per_class: int = 500) -> Records:
    """Shrink to ~max_rows, proportionally per class but never starving rare classes."""
    if len(rec) <= max_rows:
        return rec
    rng = np.random.default_rng(seed)
    frac = max_rows / len(rec)
    idx = []
    for _, rows in rec.attack.groupby(rec.attack).groups.items():
        rows = np.asarray(rows)
        n = min(len(rows), max(round(len(rows) * frac), min_per_class))
        idx.append(rng.choice(rows, size=n, replace=False))
    return rec.subset(np.sort(np.concatenate(idx)))


def stratified_split(rec: Records, test_size: float, seed: int) -> tuple[Records, Records]:
    """Split stratified on the fine-grained label; singleton labels stay in the first part."""
    rng = np.random.default_rng(seed)
    a, b = [], []
    for _, rows in rec.attack.groupby(rec.attack).groups.items():
        rows = rng.permutation(np.asarray(rows))
        n_test = int(round(len(rows) * test_size)) if len(rows) > 1 else 0
        b.append(rows[:n_test])
        a.append(rows[n_test:])
    return rec.subset(np.sort(np.concatenate(a))), rec.subset(np.sort(np.concatenate(b)))


def load_cicids2017(path: str | Path, test_size: float = 0.2, seed: int = 0,
                    max_rows: int | None = None) -> tuple[Records, Records]:
    """CICIDS2017 ships no official split, so we make a stratified one.

    Exact duplicate rows are removed first; CICIDS2017 has hundreds of thousands of
    them, and leaving them in leaks test rows into training and inflates every metric.
    """
    rec = read_cicids(path)
    frame = pd.concat([rec.X, rec.attack], axis=1)
    dup = frame.duplicated().to_numpy()
    rec = rec.subset(np.flatnonzero(~dup))
    if max_rows:
        rec = _downsample(rec, max_rows, seed)
    return stratified_split(rec, test_size, seed)


# ---------------------------------------------------------------------------

DATASETS = {
    "nsl-kdd": {"categorical": NSL_KDD_CATEGORICAL, "default_path": "data/nsl-kdd"},
    "cicids2017": {"categorical": [], "default_path": "data/cicids2017"},
}


def load_dataset(name: str, path: str | Path | None = None, *, seed: int = 0,
                 test_size: float = 0.2, max_rows: int | None = None) -> tuple[Records, Records]:
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {sorted(DATASETS)}")
    path = path or DATASETS[name]["default_path"]
    if name == "nsl-kdd":
        train, test = load_nsl_kdd(path)
        if max_rows:
            train = _downsample(train, max_rows, seed)
        return train, test
    return load_cicids2017(path, test_size=test_size, seed=seed, max_rows=max_rows)


def read_records(name: str, path: str | Path) -> Records:
    """Read traffic to score with a model trained on dataset `name` (labels optional)."""
    if name == "nsl-kdd":
        return read_nsl_kdd(path)
    if name == "cicids2017":
        return read_cicids(path, require_label=False)
    raise ValueError(f"unknown dataset {name!r}")
