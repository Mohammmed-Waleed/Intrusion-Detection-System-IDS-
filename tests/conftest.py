"""Small synthetic stand-ins for NSL-KDD and CICIDS2017, written in the real file formats."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from netsentry.datasets import NSL_KDD_FEATURES

CICIDS_COLUMNS = [" Destination Port", " Flow Duration", " Total Fwd Packets", " Total Backward Packets",
                  "Total Length of Fwd Packets", "Flow Bytes/s", " Flow Packets/s", " SYN Flag Count",
                  " Fwd Header Length", " Fwd Header Length.1", "Init_Win_bytes_forward"]


def _kdd_rows(kind: str, n: int, rng) -> pd.DataFrame:
    row = {c: np.zeros(n) for c in NSL_KDD_FEATURES}
    row.update(protocol_type=["tcp"] * n, service=["http"] * n, flag=["SF"] * n)
    if kind == "normal":
        row.update(src_bytes=rng.integers(100, 2000, n), dst_bytes=rng.integers(500, 20000, n),
                   logged_in=np.ones(n), count=rng.integers(1, 10, n), same_srv_rate=np.ones(n),
                   dst_host_srv_count=rng.integers(50, 255, n))
    elif kind == "neptune":      # SYN flood: many half-open connections
        row.update(service=["private"] * n, flag=["S0"] * n, count=rng.integers(100, 500, n),
                   serror_rate=np.ones(n), srv_serror_rate=np.ones(n), dst_host_serror_rate=np.ones(n))
    elif kind in ("satan", "mscan"):  # scans: many services, rejected connections
        row.update(service=["other"] * n, flag=["REJ"] * n, rerror_rate=np.ones(n),
                   diff_srv_rate=rng.uniform(0.5, 1, n), dst_host_diff_srv_rate=rng.uniform(0.5, 1, n),
                   count=rng.integers(20, 80, n))
    elif kind == "guess_passwd":
        row.update(service=["telnet"] * n, num_failed_logins=np.ones(n), src_bytes=rng.integers(100, 140, n),
                   duration=rng.integers(1, 5, n))
    df = pd.DataFrame(row)[NSL_KDD_FEATURES]
    df["label"] = kind
    df["difficulty"] = 20
    return df


@pytest.fixture
def nsl_kdd_dir(tmp_path):
    """KDDTrain+/KDDTest+ lookalikes; the test file contains a novel attack (mscan)."""
    rng = np.random.default_rng(0)
    train = pd.concat([_kdd_rows("normal", 600, rng), _kdd_rows("neptune", 300, rng),
                       _kdd_rows("satan", 150, rng), _kdd_rows("guess_passwd", 60, rng)])
    test = pd.concat([_kdd_rows("normal", 200, rng), _kdd_rows("neptune", 100, rng),
                      _kdd_rows("mscan", 50, rng), _kdd_rows("guess_passwd", 20, rng)])
    d = tmp_path / "nsl-kdd"
    d.mkdir()
    train.sample(frac=1, random_state=0).to_csv(d / "KDDTrain+.txt", header=False, index=False)
    test.sample(frac=1, random_state=1).to_csv(d / "KDDTest+.txt", header=False, index=False)
    return d


def cicids_frame(n_benign: int = 400, n_ddos: int = 150, n_scan: int = 100, n_web: int = 30, seed: int = 0):
    rng = np.random.default_rng(seed)

    def block(n, label, port, dur, fwd, syn):
        hdr = rng.integers(20, 400, n)
        return pd.DataFrame({
            " Destination Port": np.full(n, port), " Flow Duration": rng.integers(*dur, n),
            " Total Fwd Packets": rng.integers(*fwd, n), " Total Backward Packets": rng.integers(0, 20, n),
            "Total Length of Fwd Packets": rng.integers(0, 5000, n),
            "Flow Bytes/s": rng.uniform(0, 1e5, n), " Flow Packets/s": rng.uniform(0, 1e3, n),
            " SYN Flag Count": np.full(n, syn), " Fwd Header Length": hdr, " Fwd Header Length.1": hdr,
            "Init_Win_bytes_forward": rng.integers(-1, 65535, n), " Label": [label] * n})

    return pd.concat([block(n_benign, "BENIGN", 443, (1000, 10**7), (2, 50), 0),
                      block(n_ddos, "DDoS", 80, (1, 1000), (1, 3), 1),
                      block(n_scan, "PortScan", 0, (1, 100), (1, 2), 1),
                      block(n_web, "Web Attack \x96 Brute Force", 80, (10**5, 10**6), (3, 10), 0)],
                     ignore_index=True)


@pytest.fixture
def cicids_dir(tmp_path):
    d = tmp_path / "cicids2017"
    d.mkdir()
    df = cicids_frame()
    df.loc[0, "Flow Bytes/s"] = np.inf    # the real files contain these
    df.loc[1, " Flow Packets/s"] = np.nan
    half = len(df) // 2
    df.iloc[:half].to_csv(d / "Monday.pcap_ISCX.csv", index=False, encoding="latin-1")
    df.iloc[half:].to_csv(d / "Friday.pcap_ISCX.csv", index=False, encoding="latin-1")
    return d
