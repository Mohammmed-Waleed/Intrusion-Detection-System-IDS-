import numpy as np
import pandas as pd
import pytest

from netsentry.datasets import (NSL_KDD_FEATURES, clean_cicids, cicids_family, load_cicids2017, load_nsl_kdd,
                                nsl_kdd_family, read_nsl_kdd, stratified_split)

from .conftest import cicids_frame


def test_read_nsl_kdd_official_format(nsl_kdd_dir):
    train, test = load_nsl_kdd(nsl_kdd_dir)
    assert list(train.X.columns) == NSL_KDD_FEATURES
    assert set(train.category) == {"normal", "DoS", "Probe", "R2L"}
    assert set(test.attack) == {"normal", "neptune", "mscan", "guess_passwd"}
    # object dtype on pandas 2, the dedicated string dtype on pandas 3
    assert pd.api.types.is_string_dtype(train.X["service"])


def test_read_nsl_kdd_header_and_unlabelled(tmp_path, nsl_kdd_dir):
    raw = pd.read_csv(nsl_kdd_dir / "KDDTest+.txt", header=None)
    with_header = tmp_path / "flows.csv"
    raw.iloc[:, :41].set_axis(NSL_KDD_FEATURES, axis=1).to_csv(with_header, index=False)
    rec = read_nsl_kdd(with_header)
    assert rec.attack is None and len(rec) == len(raw)

    no_header = tmp_path / "flows.txt"
    raw.iloc[:, :41].to_csv(no_header, header=False, index=False)
    assert read_nsl_kdd(no_header).category is None


def test_nsl_kdd_family_mapping():
    assert nsl_kdd_family("neptune") == "DoS"
    assert nsl_kdd_family("Smurf.") == "DoS"       # KDD'99 style trailing dot
    assert nsl_kdd_family("mscan") == "Probe"      # test-only attack
    assert nsl_kdd_family("snmpguess") == "R2L"
    assert nsl_kdd_family("buffer_overflow") == "U2R"
    assert nsl_kdd_family("normal") == "normal"
    assert nsl_kdd_family("brand_new_attack") == "Unknown"


def test_missing_dataset_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="netsentry download"):
        load_nsl_kdd(tmp_path)


def test_clean_cicids_handles_raw_quirks():
    df = cicids_frame(n_benign=5, n_ddos=3, n_scan=2, n_web=2)
    df[" Source IP"] = "192.168.10.5"
    df.loc[0, "Flow Bytes/s"] = np.inf
    df[" Flow Packets/s"] = df[" Flow Packets/s"].astype(object)
    df.loc[1, " Flow Packets/s"] = "NaN"                     # text in a numeric column
    df.loc[len(df)] = [np.nan] * df.shape[1]   # blank padding row
    rec = clean_cicids(df)

    assert "Flow Duration" in rec.X.columns                  # whitespace stripped
    assert "Fwd Header Length.1" not in rec.X.columns        # duplicate dropped
    assert "Source IP" not in rec.X.columns                  # identifier dropped
    assert len(rec) == 12 - 2                                # inf / NaN / blank rows dropped
    assert np.isfinite(rec.X.to_numpy()).all()
    assert "Web Attack - Brute Force" in set(rec.attack)
    assert set(rec.category) == {"normal", "DDoS", "PortScan", "WebAttack"}


def test_cicids_family():
    assert cicids_family("BENIGN") == "normal"
    assert cicids_family("DoS Hulk") == "DoS"
    assert cicids_family("DDoS") == "DDoS"
    assert cicids_family("SSH-Patator") == "BruteForce"
    assert cicids_family("Web Attack - XSS") == "WebAttack"
    assert cicids_family("Bot") == "Botnet"


def test_load_cicids2017_dedups_and_stratifies(cicids_dir):
    df = pd.concat([pd.read_csv(f, encoding="latin-1") for f in sorted(cicids_dir.glob("*.csv"))])
    df.iloc[:20].to_csv(cicids_dir / "dupes.csv", index=False)   # exact duplicates must not leak
    train, test = load_cicids2017(cicids_dir, test_size=0.25, seed=0)
    both = pd.concat([train.X, test.X])
    assert not both.duplicated().any()
    for name in ("BENIGN", "DDoS", "PortScan", "Web Attack - Brute Force"):
        assert (test.attack == name).any() and (train.attack == name).any()


def test_stratified_split_keeps_singletons_in_train(nsl_kdd_dir):
    rec, _ = load_nsl_kdd(nsl_kdd_dir)
    rec = rec.subset(list(range(50)) + [int(np.flatnonzero(rec.attack == "guess_passwd")[0])])
    a, b = stratified_split(rec, 0.3, seed=0)
    assert len(a) + len(b) == len(rec)
    assert (a.attack == "guess_passwd").sum() >= 1
