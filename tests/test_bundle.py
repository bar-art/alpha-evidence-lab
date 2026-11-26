import hashlib
import json

import pytest

from alpha_evidence_lab.bundle import load_bundle


def bundle(tmp_path):
    alpha = tmp_path / "alpha"
    alpha.mkdir()
    path = alpha / "000001.SZ.csv"
    path.write_text(
        "date,open,close,available_at,is_suspended,up_limit,down_limit,price_limit_exempt\n2024-01-02,10,10,2024-01-02T15:00:00+08:00,False,11,9,False\n"
    )
    meta = {
        "schema": "research_bundle_v1",
        "visibility": "close_assumed",
        "source": "Offline fixture, not market data",
        "alpha_files": [
            {"file": "alpha/000001.SZ.csv", "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        ],
    }
    (tmp_path / "bundle.json").write_text(json.dumps(meta))
    return meta


def test_bundle_hash_enters_research_identity_and_protects_files(tmp_path):
    metadata = bundle(tmp_path)
    paths = load_bundle(tmp_path)
    assert paths[0].bars[0].up_limit == 11
    assert "bundle_sha256=" in paths[0].source
    (tmp_path / "alpha" / "000001.SZ.csv").write_text("changed")
    with pytest.raises(ValueError, match="hash"):
        load_bundle(tmp_path)
    assert metadata["schema"] == "research_bundle_v1"


@pytest.mark.parametrize("change", ["version", "visibility", "traversal", "duplicate", "source"])
def test_bundle_fails_closed(tmp_path, change):
    metadata = bundle(tmp_path)
    if change == "version":
        metadata["schema"] = "unreviewed_v2"
    elif change == "visibility":
        metadata["visibility"] = "next_open"
    elif change == "traversal":
        metadata["alpha_files"][0]["file"] = "../outside.csv"
    elif change == "duplicate":
        metadata["alpha_files"] *= 2
    else:
        metadata["source"] = ""
    (tmp_path / "bundle.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load_bundle(tmp_path)
