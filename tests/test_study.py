import csv
import json
from dataclasses import replace
from datetime import timedelta

import pytest

from alpha_evidence_lab.cli import main
from alpha_evidence_lab.data import PricePath, load_csv
from alpha_evidence_lab.report import export, render
from alpha_evidence_lab.study import run_study


def test_preregister_budgets_and_one_shot_protocol(tmp_path, paths, plan):
    result = run_study(plan, tmp_path / "study", paths)
    assert len(result["trials"]) == len(paths) * plan.episodes * 3 * plan.budget
    assert len(result["results"]) == len(paths) * plan.episodes * 3
    assert result["gate"]["outcome"] == "insufficient_evidence"
    assert all(row["status"] == "ok" for row in result["trials"])
    first = result["batches"][0]
    assert first["visible_memory"] == []
    assert result["batches"][1]["visible_memory"]
    export(result, tmp_path / "study")
    assert main(["verify", "--output", str(tmp_path / "study")]) == 0
    with pytest.raises(ValueError):
        run_study(plan, tmp_path / "study", paths)
    # 默认报告不执行用户HTML。
    changed = {**result, "limitations": ["<script>unsafe</script>"]}
    assert "<script>" not in render(changed)
    assert "&lt;script&gt;" in render(changed)
    checkpoint = json.loads((tmp_path / "study/checkpoint.json").read_text())
    assert "audit.sqlite" in checkpoint["artifact_hashes"]
    (tmp_path / "study/attempts.csv").write_text("tampered")
    assert main(["verify", "--output", str(tmp_path / "study")]) == 1


def test_holdout_perturbation_cannot_change_any_search(tmp_path, paths, plan):
    original = run_study(plan, tmp_path / "original", paths)
    updated_paths = []
    size = plan.train_sessions + plan.validation_sessions + plan.holdout_sessions
    for path in paths:
        changed = tuple(
            replace(bar, opening=bar.opening * 3, closing=bar.closing * 3)
            if i % size >= size - plan.holdout_sessions
            else bar
            for i, bar in enumerate(path.bars)
        )
        updated_paths.append(PricePath(path.id, changed))
    updated = run_study(plan, tmp_path / "changed", tuple(updated_paths))
    assert [row["winners"] for row in original["batches"]] == [
        row["winners"] for row in updated["batches"]
    ]
    assert [row["score"] for row in original["trials"]] == [
        row["score"] for row in updated["trials"]
    ]


def test_dataset_integrity_and_fail_closed_split(tmp_path, paths, plan):
    with pytest.raises(ValueError):
        run_study(plan, tmp_path / "short", (PricePath("short", paths[0].bars[:20]),))
    with pytest.raises(ValueError):
        run_study(plan, tmp_path / "duplicate", (paths[0], paths[0]))
    changed = list(paths[0].bars)
    changed[1] = replace(changed[1], available_at=changed[1].available_at + timedelta(days=100))
    with pytest.raises(ValueError):
        run_study(plan, tmp_path / "late", (PricePath("late", tuple(changed)),))


def test_csv_cli_and_no_fake_network(tmp_path, paths, plan):
    source = tmp_path / "bars.csv"
    with source.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "open", "close", "available_at"])
        for bar in paths[0].bars:
            writer.writerow([bar.day, bar.opening, bar.closing, bar.available_at.isoformat()])
    loaded = load_csv(source, "external", "人工测试输入")
    assert loaded.kind == "user_supplied"
    config = tmp_path / "plan.json"
    config.write_text(plan.model_dump_json())
    args = [
        "research",
        "--input",
        str(source),
        "--source",
        "人工测试输入",
        "--config",
        str(config),
        "--output",
        str(tmp_path / "cli"),
    ]
    assert main(args) == 0
    assert main(["research", "--output", str(tmp_path / "invalid")]) == 1
    assert main(["verify", "--output", str(tmp_path / "absent")]) == 1
    source.write_text("x,y\n1,2\n")
    with pytest.raises(ValueError):
        load_csv(source, "bad", "missing fields")
