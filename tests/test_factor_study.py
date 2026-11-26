import csv
import json
import sqlite3
from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from alpha_evidence_lab.cli import main
from alpha_evidence_lab.factor_study import episodes, run_factor_study
from alpha_evidence_lab.models import FactorPlan
from alpha_evidence_lab.panel import Panel, load_panel
from alpha_evidence_lab.report import export_factor


def random_panel(days=480, stocks=35):
    rng = np.random.default_rng(42)
    prices = 10 * np.exp(np.cumsum(rng.normal(0, 0.02, (days, stocks)), axis=0))
    shape = prices.shape
    return Panel(
        np.arange(np.datetime64("2024-01-01"), np.datetime64("2024-01-01") + days),
        tuple(f"S{i:03d}" for i in range(stocks)),
        {
            "open": prices,
            "close": prices,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "volume": np.exp(rng.normal(10, 0.5, shape)),
        },
        np.zeros(shape, bool),
        np.ones(shape, bool),
        prices * 2,
        prices * 0.5,
        np.ones(shape, bool),
        "synthetic-fixture",
        "Synthetic test only",
    )


def write_panel(p, root, reverse=False):
    root.mkdir()
    keys = [
        "ts_code",
        "trade_date",
        "available_at",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "is_suspended",
        "up_limit",
        "down_limit",
        "price_limit_exempt",
        "universe_member",
    ]
    rows = [
        {
            "ts_code": symbol,
            "trade_date": str(day),
            "available_at": str(day) + "T15:00:00+08:00",
            **{k: float(v[t, n]) for k, v in p.fields.items()},
            "is_suspended": False,
            "up_limit": 100000,
            "down_limit": 0.0001,
            "price_limit_exempt": True,
            "universe_member": True,
        }
        for t, day in enumerate(p.dates)
        for n, symbol in enumerate(p.symbols)
    ]
    with (root / "bars.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows[::-1] if reverse else rows)
    (root / "manifest.json").write_text(json.dumps({"source": "Synthetic offline fixture"}))
    return root / "bars.csv", root / "manifest.json"


def test_calendar_boundaries_and_confirmation():
    p = random_panel()
    plan = FactorPlan(confirm_start=p.dates[395].astype(object), max_episodes=3)
    schedule = episodes(p, plan)
    assert schedule[0]["test_start"] == 375
    assert schedule[0]["test_start"] - schedule[0]["dev_stop"] == 5
    assert schedule[0]["dev_stop"] - schedule[0]["dev_start"] == 120
    assert schedule[0]["test_stop"] == schedule[1]["test_start"]
    assert [r["phase"] for r in schedule] == ["exploration", "exploration", "confirmation"]
    assert (
        len(episodes(p, plan.model_copy(update={"confirmation_only": True, "max_episodes": 1})))
        == 1
    )
    with pytest.raises(ValueError):
        episodes(p.window(0, 300), plan)
    with pytest.raises(ValueError):
        FactorPlan(holdout_sessions=5, horizon=5)


def test_joint_freeze_and_replay(tmp_path, monkeypatch):
    import alpha_evidence_lab.factor_study as module

    actual = module.evaluate_factor
    out = tmp_path / "study"

    def checked(*args, **kwargs):
        with sqlite3.connect(out / "audit.sqlite") as db:
            state = db.execute("SELECT state FROM batches ORDER BY rowid DESC LIMIT 1").fetchone()[
                0
            ]
        assert state == "frozen"
        return actual(*args, **kwargs)

    monkeypatch.setattr(module, "evaluate_factor", checked)
    plan = FactorPlan(budget=2, max_episodes=2, confirm_start=date(2020, 1, 1))
    first = run_factor_study(plan, random_panel(), out)
    assert len(first["trials"]) == 8 and len(first["results"]) == 4
    assert all(len(r["metrics"]["period_returns"]) == 3 for r in first["results"])
    export_factor(first, out)
    assert main(["verify", "--output", str(out)]) == 0
    with (out / "attempts.csv").open("a") as handle:
        handle.write("tampered")
    assert main(["verify", "--output", str(out)]) == 1


def test_cli_and_input_order_do_not_change_selection(tmp_path):
    p = random_panel()
    a = write_panel(p, tmp_path / "a")
    b = write_panel(p, tmp_path / "b", True)
    plan = FactorPlan(budget=2, max_episodes=1, confirm_start=date(2020, 1, 1))
    config = tmp_path / "config.json"
    config.write_text(plan.model_dump_json())
    assert (
        main(
            [
                "factor-study",
                "--bars",
                str(a[0]),
                "--manifest",
                str(a[1]),
                "--config",
                str(config),
                "--output",
                str(tmp_path / "cli"),
            ]
        )
        == 0
    )
    results = [
        run_factor_study(plan, load_panel(*files), tmp_path / f"run{i}")
        for i, files in enumerate([a, b])
    ]
    assert results[0]["results"] == results[1]["results"]
    assert [(r["candidate"], r["score"]) for r in results[0]["trials"]] == [
        (r["candidate"], r["score"]) for r in results[1]["trials"]
    ]


def test_development_score_does_not_read_sealed_prices(tmp_path):
    p = random_panel()
    plan = FactorPlan(budget=2, max_episodes=1)
    modified = {k: v.copy() for k, v in p.fields.items()}
    modified["open"][375:] *= 1.2
    modified["close"][375:] *= 1.2
    a = run_factor_study(plan, p, tmp_path / "a")
    b = run_factor_study(plan, replace(p, fields=modified), tmp_path / "b")
    assert [(r["candidate"], r["score"]) for r in a["trials"]] == [
        (r["candidate"], r["score"]) for r in b["trials"]
    ]
