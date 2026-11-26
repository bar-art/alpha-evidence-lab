import csv

import numpy as np
import pytest

from alpha_evidence_lab.factor_dsl import FactorError, evaluate, parse
from alpha_evidence_lab.panel import load_panel


def inputs_from(close, volume=None):
    close = np.asarray(close, dtype=float)
    ret = np.full_like(close, np.nan)
    ret[1:] = close[1:] / close[:-1] - 1
    values = {name: close for name in ["open", "high", "low", "close"]}
    values["volume"] = np.asarray(volume if volume is not None else np.ones_like(close), float)
    values["ret"] = ret
    return values


def run(expression, close, volume=None, universe=None):
    data = inputs_from(close, volume)
    mask = np.ones(data["close"].shape, dtype=bool) if universe is None else universe
    return evaluate(parse(expression), data, mask)


@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo')",
        "close.real",
        "close[0]",
        "(lambda x: x)(close)",
        "[c for c in close]",
        "close ** 2",
        "close if close else open",
        "close > open",
        "not close",
        "ts_mean(close, window=5)",
        "ts_mean(*[close, 5])",
        "ts_mean(close, 7)",
        "ts_mean(close, 5.0)",
        "ts_mean(close, True)",
        "ts_std(close, 1)",
        "ts_corr(close, volume, 2)",
        "ts_mean(close)",
        "cs_rank(close, 5)",
        "foo(close)",
        "amount",
        "'close'",
        "close + 1e9",
        "delay(delay(delay(close, 120), 120), 120)",
        "-" * 9 + "close",
        " + ".join(["close"] * 30),
        "",
        "close; open",
    ],
)
def test_rejects_anything_outside_the_whitelist(expression):
    with pytest.raises(FactorError):
        parse(expression)


def test_canonical_form_lookback_and_id():
    factor = parse("cs_rank(  close/delay(close,5) )")
    assert factor.expression == "cs_rank(close / delay(close, 5))"
    assert factor.lookback == 5
    assert factor.id == parse("cs_rank(close / delay(close, 5))").id
    assert parse("ts_mean(ret, 20)").lookback == 20  # ret 需要前一天，均值再回看 19 天
    assert parse("ts_corr(ret, volume, 10)").lookback == 10
    assert parse("delta(ts_mean(close, 5), 3)").lookback == 7


def test_hand_calculated_time_series_operators():
    close = [[1.0], [2.0], [4.0], [4.0], [8.0]]
    assert np.allclose(
        run("ts_mean(close, 2)", close)[:, 0], [np.nan, 1.5, 3, 4, 6], equal_nan=True
    )
    assert np.allclose(
        run("ts_sum(close, 3)", close)[:, 0], [np.nan, np.nan, 7, 10, 16], equal_nan=True
    )
    std = run("ts_std(close, 3)", close)[:, 0]
    assert np.isclose(std[2], np.std([1, 2, 4], ddof=1))
    assert np.allclose(run("delta(close, 1)", close)[:, 0], [np.nan, 1, 2, 0, 4], equal_nan=True)
    assert np.allclose(
        run("delay(close, 2)", close)[:, 0], [np.nan, np.nan, 1, 2, 4], equal_nan=True
    )
    pct = run("pct_change(close, 2)", close)[:, 0]
    assert np.allclose(pct, [np.nan, np.nan, 3, 1, 1], equal_nan=True)
    assert np.allclose(run("ts_min(close, 2)", close)[:, 0], [np.nan, 1, 2, 4, 4], equal_nan=True)
    assert np.allclose(run("ts_max(close, 2)", close)[:, 0], [np.nan, 2, 4, 4, 8], equal_nan=True)
    # ts_rank：今天在窗口中的平均名次百分位；01-04 与前一天并列。
    rank = run("ts_rank(close, 2)", close)[:, 0]
    assert np.allclose(rank, [np.nan, 1.0, 1.0, 0.75, 1.0], equal_nan=True)
    corr = run("ts_corr(close, volume, 3)", close, [[1.0], [2.0], [3.0], [5.0], [4.0]])[:, 0]
    assert np.isclose(corr[2], np.corrcoef([1, 2, 4], [1, 2, 3])[0, 1])
    flat = run("ts_corr(close, volume, 3)", close, [[1.0]] * 5)[:, 0]
    assert np.isnan(flat).all()  # 零方差不给相关系数


def test_missing_values_poison_whole_window_and_unsafe_math_is_nan():
    close = [[1.0], [np.nan], [3.0], [4.0], [5.0]]
    mean = run("ts_mean(close, 2)", close)[:, 0]
    assert np.isnan(mean[:3]).all() and mean[3] == 3.5
    assert np.isnan(run("close / (close - close)", [[1.0], [2.0]])).all()
    assert np.isnan(run("log(close - 2)", [[1.0], [2.0]])).all()
    assert run("log(close)", [[np.e]])[0, 0] == pytest.approx(1)
    assert np.isfinite(run("sign(-close) * abs(close)", [[3.0]])).all()


def test_cross_sectional_operators_use_only_the_universe():
    close = np.array([[3.0, 1.0, 2.0, 2.0, 9.0]])
    universe = np.array([[True, True, True, True, False]])
    rank = run("cs_rank(close)", close, universe=universe)[0]
    assert np.allclose(rank[:4], [1.0, 0.0, 0.5, 0.5]) and np.isnan(rank[4])
    z = run("cs_zscore(close)", close, universe=universe)[0]
    members = close[0, :4]
    assert np.allclose(z[:4], (members - members.mean()) / members.std(ddof=1))
    assert np.isnan(z[4])
    single = run("cs_rank(close)", close, universe=np.array([[True, False, False, False, False]]))
    assert single[0, 0] == 0.5


ALL_OPERATORS = [
    "ts_mean(ret, 5)",
    "ts_sum(volume, 3)",
    "ts_std(ret, 10)",
    "ts_min(low, 5)",
    "ts_max(high, 5)",
    "ts_rank(close, 10)",
    "delta(close, 3)",
    "delay(open, 2)",
    "pct_change(close, 5)",
    "ts_corr(ret, volume, 10)",
    "cs_rank(ret)",
    "cs_zscore(log(volume))",
    "sign(ret) * abs(delta(close, 1)) / ts_mean(close, 20)",
]


@pytest.mark.parametrize("expression", ALL_OPERATORS)
def test_future_data_never_changes_past_factor_values(expression):
    rng = np.random.default_rng(3)
    shape = (80, 6)
    close = np.cumprod(1 + rng.normal(0, 0.02, shape), axis=0) * 10
    data = inputs_from(close, rng.uniform(1e5, 1e6, shape))
    data["high"], data["low"] = close * 1.01, close * 0.99
    universe = rng.random(shape) > 0.2
    factor = parse(expression)
    before = evaluate(factor, data, universe)
    cut = 50
    changed = {name: values.copy() for name, values in data.items()}
    for values in changed.values():
        values[cut + 1 :] = rng.normal(5, 3, values[cut + 1 :].shape)
    future_universe = universe.copy()
    future_universe[cut + 1 :] = ~future_universe[cut + 1 :]
    after = evaluate(factor, changed, future_universe)
    np.testing.assert_array_equal(before[: cut + 1], after[: cut + 1])


def write_bars(path, rows):
    columns = [
        "ts_code", "trade_date", "open", "high", "low", "close", "volume", "available_at",
        "is_suspended", "up_limit", "down_limit", "price_limit_exempt", "universe_member",
    ]  # fmt: skip
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def bar(code, day, close, suspended=False, member=True, stamp=None):
    stamp = stamp or f"{day}T15:00:00+08:00"
    return [code, day, close, close, close, close, 1000, stamp, suspended, close * 1.1, close * 0.9,
            False, member]  # fmt: skip


def test_panel_marks_suspension_and_missing_rows_as_no_data(tmp_path):
    rows = [
        bar("A", "2024-01-02", 10.0),
        bar("A", "2024-01-03", 10.0, suspended=True),
        bar("A", "2024-01-04", 11.0),
        bar("B", "2024-01-03", 5.0, member=False),
        bar("B", "2024-01-04", 6.0),
    ]
    write_bars(tmp_path / "bars.csv", rows)
    (tmp_path / "manifest.json").write_text('{"source": "unit"}')
    panel = load_panel(tmp_path / "bars.csv", tmp_path / "manifest.json")
    assert panel.symbols == ("A", "B") and len(panel.dates) == 3
    assert panel.suspended[0, 1] and not panel.member[0, 1]  # B 在 01-02 尚无行情
    inputs = panel.factor_inputs()
    assert np.isnan(inputs["close"][1, 0])  # 停牌沿用价不进入因子
    assert np.isnan(inputs["ret"][2, 0])  # 复牌日的收益跨越停牌，不计算
    assert inputs["ret"][2, 1] == pytest.approx(0.2)
    assert panel.window(1, 3).dates.tolist() == panel.dates[1:3].tolist()
    assert len(panel.sha256) == 64


@pytest.mark.parametrize(
    "rows, message",
    [
        ([bar("A", "2024-01-02", 10.0)] * 2, "重复"),
        ([bar("A", "2024-01-02", 10.0, suspended="maybe")], "布尔"),
        ([bar("A", "2024-01-02", 10.0, stamp="2024-01-03T09:00:00+08:00")], "可得时间"),
        ([bar("A", "2024-01-02", -1.0)], "为正"),
    ],
)
def test_panel_rejects_bad_input(tmp_path, rows, message):
    write_bars(tmp_path / "bars.csv", rows)
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match=message):
        load_panel(tmp_path / "bars.csv", tmp_path / "manifest.json")
