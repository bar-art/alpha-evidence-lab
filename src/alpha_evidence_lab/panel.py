"""横截面面板：读取 a-share-quant-agent 导出给 quant-research-lab 的 bars.csv（多股票长表）。

存成"交易日 × 证券"的 numpy 矩阵。缺行（未上市、已退市）为 NaN 且不可交易。
停牌日的价格是沿用值，因子计算视为无数据，不当作 0 收益。
只接受收盘后可得的日线：available_at 必须等于当日 15:00（上海时间），否则拒绝，不静默改时间戳。
"""

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

TZ = ZoneInfo("Asia/Shanghai")
REQUIRED = [
    "ts_code",
    "trade_date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "available_at",
    "is_suspended",
    "up_limit",
    "down_limit",
    "price_limit_exempt",
    "universe_member",
]
PRICE_FIELDS = ["open", "high", "low", "close", "volume"]


def _bool(value: str, column: str) -> bool:
    if value in {"True", "true", "1"}:
        return True
    if value in {"False", "false", "0"}:
        return False
    raise ValueError(f"{column} 必须是明确的布尔值，不能是 {value!r}")


def _float(value: str) -> float:
    return float(value) if value not in {"", "nan", "NaN"} else math.nan


@dataclass(frozen=True)
class Panel:
    dates: np.ndarray  # datetime64[D]，升序
    symbols: tuple[str, ...]
    fields: dict[str, np.ndarray]  # 原始字段，T×N；缺行为 NaN
    suspended: np.ndarray  # T×N bool；缺行为 True（不可交易）
    member: np.ndarray  # T×N bool；缺行为 False
    up_limit: np.ndarray
    down_limit: np.ndarray
    limit_exempt: np.ndarray
    sha256: str
    source: str

    @property
    def observed(self) -> np.ndarray:
        """有成交、可用于计算因子的格点：有行情且未停牌。"""
        return np.asarray(~self.suspended & np.isfinite(self.fields["close"]), dtype=bool)

    def factor_inputs(self) -> dict[str, np.ndarray]:
        """供因子表达式使用的字段：停牌和缺行为 NaN；ret 为相邻两日都有成交时的收盘收益。"""
        inputs = {
            name: np.where(self.observed, values, np.nan) for name, values in self.fields.items()
        }
        close = inputs["close"]
        ret = np.full_like(close, np.nan)
        ret[1:] = close[1:] / close[:-1] - 1
        inputs["ret"] = ret
        return inputs

    def window(self, start: int, stop: int) -> "Panel":
        """按交易日下标截取 [start, stop)，供训练/验证/测试分段使用。"""
        if not 0 <= start < stop <= len(self.dates):
            raise ValueError("面板截取区间无效")
        cut = slice(start, stop)
        return Panel(
            self.dates[cut],
            self.symbols,
            {name: values[cut] for name, values in self.fields.items()},
            self.suspended[cut],
            self.member[cut],
            self.up_limit[cut],
            self.down_limit[cut],
            self.limit_exempt[cut],
            self.sha256,
            self.source,
        )


def load_panel(bars: Path, manifest: Path) -> Panel:
    """读取长表并校验：必需列、键唯一、布尔值明确、可得时间为当日收盘。"""
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    digest = hashlib.sha256()
    digest.update(bars.read_bytes())
    digest.update(manifest.read_bytes())
    with bars.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"面板缺少字段：{sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError("面板为空")
    symbols = tuple(sorted({row["ts_code"] for row in rows}))
    dates = np.array(sorted({row["trade_date"][:10] for row in rows}), dtype="datetime64[D]")
    column = {symbol: index for index, symbol in enumerate(symbols)}
    row_of = {str(day): index for index, day in enumerate(dates)}
    shape = (len(dates), len(symbols))
    fields = {name: np.full(shape, np.nan) for name in PRICE_FIELDS}
    up, down = np.full(shape, np.nan), np.full(shape, np.nan)
    suspended = np.ones(shape, dtype=bool)
    member = np.zeros(shape, dtype=bool)
    exempt = np.zeros(shape, dtype=bool)
    seen = np.zeros(shape, dtype=bool)
    for row in rows:
        day = row["trade_date"][:10]
        t, n = row_of[day], column[row["ts_code"]]
        if seen[t, n]:
            raise ValueError(f"重复的证券日期：{row['ts_code']} {day}")
        seen[t, n] = True
        stamp = datetime.fromisoformat(row["available_at"])
        if stamp.tzinfo is None:
            raise ValueError("available_at 必须带时区")
        close_time = datetime.fromisoformat(f"{day}T15:00:00").replace(tzinfo=TZ)
        if stamp != close_time:
            raise ValueError(f"{row['ts_code']} {day} 的可得时间不是当日收盘；暂不支持延迟数据")
        for name in PRICE_FIELDS:
            fields[name][t, n] = _float(row[name])
        up[t, n], down[t, n] = _float(row["up_limit"]), _float(row["down_limit"])
        suspended[t, n] = _bool(row["is_suspended"], "is_suspended")
        member[t, n] = _bool(row["universe_member"], "universe_member")
        exempt[t, n] = _bool(row["price_limit_exempt"], "price_limit_exempt")
    prices = np.stack([fields[name] for name in ["open", "high", "low", "close"]])
    present = np.isfinite(prices)
    if (prices[present] <= 0).any():
        raise ValueError("价格必须为正")
    return Panel(
        dates,
        symbols,
        fields,
        suspended,
        member,
        up,
        down,
        exempt,
        digest.hexdigest(),
        str(meta.get("source", "")),
    )
