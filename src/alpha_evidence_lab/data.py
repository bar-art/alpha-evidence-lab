"""单资产日线研究输入；拒绝缺失/无穷价格，不悄悄修复时间轴。"""

import csv
import math
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

from .models import StudyPlan, digest

TZ = ZoneInfo("Asia/Shanghai")


@dataclass(frozen=True)
class Bar:
    day: date
    opening: float
    closing: float
    available_at: datetime
    is_suspended: bool = False
    up_limit: float | None = None
    down_limit: float | None = None
    price_limit_exempt: bool = True

    def validate(self) -> None:
        if any(not math.isfinite(value) or value <= 0 for value in [self.opening, self.closing]):
            raise ValueError("价格必须为正的有限数")
        if any(
            value is not None and (not math.isfinite(value) or value <= 0)
            for value in [self.up_limit, self.down_limit]
        ):
            raise ValueError("涨跌停价格必须为正数或未知")
        if self.available_at.tzinfo is None:
            raise ValueError("available_at 必须含时区")
        if self.available_at < datetime.combine(self.day, time(15), TZ):
            raise ValueError("完整日线不能早于收盘可得")


@dataclass(frozen=True)
class PricePath:
    id: str
    bars: tuple[Bar, ...]
    kind: str = "synthetic"
    source: str = "人工生成；工作日不是真实交易日历"

    def __post_init__(self) -> None:
        if (
            not self.bars
            or self.kind not in {"synthetic", "user_supplied"}
            or not self.source.strip()
        ):
            raise ValueError("数据路径、类型或来源不合法")
        previous: date | None = None
        for bar in self.bars:
            bar.validate()
            if previous is not None and bar.day <= previous:
                raise ValueError("日期重复或没有严格递增")
            previous = bar.day

    @property
    def sha256(self) -> str:
        return digest(
            {
                "id": self.id,
                "kind": self.kind,
                "source": self.source,
                "bars": [
                    {
                        **asdict(bar),
                        "day": str(bar.day),
                        "available_at": bar.available_at.isoformat(),
                    }
                    for bar in self.bars
                ],
            }
        )

    def known(self, end: int) -> tuple[Bar, ...]:
        if end < 0 or end >= len(self.bars):
            raise ValueError("数据截止位置越界")
        cutoff = datetime.combine(self.bars[end].day, time(15), TZ)
        visible = tuple(bar for bar in self.bars[: end + 1] if bar.available_at <= cutoff)
        if len(visible) != end + 1:
            raise ValueError("历史数据尚未按时公开；本版拒绝补值或隐式前视")
        return visible


def synthetic_paths(plan: StudyPlan) -> tuple[PricePath, ...]:
    count = plan.episodes * (plan.train_sessions + plan.validation_sessions + plan.holdout_sessions)
    days: list[date] = []
    day = date(2020, 1, 1)
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    paths = []
    for seed in plan.data_seeds:
        rng = np.random.default_rng(seed)
        bars = []
        last = 100.0
        for i, day in enumerate(days):
            drift = 0.0006 if (i // 60) % 2 == 0 else -0.0003
            opening = last * math.exp(float(rng.normal(0, 0.004)))
            closing = opening * math.exp(float(rng.normal(drift, 0.012)))
            bars.append(Bar(day, opening, closing, datetime.combine(day, time(15), TZ)))
            last = closing
        paths.append(PricePath(f"seed_{seed}", tuple(bars)))
    return tuple(paths)


def load_csv(path: Path, path_id: str, source: str) -> PricePath:
    def boolean(row: dict[str, str], key: str, default: bool) -> bool:
        value = row.get(key)
        if value is None:
            return default
        if value.lower() not in {"true", "false"}:
            raise ValueError(f"{key} 必须为 True/False")
        return value.lower() == "true"

    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"date", "open", "close", "available_at"} <= set(reader.fieldnames or []):
            raise ValueError("CSV需要 date/open/close/available_at 字段")
        bars = tuple(
            Bar(
                date.fromisoformat(row["date"]),
                float(row["open"]),
                float(row["close"]),
                datetime.fromisoformat(row["available_at"]),
                is_suspended=boolean(row, "is_suspended", False),
                up_limit=float(row["up_limit"]) if row.get("up_limit") else None,
                down_limit=float(row["down_limit"]) if row.get("down_limit") else None,
                price_limit_exempt=boolean(row, "price_limit_exempt", False),
            )
            for row in reader
        )
    return PricePath(path_id, bars, kind="user_supplied", source=source)
