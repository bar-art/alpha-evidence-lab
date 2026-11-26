"""受限因子表达式：白名单字段、算子和窗口；用 ast 解析后自行求值，从不 eval。

示例：cs_rank(-delta(close, 5) / ts_std(ret, 20))

- 时间序列算子只用当日及之前的数据，窗口必须完整（窗口内任一缺失或停牌即为 NaN）。
- 横截面算子只在"当日成分股且有成交"的证券之间计算，其余为 NaN。
- 除零、非正数取对数和无穷大一律变为 NaN。
"""

import ast
import hashlib
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

FIELDS = ("open", "high", "low", "close", "volume", "ret")
WINDOWS = (1, 2, 3, 5, 10, 20, 40, 60, 120)
MAX_DEPTH = 8
MAX_NODES = 50
MAX_LOOKBACK = 250
MAX_CONSTANT = 1e6
CHUNK_ELEMENTS = 20_000_000

# 名称 → (序列参数个数, 是否带窗口, 最小窗口, 额外回看=窗口-offset)
TS_FUNCS = {
    "ts_mean": (1, 1),
    "ts_sum": (1, 1),
    "ts_std": (1, 2),
    "ts_min": (1, 1),
    "ts_max": (1, 1),
    "ts_rank": (1, 2),
    "delta": (1, 1),
    "delay": (1, 1),
    "pct_change": (1, 1),
    "ts_corr": (2, 3),
}
CS_FUNCS = ("cs_rank", "cs_zscore")
UNARY_FUNCS = ("log", "abs", "sign")
SHIFTS = ("delta", "delay", "pct_change")


class FactorError(ValueError):
    pass


@dataclass(frozen=True)
class Factor:
    expression: str  # 规范化后的表达式
    lookback: int  # 需要的历史交易日数（暖机长度）
    depth: int
    nodes: int

    @property
    def id(self) -> str:
        return "f_" + hashlib.sha256(self.expression.encode()).hexdigest()[:12]


def _check(node: ast.AST, depth: int, counter: list[int]) -> tuple[int, int]:
    """校验子树，返回 (回看天数, 深度)。"""
    counter[0] += 1
    if counter[0] > MAX_NODES:
        raise FactorError(f"表达式节点超过 {MAX_NODES} 个")
    if depth > MAX_DEPTH:
        raise FactorError(f"表达式嵌套超过 {MAX_DEPTH} 层")
    if isinstance(node, ast.Name):
        if node.id not in FIELDS:
            raise FactorError(f"未知字段 {node.id}；允许 {FIELDS}")
        return (1 if node.id == "ret" else 0), depth
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise FactorError("常数只能是数值")
        if not abs(value) <= MAX_CONSTANT:
            raise FactorError("常数超出范围")
        return 0, depth
    if isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ast.USub):
            raise FactorError("只允许取负号")
        return _check(node.operand, depth + 1, counter)
    if isinstance(node, ast.BinOp):
        if not isinstance(node.op, ast.Add | ast.Sub | ast.Mult | ast.Div):
            raise FactorError("只允许 + - * /")
        left = _check(node.left, depth + 1, counter)
        right = _check(node.right, depth + 1, counter)
        return max(left[0], right[0]), max(left[1], right[1])
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.keywords:
            raise FactorError("只允许调用白名单函数，且不能用关键字参数")
        name, args = node.func.id, node.args
        if any(isinstance(arg, ast.Starred) for arg in args):
            raise FactorError("不允许 * 展开参数")
        if name in TS_FUNCS:
            series, minimum = TS_FUNCS[name]
            if len(args) != series + 1:
                raise FactorError(f"{name} 需要 {series} 个序列和 1 个窗口")
            window = args[-1]
            if (
                not isinstance(window, ast.Constant)
                or isinstance(window.value, bool)
                or not isinstance(window.value, int)
                or window.value not in WINDOWS
            ):
                raise FactorError(f"{name} 的窗口必须是 {WINDOWS} 之一")
            if window.value < minimum:
                raise FactorError(f"{name} 的窗口至少为 {minimum}")
            counter[0] += 1
            children = [_check(arg, depth + 1, counter) for arg in args[:-1]]
            extra = window.value if name in SHIFTS else window.value - 1
            return max(c[0] for c in children) + extra, max(c[1] for c in children)
        if name in CS_FUNCS or name in UNARY_FUNCS:
            if len(args) != 1:
                raise FactorError(f"{name} 只接受 1 个参数")
            return _check(args[0], depth + 1, counter)
        raise FactorError(f"未知函数 {name}")
    raise FactorError(f"不允许的语法：{type(node).__name__}")


def parse(expression: str) -> Factor:
    if not isinstance(expression, str) or not expression.strip() or len(expression) > 500:
        raise FactorError("表达式必须是 1–500 个字符的字符串")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise FactorError(f"语法错误：{error.msg}") from None
    counter = [0]
    lookback, depth = _check(tree.body, 1, counter)
    if lookback > MAX_LOOKBACK:
        raise FactorError(f"回看 {lookback} 天，超过上限 {MAX_LOOKBACK}")
    return Factor(ast.unparse(tree.body), lookback, depth, counter[0])


def _clean(values: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(values), values, np.nan)


def _rolling(x: np.ndarray, n: int, reduce: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """完整窗口滚动：结果第 t 行只用 x[t-n+1 : t+1]；窗口内有 NaN 时为 NaN。按列分块控制内存。"""
    rows, cols = x.shape
    result = np.full((rows, cols), np.nan)
    if rows < n:
        return result
    step = max(1, CHUNK_ELEMENTS // max(1, rows * n))
    for begin in range(0, cols, step):
        block = x[:, begin : begin + step]
        windows = sliding_window_view(block, n, axis=0)  # (rows-n+1, chunk, n)
        with np.errstate(invalid="ignore", divide="ignore"):
            values = reduce(windows)
        complete = np.isfinite(windows).all(axis=-1)
        result[n - 1 :, begin : begin + step] = np.where(complete, values, np.nan)
    return result


def _ts_rank(windows: np.ndarray) -> np.ndarray:
    last = windows[..., -1:]
    less = (windows < last).sum(axis=-1)
    equal = (windows == last).sum(axis=-1)
    return np.asarray((less + (equal + 1) / 2) / windows.shape[-1], dtype=float)


def _shift(x: np.ndarray, n: int) -> np.ndarray:
    result = np.full_like(x, np.nan)
    result[n:] = x[:-n]
    return result


def _corr(x: np.ndarray, y: np.ndarray, n: int) -> np.ndarray:
    stacked = np.stack([x, y], axis=-1)  # T×N×2
    rows, cols, _ = stacked.shape
    result = np.full((rows, cols), np.nan)
    if rows < n:
        return result
    step = max(1, CHUNK_ELEMENTS // max(1, rows * n * 2))
    for begin in range(0, cols, step):
        block = stacked[:, begin : begin + step]
        windows = sliding_window_view(block, n, axis=0)  # (rows-n+1, chunk, 2, n)
        a, b = windows[:, :, 0, :], windows[:, :, 1, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            da, db = a - a.mean(axis=-1, keepdims=True), b - b.mean(axis=-1, keepdims=True)
            denominator = np.sqrt((da**2).sum(axis=-1) * (db**2).sum(axis=-1))
            values = (da * db).sum(axis=-1) / denominator
        complete = np.isfinite(a).all(axis=-1) & np.isfinite(b).all(axis=-1)
        result[n - 1 :, begin : begin + step] = np.where(
            complete & (denominator > 1e-12), values, np.nan
        )
    return result


def _cs_rank(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """每日在可用证券中求平均名次百分位，范围 [0, 1]；只有 1 只时为 0.5。"""
    result = np.full_like(x, np.nan)
    for t in range(x.shape[0]):
        valid = mask[t] & np.isfinite(x[t])
        count = int(valid.sum())
        if count == 0:
            continue
        values = x[t, valid]
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty(count)
        sorted_values = values[order]
        start = 0
        while start < count:
            stop = start
            while stop + 1 < count and sorted_values[stop + 1] == sorted_values[start]:
                stop += 1
            ranks[order[start : stop + 1]] = (start + stop) / 2
            start = stop + 1
        result[t, valid] = ranks / (count - 1) if count > 1 else 0.5
    return result


def _cs_zscore(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """每日在可用证券中标准化；少于 2 只或标准差为 0 时为 NaN。"""
    valid = mask & np.isfinite(x)
    count = valid.sum(axis=1)
    data = np.where(valid, x, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = data.sum(axis=1) / count
        spread = np.where(valid, x - mean[:, None], 0.0)
        std = np.sqrt((spread**2).sum(axis=1) / (count - 1))
        result = (x - mean[:, None]) / std[:, None]
    usable = (count >= 2) & (std > 1e-12)
    return _clean(np.where(valid & usable[:, None], result, np.nan))


def evaluate(factor: Factor, inputs: dict[str, np.ndarray], universe: np.ndarray) -> np.ndarray:
    """在 T×N 输入上求因子值。universe 为横截面算子使用的"当日可比证券"掩码。"""
    tree = ast.parse(factor.expression, mode="eval").body

    def run(node: ast.AST) -> np.ndarray:
        if isinstance(node, ast.Name):
            return inputs[node.id].astype(float)
        if isinstance(node, ast.Constant):
            assert isinstance(node.value, int | float)
            return np.full(universe.shape, float(node.value))
        if isinstance(node, ast.UnaryOp):
            return -run(node.operand)
        if isinstance(node, ast.BinOp):
            left, right = run(node.left), run(node.right)
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                if isinstance(node.op, ast.Add):
                    return _clean(left + right)
                if isinstance(node.op, ast.Sub):
                    return _clean(left - right)
                if isinstance(node.op, ast.Mult):
                    return _clean(left * right)
                return _clean(np.where(np.abs(right) > 1e-12, left / right, np.nan))
        assert isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        name, args = node.func.id, node.args
        if name in TS_FUNCS:
            window = int(args[-1].value)  # type: ignore[attr-defined]
            x = run(args[0])
            with np.errstate(invalid="ignore", divide="ignore"):
                if name == "ts_mean":
                    return _rolling(x, window, lambda w: w.mean(axis=-1))
                if name == "ts_sum":
                    return _rolling(x, window, lambda w: w.sum(axis=-1))
                if name == "ts_std":
                    return _rolling(x, window, lambda w: w.std(axis=-1, ddof=1))
                if name == "ts_min":
                    return _rolling(x, window, lambda w: w.min(axis=-1))
                if name == "ts_max":
                    return _rolling(x, window, lambda w: w.max(axis=-1))
                if name == "ts_rank":
                    return _rolling(x, window, _ts_rank)
                if name == "delta":
                    return _clean(x - _shift(x, window))
                if name == "delay":
                    return _shift(x, window)
                if name == "pct_change":
                    previous = _shift(x, window)
                    return _clean(np.where(np.abs(previous) > 1e-12, x / previous - 1, np.nan))
                return _corr(x, run(args[1]), window)
        x = run(args[0])
        with np.errstate(invalid="ignore", divide="ignore"):
            if name == "cs_rank":
                return _cs_rank(x, universe)
            if name == "cs_zscore":
                return _cs_zscore(x, universe)
            if name == "log":
                return _clean(np.where(x > 0, np.log(np.where(x > 0, x, 1.0)), np.nan))
            if name == "abs":
                return np.asarray(np.abs(x), dtype=float)
            return np.asarray(np.sign(x), dtype=float)

    return _clean(run(tree))
