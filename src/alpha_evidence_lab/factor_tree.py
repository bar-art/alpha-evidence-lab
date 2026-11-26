"""Versioned factor trees: deterministic serialization, never execute model code."""

import ast
from typing import Any

from .factor_dsl import CS_FUNCS, FIELDS, TS_FUNCS, UNARY_FUNCS, WINDOWS, FactorError

OPS = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
FUNCTIONS: dict[str, dict[str, Any]] = {
    **{
        name: {
            "series_args": arity,
            "window_min": minimum,
            "meaning": {
                "ts_mean": "rolling mean",
                "ts_sum": "rolling sum",
                "ts_std": "sample standard deviation",
                "ts_min": "rolling minimum",
                "ts_max": "rolling maximum",
                "ts_rank": "last value rolling rank",
                "delta": "difference from lag",
                "delay": "lagged value",
                "pct_change": "fractional change from lag",
                "ts_corr": "rolling Pearson correlation",
            }[name],
        }
        for name, (arity, minimum) in TS_FUNCS.items()
    },
    **{
        name: {
            "series_args": 1,
            "window_min": None,
            "meaning": {
                "cs_rank": "cross-sectional average rank",
                "cs_zscore": "cross-sectional z score",
                "log": "natural logarithm",
                "abs": "absolute value",
                "sign": "sign",
            }[name],
        }
        for name in (*CS_FUNCS, *UNARY_FUNCS)
    },
}
EXAMPLES = [
    {"op": "neg", "args": [{"fn": "pct_change", "args": [{"field": "close"}], "window": 5}]},
    {"fn": "ts_std", "args": [{"field": "ret"}], "window": 20},
    {
        "op": "div",
        "args": [
            {"fn": "ts_mean", "args": [{"field": "volume"}], "window": 5},
            {"fn": "ts_mean", "args": [{"field": "volume"}], "window": 20},
        ],
    },
]


def object_schema(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def tree_schema() -> dict[str, Any]:
    # Official OpenAI Structured Outputs docs (checked 2026-11-26): recursive
    # $defs/$ref and nested anyOf supported; root must be object, all keys required,
    # additionalProperties false. Functions with/without window are separate variants.
    # https://developers.openai.com/api/docs/guides/structured-outputs
    args = {"type": "array", "items": {"$ref": "#/$defs/node"}}
    return {
        "anyOf": [
            object_schema({"field": {"type": "string", "enum": list(FIELDS)}}),
            object_schema({"const": {"type": "number"}}),
            object_schema({"op": {"type": "string", "enum": [*OPS, "neg"]}, "args": args}),
            object_schema(
                {
                    "fn": {"type": "string", "enum": list(TS_FUNCS)},
                    "args": args,
                    "window": {"type": "integer", "enum": list(WINDOWS)},
                }
            ),
            object_schema(
                {"fn": {"type": "string", "enum": [*CS_FUNCS, *UNARY_FUNCS]}, "args": args}
            ),
        ]
    }


def schema_issue(value: Any, rule: dict[str, Any], root: dict[str, Any], depth: int = 0) -> str:
    """Small offline validator for exactly our recursive schema subset; no dependency."""
    if depth > 64:
        return "Schema nesting safety limit"
    if "$ref" in rule:
        return schema_issue(value, root["$defs"][rule["$ref"].split("/")[-1]], root, depth + 1)
    if "anyOf" in rule:
        return (
            "No matching node variant"
            if all(schema_issue(value, r, root, depth + 1) for r in rule["anyOf"])
            else ""
        )
    kind = rule.get("type")
    valid = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "number": isinstance(value, int | float) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
    }
    if kind in valid and not valid[kind]:
        return "Invalid node type"
    if "enum" in rule and value not in rule["enum"]:
        return "Invalid node enum"
    if kind == "object":
        props = rule["properties"]
        if any(k not in value for k in rule["required"]) or any(k not in props for k in value):
            return "Missing or extra node keys"
        for key, child in value.items():
            problem = schema_issue(child, props[key], root, depth + 1)
            if problem:
                return problem
    if kind == "array":
        for child in value:
            problem = schema_issue(child, rule["items"], root, depth + 1)
            if problem:
                return problem
    if kind == "string" and len(value) > rule.get("maxLength", 10**9):
        return "Text exceeds schema limit"
    return ""


def serialize_tree(tree: dict[str, Any], depth: int = 0) -> str:
    if depth > 32:
        raise FactorError("Tree exceeds serializer safety depth")
    if "field" in tree:
        return str(tree["field"])
    if "const" in tree:
        return repr(tree["const"])
    args = [serialize_tree(x, depth + 1) for x in tree["args"]]
    if "op" in tree:
        op = tree["op"]
        if len(args) != (1 if op == "neg" else 2):
            raise FactorError("Operator argument count mismatch")
        return f"(-{args[0]})" if op == "neg" else f"({args[0]} {OPS[op]} {args[1]})"
    if "window" in tree:
        args.append(str(tree["window"]))
    return f"{tree['fn']}({', '.join(args)})"


def expression_tree(expression: str) -> dict[str, Any]:
    """Trusted parsed-expression adapter, used by random search and round-trip tests."""

    def walk(node: ast.AST) -> dict[str, Any]:
        if isinstance(node, ast.Name):
            return {"field": node.id}
        if isinstance(node, ast.Constant):
            return {"const": node.value}
        if isinstance(node, ast.UnaryOp):
            return {"op": "neg", "args": [walk(node.operand)]}
        if isinstance(node, ast.BinOp):
            op = {ast.Add: "add", ast.Sub: "sub", ast.Mult: "mul", ast.Div: "div"}[type(node.op)]
            return {"op": op, "args": [walk(node.left), walk(node.right)]}
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            args = node.args[:-1] if node.func.id in TS_FUNCS else node.args
            result: dict[str, Any] = {"fn": node.func.id, "args": [walk(x) for x in args]}
            if node.func.id in TS_FUNCS:
                result["window"] = node.args[-1].value  # type: ignore[attr-defined]
            return result
        raise FactorError("Unsupported expression-tree node")

    return walk(ast.parse(expression, mode="eval").body)
