"""研究证据报告；数据和模型文字统一转义，报告无外部资源依赖。"""

import csv
import hashlib
import html
import json
import platform
from pathlib import Path
from typing import Any

from .models import canonical


def table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p>无记录</p>"
    keys = list(rows[0])
    head = "".join(f"<th>{html.escape(str(key))}</th>" for key in keys)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(str(row.get(key, '')))}</td>" for key in keys) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render(result: dict[str, Any]) -> str:
    escape = html.escape
    summary = []
    for arm in ["ordered", "random", "memory"]:
        rows = [
            row
            for row in result["results"]
            if row["arm"] == arm
            and row["episode"] >= result["registration"]["plan"].get("warmup_episodes", 1)
        ]
        summary.append(
            {
                "搜索方法": arm,
                "配对单元": len(rows),
                "平均测试段收益": f"{sum(row['return'] for row in rows) / len(rows):.2%}",
                "最差测试段回撤": f"{max(row['drawdown'] for row in rows):.2%}",
                "实际尝试数": sum(row["arm"] == arm for row in result["trials"]),
            }
        )
    warmup = []
    for arm in ["ordered", "random", "memory"]:
        rows = [
            r
            for r in result["results"]
            if r["arm"] == arm
            and r["episode"] < result["registration"]["plan"].get("warmup_episodes", 1)
        ]
        if rows:
            warmup.append(
                {
                    "方法": arm,
                    "暖机配对单元": len(rows),
                    "暖机平均收益": f"{sum(r['return'] for r in rows) / len(rows):.2%}",
                }
            )
    comparisons = [
        {
            "方法": row["arm"],
            "相对基线平均差": f"{row['mean_net_return_delta']:.2%}",
            "校正p值": f"{row['p_corrected']:.4f}" if row["p_corrected"] is not None else "未估计",
            "研究状态": row["outcome"],
        }
        for row in result["gate"].get("comparisons", [])
    ]
    attempts = [
        {
            "episode": row["episode"],
            "方法": row["arm"],
            "候选": row["candidate"],
            "状态": row["status"],
            "验证评分": f"{row['score']:.4f}" if row["score"] is not None else "缺失",
        }
        for row in result["trials"][:18]
    ]
    limits = "".join(f"<li>{escape(line)}</li>" for line in result["limitations"])
    all_synthetic = all(item["kind"] == "synthetic" for item in result["data_manifest"])
    badge = "人工行情 · 工程与研究流程验证" if all_synthetic else "用户数据 · 来源与独立性未审计"
    return f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Alpha Evidence Lab</title>
<style>body{{margin:0;background:#f5f5f2;color:#202d39;font:15px/1.7 system-ui,"PingFang SC",sans-serif}}main{{max-width:1100px;margin:auto;padding:45px 30px}}header{{border-top:5px solid #b45309;padding:28px 0}}h1{{font-size:38px;margin:8px 0}}small{{color:#66717d;letter-spacing:2px}}h2{{margin-top:34px}}.badge{{background:#fef3c7;color:#92400e;padding:7px 14px;border-radius:30px;font-size:13px}}section{{padding:22px;background:white;border:1px solid #deded8;border-radius:12px;margin:16px 0;overflow:auto}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:15px}}.card{{background:white;border:1px solid #deded8;padding:20px;border-radius:12px}}strong{{display:block;font-size:28px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:10px;text-align:left;border-bottom:1px solid #e5e7eb;white-space:nowrap}}th{{color:#64748b}}.notice{{border-left:4px solid #b45309;padding-left:15px}}code{{overflow-wrap:anywhere}}footer{{margin-top:35px;color:#64748b;font-size:13px}}@media(max-width:650px){{.cards{{grid-template-columns:1fr}}h1{{font-size:29px}}main{{padding:20px}}}}</style>
<main><header><small>ALPHA EVIDENCE LAB / PRE-REGISTERED SEARCH</small><h1>比“找到好结果”更重要的，<br>是知道它经得起怎样的检验。</h1><span class="badge">{badge}</span><p>研究ID：{escape(result["study_id"])}</p></header>
<p class="notice">{escape(result["registration"]["plan"]["question"])}<br>本项目检验搜索过程；测试收益不是盈利承诺，研究阈值不是实盘授权。</p>
<div class="cards"><div class="card">输入价格路径<strong>{len(result["data_manifest"])}</strong></div><div class="card">每方法每单元尝试预算<strong>{result["registration"]["plan"]["budget"]}</strong></div><div class="card">冻结后批量揭晓<strong>{len(result["batches"])} 批</strong></div></div>
<h2>01 / 同预算，同评估器</h2><section>{table(summary)}<p>各episode净值从1开始，上表是测试段收益的算术平均，不是跨期组合累计收益。</p></section>
<h3>不进入主检验的暖机段</h3><section>{table(warmup)}</section><h2>02 / 配对证据门槛</h2><section><p>{escape(result["gate"]["outcome"])}：{escape(result["gate"]["reason"])}</p>{table(comparisons)}<p>单侧精确符号翻转；Bonferroni覆盖两个预注册比较。主检验排除预注册暖机episode。至少需要6个有效独立簇；5簇的最小校正p值为0.0625。精确性仍依赖簇独立及符号对称。</p></section>
<section><h3>预注册与样本量</h3><pre>{escape(json.dumps({k: result["gate"].get(k) for k in ["cluster_unit", "clusters", "minimum_clusters", "warmup_episodes", "excluded_cells_per_arm", "smallest_possible_corrected_p", "assumption"]}, ensure_ascii=False, indent=2))}</pre></section><h2>03 / 每一次尝试都留下证据</h2><section>{table(attempts)}<p>展示前18次；全部尝试与失败详见 attempts.csv 和 audit.sqlite。预算耗尽不会被标成成功晋级。</p></section>
<h2>04 / 研究边界</h2><section><ul>{limits}</ul><p>后端：确定性搜索策略；没有调用大模型，也没有声称实现自我进化。</p></section>
<footer>仅供研究，不构成投资建议。<br>审计链头：<code>{escape(result["ledger_head"])}</code><br>架构学习来源：<a href="https://github.com/OnePunchMonk/AgentQuant">OnePunchMonk/AgentQuant</a>；新实现与阅读证据见项目文档。</footer></main></html>"""


def export(result: dict[str, Any], output: Path) -> None:
    (output / "report.html").write_text(render(result), encoding="utf-8")
    (output / "study.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    for key, name in [("results", "holdout_results.csv"), ("trials", "attempts.csv")]:
        rows = [
            {k: canonical(v) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()}
            for row in result[key]
        ]
        with (output / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    _checkpoint(result, output)


def _checkpoint(result: dict[str, Any], output: Path) -> None:
    files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(output.iterdir())
        if path.is_file() and path.name != "checkpoint.json"
    }
    checkpoint = {
        "study_id": result["study_id"],
        "ledger_head": result["ledger_head"],
        "python": platform.python_version(),
        "artifact_hashes": files,
    }
    (output / "checkpoint.json").write_text(json.dumps(checkpoint, indent=2), encoding="utf-8")


def export_factor(result: dict[str, Any], output: Path) -> None:
    escape = html.escape
    primary = [r for r in result["results"] if r["phase"] == "confirmation"]
    summary = []
    for arm in result["registration"]["arms"]:
        held = [r for r in primary if r["arm"] == arm and r["status"] == "ok"]
        trials = [r for r in result["trials"] if r["arm"] == arm]
        failed = [r for r in trials if r["status"] == "failed"]
        classic = set(result["registration"]["classic_library"])
        from .factor_arms import normalize

        normalized = []
        for row in trials:
            try:
                normalized.append(normalize(row["payload"]["proposal"]["expression"]).expression)
            except ValueError:
                pass
        summary.append(
            {
                "method": arm,
                "confirmation_rounds": len(held),
                "mean_test_return": sum(r["return"] for r in held) / len(held) if held else None,
                "mean_test_ic": (
                    sum(
                        r["metrics"]["ic_mean"] * r["metrics"]["ic_observations"]
                        for r in held
                        if r["metrics"].get("ic_mean") is not None
                    )
                    / sum(
                        r["metrics"]["ic_observations"]
                        for r in held
                        if r["metrics"].get("ic_mean") is not None
                    )
                    if any(
                        r["metrics"].get("ic_mean") is not None
                        and r["metrics"].get("ic_observations", 0)
                        for r in held
                    )
                    else None
                ),
                "ic_observations": sum(r["metrics"].get("ic_observations", 0) for r in held),
                "ic_valid_days": sum(r["metrics"].get("ic_valid_days", 0) for r in held),
                "trials": len(trials),
                "failure_rate": len(failed) / len(trials) if trials else None,
                "duplicate_rate": sum("Duplicate" in r["payload"].get("error", "") for r in trials)
                / len(trials)
                if trials
                else None,
                "library_expression_equivalence": sum(
                    x in {normalize(c).expression for c in classic} for x in normalized
                )
                / len(normalized)
                if normalized
                else None,
            }
        )
    result["summary"] = summary
    limits = "".join("<li>" + escape(x) + "</li>" for x in result["limitations"])
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Factor study</title><style>body{max-width:1100px;margin:35px auto;font:15px/1.6 system-ui}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:7px}pre{white-space:pre-wrap}</style>'
    page += (
        "<h1>Price-volume factor study</h1><p>"
        + escape(result["registration"]["plan"]["question"])
        + "</p>"
    )
    page += (
        "<h2>Confirmation: primary result</h2><pre>"
        + escape(canonical(result["gate"]))
        + "</pre>"
        + table(summary)
    )
    page += (
        "<h2>Exploration only</h2><pre>" + escape(canonical(result["exploration_gate"])) + "</pre>"
    )
    page += "<h2>Registered design</h2><pre>" + escape(canonical(result["registration"])) + "</pre>"
    page += "<h2>Model usage</h2><pre>" + escape(canonical(result["llm_usage"])) + "</pre>"
    page += (
        "<h2>Limitations</h2><ul>"
        + limits
        + "</ul><p>Research diagnostics only; not investment advice.</p></html>"
    )
    (output / "report.html").write_text(page)
    (output / "study.json").write_text(canonical(result))
    for key, name in [("results", "holdout_results.csv"), ("trials", "attempts.csv")]:
        rows = [
            {k: canonical(v) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()}
            for row in result[key]
        ]
        with (output / name).open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    _checkpoint(result, output)
