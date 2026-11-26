"""离线可复现CLI；新实验使用新目录，不自动覆盖旧结果。"""

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path

from .bundle import load_bundle
from .data import load_csv
from .ledger import Ledger
from .models import StudyPlan, digest
from .report import export
from .study import run_study


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Alpha Evidence Lab；仅供研究，不构成投资建议")
    parser.add_argument("command", choices=["demo", "research", "verify", "factor-study"])
    parser.add_argument("--output", type=Path, default=Path("output/demo"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--bars", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--input", type=Path, nargs="+")
    parser.add_argument("--bundle", type=Path, help="Hash-checked research_bundle_v1 directory")
    parser.add_argument("--source", help="用户数据来源说明")
    parser.add_argument(
        "--llm-cache", type=Path, help="大模型响应缓存（JSONL）；放在输出目录外可免费重跑"
    )
    args = parser.parse_args(argv)
    print("仅供研究，不构成投资建议")
    try:
        if args.command == "verify":
            checkpoint = json.loads((args.output / "checkpoint.json").read_text())
            registration = json.loads((args.output / "registration.json").read_text())
            if not (args.output / "audit.sqlite").is_file():
                raise ValueError("审计数据库不存在，校验不会重建它")
            ledger = Ledger(
                args.output / "audit.sqlite",
                digest(registration["plan"]),
                registration["data_hash"],
                registration["code_hash"],
                budget=registration["plan"]["budget"],
                arms=tuple(registration["arms"]),
            )
            ledger.verify(checkpoint["ledger_head"])
            for name, expected in checkpoint["artifact_hashes"].items():
                if hashlib.sha256((args.output / name).read_bytes()).hexdigest() != expected:
                    raise ValueError(f"交付物哈希不一致：{name}")
            print("事件链、状态投影与文件哈希校验通过；本地检查点不是第三方认证。")
            return 0
        if args.command == "factor-study":
            from .factor_study import run_factor_study
            from .models import FactorPlan
            from .panel import load_panel
            from .report import export_factor

            if args.bars is None or args.manifest is None:
                raise ValueError("factor-study requires --bars and --manifest")
            factor_plan = (
                FactorPlan.model_validate_json(args.config.read_text())
                if args.config
                else FactorPlan()
            )
            result = run_factor_study(
                factor_plan,
                load_panel(args.bars, args.manifest),
                args.output,
                llm_cache=args.llm_cache,
            )
            export_factor(result, args.output)
            print(
                f"Factor study: {len(result['batches'])} rounds; "
                f"confirmation gate {result['gate']['outcome']}"
            )
            return 0
        plan = (
            StudyPlan.model_validate_json(args.config.read_text()) if args.config else StudyPlan()
        )
        paths = None
        if args.command == "research":
            if args.bundle:
                if args.input or args.source:
                    raise ValueError("--bundle cannot be combined with --input or --source")
                paths = load_bundle(args.bundle)
            else:
                if not args.input or not args.source:
                    raise ValueError("research 需要 --input CSV列表和 --source 来源说明")
                paths = tuple(load_csv(path, path.stem, args.source) for path in args.input)
        result = run_study(plan, args.output, paths, llm_cache=args.llm_cache)
        export(result, args.output)
        print(
            f"完成 {len(result['trials'])} 次尝试、{len(result['batches'])} 批冻结；"
            f"研究门槛：{result['gate']['outcome']}；报告 {args.output / 'report.html'}"
        )
        if result["llm_usage"]:
            usage = result["llm_usage"]
            print(
                f"大模型 {usage['model']}：API 调用 {usage['api_calls']} 次、缓存命中 "
                f"{usage['cache_hits']} 次，费用 ${usage['cost_usd']:.4f}"
            )
        return 0
    except (OSError, ValueError, KeyError, sqlite3.Error) as error:
        print(f"停止：{type(error).__name__}：{str(error)[:250]}")
        print("请检查配置、时间戳、输入长度与新输出目录；不会取消封存或自动替换数据。")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
