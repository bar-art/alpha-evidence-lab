# Alpha Evidence Lab

[![tests](https://github.com/bar-art/alpha-evidence-lab/actions/workflows/tests.yml/badge.svg)](https://github.com/bar-art/alpha-evidence-lab/actions/workflows/tests.yml)

**研究 Agent 真的比简单方法更会挑实验吗？**

让 AI 反复尝试策略，总能找到一个回测好看的结果，但这往往只是试的次数多。这个项目给每种搜索方法相同的尝试次数，把测试数据封存到所有方法都做完选择之后才一起揭晓，再用配对统计检验判断差异是否显著。每一次尝试，包括失败的，都记在可校验的审计记录里。

项目在精读 [OnePunchMonk/AgentQuant](https://github.com/OnePunchMonk/AgentQuant) 的源码后独立实现。阅读笔记和改进对照见 [docs/AgentQuant_精读与改进.md](docs/AgentQuant_精读与改进.md)。

## 结论

两项研究都用沪深300的真实数据，结论都是**没有一种搜索方法显著好于最简单的基线**，其中也包括大模型。

| 研究 | 样本 | 对比基线 | 结果 |
|---|---|---|---|
| 双均线策略搜索 | 2014 年底权重前 10 的股票，2015–2026，31 个时间块 | 按目录顺序选 | 随机 −0.09%、历史记忆 −0.08%、大模型 −0.30%，校正后 p 均为 1.0 |
| 横截面因子搜索 | 沪深300历史成分（668 只，含退市），模型发布后的 11 个确认时间块 | 10 个经典因子 | 大模型 −0.12 个百分点，校正后 p = 1.0；随机 +1.21 个百分点，仅作参考 |

大模型（gpt-5-nano）的行为本身也值得一看：

- 在双均线研究中，18% 的尝试是在重复本批次已经试过的候选；
- 在因子研究中，605 次提议有 40% 无效，其中 198 次是同一轮内的重复，36 次是函数参数个数写错。

两项研究的大模型费用合计约 0.25 美元。

## 怎样防止"看起来有效"

1. **预注册：** 研究问题、候选范围、尝试预算、成本、统计方法和数据指纹在搜索前写入 `registration.json`。
2. **相同预算：** 每种方法每轮尝试次数相同；失败和重复也算一次，没有免费重试。
3. **封存测试集：** 所有方法冻结选择之后，测试结果才一次性全部揭晓，不能挑着公布。
4. **统一评估：** 搜索、验证、测试使用同一个评估器：收盘出信号，次日开盘成交，计入成本。
5. **配对检验：** 按时间块配对比较各方法与基线，用符号翻转检验并做 Bonferroni 校正。
6. **可审计：** 每次尝试、冻结和揭晓都写入 SQLite 哈希链；`alpha-lab verify` 可以核对记录和交付文件是否被改动。

## 两种研究

### 双均线策略搜索（`research`）

候选是 6 个预先登记的长仓 / 空仓双均线策略。参与比较的方法有：

- `ordered`：按目录顺序取前 K 个，作为基线；
- `random`：用固定种子随机取 K 个；
- `memory`：按此前轮次的验证表现排序；
- `llm`（可选）：大模型根据匿名化的历史验证得分逐次提议。

### 横截面因子搜索（`factor-study`）

候选是用一套受限表达式语言写出的价量因子，例如 `cs_rank(-pct_change(close, 5))`。表达式只能使用白名单里的字段、函数和窗口长度，由自己写的解析器执行，从不运行模型生成的代码。

参与比较的方法：

- `library`：从 10 个公开经典因子（反转、低波动、动量、成交量等）中按固定顺序取前 5 个，作为基线；
- `random`：在同一套语言中随机生成；
- `llm`：大模型以表达式树的形式提出因子，并附上经济学假设和失效条件。

随机生成和大模型共用同样的限制：最大深度 4、最多 25 个节点、最长回看 250 天。

每种方法每轮提交 5 个因子。每一轮先在 120 天的开发段里按 ICIR 选出最佳因子，再在随后 20 天的测试段里做多排名前 20%、做空后 20% 的股票，按扣除成本后的段收益比较。为贴近 A 股的交易限制：

- 开盘涨停买不进；跌停或停牌卖不出时，顺延到第一个能卖的开盘，最多顺延 10 天，亏损照常计入；
- 测试段之后留 15 天结算期，只用来平掉测试段里建立的持仓，不开新仓、不参与打分；
- 开发段与测试段之间隔开 15 天，开发段末尾持仓的退出不会用到测试段的价格。

主比较只用模型发布日期（2025-08-07）之后的 11 个确认时间块；更早的 110 个时间块属于探索段，单独报告。两轮试跑揭晓过的时间块已排除。

## 快速开始

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install -e .

alpha-lab demo --output output/demo        # 人工数据，离线运行
open output/demo/report.html
alpha-lab verify --output output/demo      # 核对审计记录与文件哈希
```

每次运行都要用新的输出目录，避免覆盖已经揭晓的实验。运行依赖只有 numpy 和 pydantic。

## 使用真实数据

**单资产价格路径（双均线研究）。** CSV 需要包含 `date, open, close, available_at`，时间戳必须带时区。A 股数据还应包含 `is_suspended, up_limit, down_limit, price_limit_exempt`：停牌和开盘涨跌停时不会成交，涨跌停未知且未豁免时拒绝成交。每个文件是一条路径。

```bash
alpha-lab research --input examples/synthetic_prices.csv \
  --source '人工样例' --config examples/small_plan.json --output output/csv-example
```

**多股票面板（因子研究）。** 使用 [ashare-data-pipeline](https://github.com/bar-art/ashare-data-pipeline) 导出的 `bars.csv` 和 `manifest.json`，其中包含逐日的历史成分、停牌和涨跌停信息：

```bash
alpha-lab factor-study --bars <导出目录>/bars.csv --manifest <导出目录>/manifest.json \
  --config config/factor_hs300_v3.json --output output/factor-control
```

A 股印花税可以按日期分段设置：`config/ashare.json` 中，2008-09-19 起卖出 10bps，2023-08-28 起 5bps。

三个项目并列放在同一目录时，可以用 [quant-research-lab](https://github.com/bar-art/quant-research-lab) 的 `scripts/run_open_data.sh` 生成带哈希校验的共同数据包，再用 `alpha-lab research --bundle <目录>` 读取。

## 大模型方法（可选，需要 OpenRouter）

```bash
set -a; . ./.env; set +a        # .env 中写 OPENROUTER_API_KEY，不要提交
alpha-lab factor-study --bars ... --manifest ... \
  --config config/factor_hs300_llm_v3.json --output output/factor-llm
```

配置里写明模型、发布日期及来源、服务商端点、推理强度、调用次数和费用上限，这些都会进入预注册。运行时的规则如下：

- **匿名输入：** 模型只看到候选编号、窗口参数和匿名化的历史开发得分，看不到证券代码、日期或任何测试结果。
- **严格格式：** 候选、证据编号和表达式树的节点都由 JSON schema 约束，模型无法引用不存在的证据。
- **预算规则：** 格式错误、空回复或服务商过滤器误判会让模型重答，最多两次；重答只计入调用次数和费用，不占研究预算。格式正确但无效的提议（重复、参数错误、覆盖率不足）算一次失败的尝试。
- **费用硬上限：** 每次请求前按最坏情况预留费用，预留后会超过上限就停止发送。服务商返回的费用缺失时不当成 0。
- **可复现：** 合规的回复保存在缓存里；缓存放在输出目录之外时，可以在新目录中零费用重跑出相同结果。

## 输出文件

| 文件 | 内容 |
|---|---|
| `registration.json` | 搜索前固定的问题、参数、比较方式、数据与代码指纹 |
| `audit.sqlite` | 全部尝试、失败、冻结和揭晓的哈希链 |
| `attempts.csv` | 每次提议及其开发 / 验证结果、失败原因 |
| `holdout_results.csv` | 各方法在每个测试段的结果 |
| `study.json` | 完整研究记录和统计结果 |
| `checkpoint.json` | 事件链末端和交付文件哈希，供 `verify` 使用 |
| `report.html` | 离线可打开的研究报告 |

## 统计检验

比较的单位是**时间块**：同一时期的多只股票高度相关，不能各算一个独立样本。

- 差值不为 0 的时间块不超过 20 个时，穷举全部符号组合，做精确符号翻转检验；
- 更多时用固定种子的蒙特卡洛符号翻转（99,999 次），p 值按 (1 + 超过次数) / (1 + 抽样数) 计算，不会为 0；
- 多个方法与基线比较时做 Bonferroni 校正；
- 至少需要 6 个时间块，否则无法达到 0.05 的显著性门槛。

## 局限

- **样本短：** 因子研究的确认段只有 11 个时间块，只能检出较大的差异。结论是"没有发现显著的增量"，不等于"证明没有差异"。
- **前视风险没有完全排除：** 确认段放在模型发布之后，但模型可能读过公开的因子研究，知道哪些因子历史上有效。研究设计在登记前看过部分历史结果，这些都已写入预注册。
- **统计假设：** 符号翻转检验假设各时间块相互独立、差值的符号对称。相邻时间块的结算期有重叠，可能共享市场冲击。
- **交易简化：** 允许碎股、按比例收取成本；多空组合没有模拟 A 股的融券限制、借券费和保证金；未能卖出的持仓按收盘价估值，不等于真正变现。
- **数据范围：** 只有价量数据，没有时点准确的基本面；每段净值从 1 重新计算，平均段收益不是累计或年化收益。
- **只测了一个模型和一种推理设置，** 结论不能推广到其他模型。
- **本地预注册和哈希链不是第三方认证。**

## 开发

```bash
sh scripts/verify.sh        # ruff、mypy、pytest（离线，覆盖率门槛 80%）
```

设计说明见 [docs/architecture.md](docs/architecture.md)，验收记录见 [docs/acceptance.md](docs/acceptance.md)，因子研究的完整结果见 [docs/factor_v3_paid_run_summary.json](docs/factor_v3_paid_run_summary.json)。`references/agentquant/` 保存了上游项目的源码快照，只供阅读，不被导入或执行，并保留其 MIT 许可。

仅供研究，不构成投资建议。
