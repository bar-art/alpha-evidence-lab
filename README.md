# Alpha Evidence Lab

**给每一次策略尝试留下证据，让搜索方法接受相同预算的检验。**

这是根据 [OnePunchMonk/AgentQuant](https://github.com/OnePunchMonk/AgentQuant) 核心源码精读后独立实现的新项目。它关注“研究Agent有没有更有效地选择实验”，与已有 `quant-research-lab` 的委员会报告工作流不同。

核心过程：**预注册 → 分析/提议/统一评估/反思 → 全方法冻结 → 测试集批量揭晓 → 配对证据门槛 → 可核查报告**。

[源码精读与改进证据](docs/AgentQuant_精读与改进.md) · [架构与边界](docs/architecture.md) · [验收记录](docs/acceptance.md)

## 已实现的改进

- 候选只能引用预注册目录ID；拒绝目录外参数和浮点窗口，不靠提示词约束。
- 搜索、验证、holdout共用同一评估器：前一收盘信号、下一开盘成交、成本同步、初始净值为1。
- 每个方法同样的尝试预算；失败和重复提议都计数，不提供免费重试。
- 三个方法全部完成并冻结选择后，才批量评估holdout；一次揭晓，不允许选择性公布。
- 独立随机数实例、数据/代码/配置绑定；不修改全局环境变量或随机状态。
- 带日期与市场路径作用域的历史验证记忆；holdout不进入搜索记忆。
- 逐 `(数据路径, episode)` 配对，拒绝缺失/重复/失败单元；独立簇精确符号翻转检验与 Bonferroni 方法比较校正；首个 episode 预先排除为暖机段。
- SQLite事务事件链，同时验证试验表和封存状态投影；文件哈希检查点；CSV/JSON/HTML导出。

三个确定性搜索方法是：`ordered`（目录前K个，非全网格穷举）、`random`（固定种子随机K个）、`memory`（此前验证证据排序K个）。配置 `llm` 后增加第四个方法 `llm`：大模型在相同尝试预算下逐次提议候选。不包含提示词自动改写或强化学习训练。

### 大模型搜索方法（可选，OpenRouter）

```bash
export OPENROUTER_API_KEY=...        # 或 set -a; . ./.env; set +a
alpha-lab research --config <含 llm 段的配置> --input ... --source ... \
  --llm-cache output/llm_cache/my-study.jsonl --output output/my-llm-study
```

配置中的 `llm` 段写明模型、发布日期及其来源、服务商端点、推理强度、调用次数与费用上限，全部进入预注册。

- **匿名化：** 模型只看到候选编号与窗口、历史验证得分（证据编号为 e1、e2……）和本批次已做的尝试；不含证券代码和日期。
- **严格格式：** 候选和引用都是枚举，模型只能选目录内的候选、引用输入里真实存在的证据。
- **预算公平：** 每次合规提议都消耗一次尝试（包括重复候选）。格式错误、空回复或服务商过滤器误判属于通信问题，会重试（最多 2 次），只计费用和调用次数，不占研究预算。
- **可复现：** 响应缓存以首次请求为键，只保存合规结果；放在输出目录之外时，可在新目录中零费用重跑出相同结果。

## 直接运行

本机 `.venv` 已配置 Python3.11：

```bash
cd alpha-evidence-lab
source .venv/bin/activate
alpha-lab demo --output output/my-study
open output/my-study/report.html
alpha-lab verify --output output/my-study
```

已生成的演示在 `output/final/report.html`。每次新研究必须使用新目录，防止覆盖已揭晓实验。默认演示完全离线，不需要API密钥，不访问海外行情。

新机器安装：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.lock
python -m pip install -e .
alpha-lab demo --config config/default.json --output output/demo
```

`requirements.in` 固定直接依赖；`requirements.lock` 固定测试/开发/运行的传递依赖。运行依赖只有 numpy、Pydantic及其依赖。

## 用户数据

```bash
alpha-lab research --input examples/synthetic_prices.csv \
  --source '人工样例，仅验证导入' --config examples/small_plan.json \
  --output output/csv-example
```

示例CSV也是人工数据。真实CSV必须显式包含：`date,open,close,available_at`，时间戳必须带时区，完整日线不能标为收盘前已知。仅支持单资产的一个或多个价格路径；每个文件是一条路径，文件名作为路径ID。不要把同一价格路径改文件名或换搜索随机种子，当成独立市场证据。

真实 A 股路径可由 `a-share-quant-agent` 导出（BaoStock 后复权，停牌日移除，每只股票一个文件）：

```bash
cd ../a-share-quant-agent
quant-agent export --provider baostock --codes 600000.SH 000001.SZ 600519.SH 300750.SZ 688981.SH \
  --start 2020-01-02 --end 2024-12-31 --target alpha-evidence-lab --output output/alpha_input
cd ../alpha-evidence-lab
alpha-lab research --input ../a-share-quant-agent/output/alpha_input/*.csv \
  --source 'BaoStock后复权日线；同一市场股票相关，不是独立路径' --output output/ashare
```

同一市场的股票高度相关，证据门槛的"独立路径"假设不成立，结果只作探索参考。实测输出：`output/ashare-2020-2024/`。

至少需要6个独立簇；5簇的最小双重比较校正 p 值为0.0625。真实路径默认按重叠测试时间窗口聚类，同期股票不会分别计作独立样本。非重叠时间簇仍可能存在序列相关；研究者必须论证独立性和零假设下簇差值的符号对称性，均值相等本身不足以保证检验有效。非零簇不超过20个时穷举全部符号组合（精确检验）；更多时改用固定种子的蒙特卡洛符号翻转（99,999次，p = (1 + 超过次数) / (1 + 抽样数)，Phipson & Smyth 2010），有限抽样下仍是有效的保守 p 值。检验方式、抽样数和种子都写入预注册。

## 查看交付物

| 文件 | 内容 |
|---|---|
| `registration.json` | 在搜索前固定的问题、目录、预算、费用、指标、代码与数据指纹 |
| `audit.sqlite` | 全部尝试、失败、冻结和一次性揭晓 |
| `attempts.csv` | 每一次提议、训练/验证结果和错误 |
| `holdout_results.csv` | 所有方法的匹配测试单元 |
| `study.json` | 完整研究记录、记忆来源与统计结果 |
| `checkpoint.json` | 事件链头、交付物哈希、Python版本 |
| `report.html` | 无远程资源的中文研究报告 |

默认6条独立人工路径 × 7个episode × 3种方法 × 每种3次尝试，共378次。首个 episode 单独报告、不进入主检验；暖机排除和聚类方式在运行前登记。测试段收益按episode报告，报告中的平均值不是跨期累计净值。各段从现金重新开始，末日持仓按收盘估值，不强制清仓。

## 验证

```bash
sh scripts/verify.sh
```

执行 ruff、mypy、pytest，覆盖率门槛80%。测试无需网络。单元测试涵盖手算费用/收益、初始亏损、延迟公开数据、目录越界、预算、状态机、篡改、配对覆盖、保护指标、holdout扰动不改变搜索及CLI。

## 适用范围

这是研究流程MVP。统一成本是研究假设；没有完整A股板块规则、手数、停牌、涨跌停、税费历史、除权或实盘交易。目录只包含长仓/空仓双均线策略。统计门槛是探索性研究判断，不代表市场盈利或自主系统已完成自我改进。

本地预注册不是公开时间戳；哈希链不是第三方认证；封存接口不是防止本机操作者读文件的安全沙箱。上游快照保存在 `references/agentquant/`，只供阅读，未导入或执行，保留原MIT许可。

仅供研究，不构成投资建议。

### 统计口径迁移（2026-11-24）

旧版 bootstrap 报告不得沿用为新版门槛通过证据，应重新预注册并运行。精确检验给出单侧 p 值，不再输出旧版 bootstrap 区间。`bootstrap_samples` 为兼容旧配置保留，对新版检验无效。聚类单位、暖机排除数量、可达 p 值和适用假设均写入报告。

### 真实数据执行字段

CSV 增加 `is_suspended,up_limit,down_limit,price_limit_exempt`。保留停牌日用于估值；禁止停牌、开盘涨停买入、开盘跌停卖出。未知涨跌停且未明确豁免时拒绝成交。旧版四列 CSV 仍能读取，但不再默认无限制可成交，应重新导出。人工路径继续使用明确的无限制研究假设。可选 `stamp_duty_schedule` 按成交日对卖出加收分段印花税，`config/ashare.json` 给出 A 股口径（2008-09-19 起 10bps、2023-08-28 起 5bps），早于首个生效日的成交直接拒绝。当前账本仍允许碎股、采用比例成本，不代表完整 A 股实盘交易规则；100股交易单位、最低佣金及公司行动现金流仍待实现。

### 与 quant 共用数据包

支持 `research --bundle ../quant-research-lab/output/tushare_bundle --output output/tushare_study`。导入器验证 `research_bundle_v1` 清单、路径和 CSV 哈希，并将清单哈希纳入数据身份；不得同时传入 `--input` 或 `--source`。同一市场的股票仍按原有时间簇规则计算统计证据，数据包不证明路径独立或历史股票池完整。

当前仅接受 `close_assumed` 可得时间策略；`next_open` 延迟包明确拒绝，不静默改动时间戳。当前默认使用免费 BaoStock 数据，运行相邻 quant 项目的 `bash scripts/run_open_data.sh` 可生成共同数据包并运行两个项目，详见其 `docs/open_data.md`。原 Tushare 接口已按用户选择弃用。2026-11-25 已用真实 BaoStock 缓存完成端到端研究与审计校验；此次 alpha 的 random 和 memory 均未通过增量证据门槛。

### 长样本实测：沪深300权重股 2015-2026（2026-11-26）

数据来自 `a-share-quant-agent` 的 Tushare 本地快照（后复权，保留停牌行）。路径为 2014-12-31 沪深300权重前10，只用起点已知的信息选择；剔除了 2025 年因合并退市、无法形成完整路径的 600837.SH，所以仍有"存续到 2026"的条件。

```bash
alpha-lab research --input ../a-share-quant-agent/output/export_alpha_top10_2015_2026/*.csv \
  --source '...' --config <config/ashare.json 并设 episodes=32> --output output/ashare-top10-2015-2026
```

| 项目 | 结果 |
|---|---|
| 规模 | 10 条路径 × 32 个 episode × 3 种方法 × 3 次尝试 = 2,880 次尝试，6 秒 |
| 聚类 | 按测试时间块 31 簇（首个 episode 暖机排除），超过精确穷举上限，使用蒙特卡洛符号翻转 |
| random vs ordered | 平均净收益差 −0.09%，校正 p = 1.0，未晋级 |
| memory vs ordered | 平均净收益差 −0.08%，校正 p = 1.0，未晋级 |
| 审计 | `alpha-lab verify` 事件链、状态投影与文件哈希校验通过 |

在 11 年、31 个独立时间块上，"历史记忆"和"随机"两种搜索方法都没有比按目录顺序选择更好。修改前这个样本会因簇数超过 20 而直接返回"证据不足"。

#### 加入大模型方法后的同一研究（gpt-5-nano，2026-11-26）

同样的 10 条路径 × 32 个 episode，增加 `llm` 方法（openai/gpt-5-nano，OpenAI flex 端点，推理强度 minimal），共 3,840 次尝试；Bonferroni 家族扩大为 3 个比较。

| 方法 vs ordered（31 个时间块） | 平均净收益差 | 校正 p | 结论 |
|---|---|---|---|
| random | −0.09% | 1.0 | 未晋级 |
| memory | −0.08% | 1.0 | 未晋级 |
| llm | −0.30% | 1.0 | 未晋级 |

- 大模型调用 914 次、缓存命中 55 次，费用 0.068 美元；格式修复 4 次，服务商过滤器误判重发 5 次。
- 960 次大模型尝试中有 176 次（18%）是重复候选：输入里已经列出本批次的尝试，模型仍然再次提议，白白消耗预算。
- 结论：在这个设定下，小模型作为研究 Agent 没有比按目录顺序选择更好。局限：只测了一个模型，推理强度为 minimal，候选目录只有 6 个双均线策略。

过程记录：最初使用 gpt-oss-120b（DeepInfra），它会编造证据引用、偶尔陷入重复生成导致长时间无响应，所以改用 OpenAI 端点严格执行格式的 gpt-5-nano；引用也随之改成枚举约束。所有调整都在揭晓任何测试集之前完成。
