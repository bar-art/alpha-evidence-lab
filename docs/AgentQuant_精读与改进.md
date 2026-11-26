# AgentQuant 精读与新项目实现记录

精读日期：2026-11-24。

上游：[OnePunchMonk/AgentQuant](https://github.com/OnePunchMonk/AgentQuant)。固定版本：`830376cbca073225c955c29b7f61f4b19977ebd6`。本次保存29个完整文件快照，共约6,089行，逐一可查 `source_manifest.json`。核心评估、搜索、提议、封存、记忆和相关测试作重点阅读；大型图编排/记忆模块沿相关调用链阅读。本记录不是对整个仓库全部文件的安全审计，未执行上游整套测试或复现其完整实验。

## 1. 它究竟在解决什么

AgentQuant 已由简单的交易参数生成器演化为研究代理实验平台。README提出的核心问题是：Agent在相同预算下是否更会选择实验，而不是只多跑了实验。其工程价值主要在研究循环、来源可追溯的记忆、候选记录、可重放artifact和对照实验，而不是某个收益数字。

主路径为 analyze → hypothesize → backtest → reflect → store，另有 holdout_eval。关键组件包括：

| 模块 | 职责 | 本次重点阅读 |
|---|---|---|
| agent_graph | 研究状态、候选循环、退出状态、holdout和记忆接入 | 核心调用链 |
| proposal_generator / parameter_grid | LLM、网格、随机、记忆提议 | 全文件 |
| backtest/runner / metrics | 主评估器、成本、暖机、绩效与bootstrap | 全文件 |
| episode_splits / search_arms / fair_search_benchmark | 不同方法的开发与测试比较 | 核心评估和预算/配对路径 |
| memory/service / pack | 可见时点、证据检索、token预算 | 时间过滤及读取路径 |
| gate/ledger / gate / prospective | 事件链、封存读取、校正检验、前瞻评分 | 全文件 |
| run_manifest / harness_config | 运行证据和实际生效配置 | 全文件 |

## 2. 值得保留的设计

- **明确研究对象**：比较“搜索政策”，而非把所有实验增益归因于模型智能。
- **保存失败与负证据**：不是只把赢家写入记忆。
- **实际生效配置**：`resolve_effective_config` 对未接线的参数报错，比表面支持很多配置更可信。
- **时间受限记忆**：新版 `MemoryService` 的可见性过滤和图检索已有 as-of 机制；不能因为旧 `StrategyMemory.recall` 没有 cutoff 就断言整个项目没有时间隔离。
- **正确区分退出状态**：当前 `agent_graph` 已区分 `passed_quality_gate`、`budget_exhausted`、`no_valid_candidate`、`execution_failed`。较旧 DESIGN 中“预算耗尽后接受最好结果”的描述不应替代当前代码事实。
- **不虚构DSR**：gate对Sharpe晋级明确返回 `dsr-not-implemented`，README也承认完整晋级集成与前瞻验证尚待完成。
- **公平基准有进步**：脚本已按 episode/seed 配对并按episode重采样，没有简单把各方法不同子集的平均值相减。
- **审计与封存不是空话**：上游已有带锁、hash链和一次holdout读取标记。新项目不把这些概念称为自己发明。

## 3. 具体改进点：证据、影响与实现

以下结论限于固定版本和注明路径。静态确认指源码可直接看到该行为，不表示已运行整个上游应用。

### A. 主回测的费用与信号时点不一致

证据：[runner.py](https://github.com/OnePunchMonk/AgentQuant/blob/830376cbca073225c955c29b7f61f4b19977ebd6/src/backtest/runner.py)。`_apply_transaction_costs` 用 `signal.diff()`，策略收益用 `daily_ret * signal.shift(1)`。因此信号在t变化时费用在t扣，收益头寸却到t+1才变化。成本所属日期影响净值路径、最大回撤与分段比较。这是成本时序问题，不等同于已经证明存在未来收益作弊。

新实现：`evaluator.evaluate` 显式取截至前一收盘的完整滚动窗口，在下一开盘买卖并同步计费；当日收盘估值。`test_next_open_and_fee_hand_calculation`、`test_sell_cost_and_overnight_gap` 逐数核对。

### B. 初始净值没有作为独立观测保留

证据：[search_arms.py L183–198](https://github.com/OnePunchMonk/AgentQuant/blob/830376cbca073225c955c29b7f61f4b19977ebd6/src/agent/search_arms.py#L183) 的 buy-and-hold：先扣首期成本，再累乘生成equity；[metrics.py](https://github.com/OnePunchMonk/AgentQuant/blob/830376cbca073225c955c29b7f61f4b19977ebd6/src/backtest/metrics.py) 的 total_return 用 `eq[-1]/eq[0]-1`。若首个equity已经扣费，这个比值会把首期扣费排除在总收益之外；基于首个观测起峰的drawdown也可能遗漏初始下跌。

新实现：每段净值显式以1开头，首笔费用及首日亏损都计入总收益和回撤。`test_initial_loss_is_included_in_drawdown_and_return` 验证第一天亏50%时报告收益-50%、回撤50%。

### C. 主评估路径与基准路径的暖机不同

证据：`search_arms._momentum_signal_returns` 使用 `rolling(..., min_periods=1)`；`_grade_on_holdout` 只传入切出的holdout片段。主runner则接收完整历史并可用 WarmupEnforcer 检查eval_start之前的长度。两条路径不是完全相同的实验口径；短holdout还可能在慢窗口未积满时产生不同信号。

新实现：训练、验证、holdout都用同一评估器；所有候选采用共同最大窗口暖机；测试前可以读取过去价格暖机，但不带入此前持仓/收益。历史长度不够时失败，不悄悄缩短窗口。

### D. “必须在网格内”没有在所有提议路径强制

证据：[proposal_generator.py](https://github.com/OnePunchMonk/AgentQuant/blob/830376cbca073225c955c29b7f61f4b19977ebd6/src/agent/proposal_generator.py)。`ProposalValidator.validate` 对动量验证正整数和 fast < slow，且用int转换；`_llm_generate` 追加验证结果时没有canonical grid membership检查。相对地，记忆seed路径另有grid_keys检查。提示词写“必须从列表选”不能替代运行时验证。

新实现：提议只引用注册目录ID，再由目录取出参数；窗口严格为整数，未知ID与重复提议记为失败，并消耗预算。`test_canonical_catalog_rejects_unsupported_params`、`test_failed_and_duplicate_attempts_consume_budget` 验证。

### E. 随机种子与全局状态不统一

证据：`PerformanceMetrics.bootstrap_sharpe` 创建未传seed的 `np.random.default_rng()`；`search_arms._run_agent_offline` 则设置 `np.random.seed` 与 `random.seed`。全局seed不控制新建且未指定seed的Generator。该offline函数还会删除调用进程中的API环境变量，finally恢复配置但没有还原这些环境变量。

新实现：所有随机抽样使用显式种子的局部 `Random` / `Generator`；不读写密钥、不更改全局环境变量。配置、代码、数据摘要一同绑定实验。`test_cluster_unit_is_market_path_not_episode_or_search_seed` 检查输入行顺序改变时结果不变。

### F. 缺失拆分时降级为样本内研究需要更醒目的模式

证据：`agent_graph._split_search_and_holdout` 在非法比例、主资产缺失、样本太短时返回原数据和None；对应tests也把这种兼容行为写成预期。对演示可用性有帮助，但研究模式如果没有单独检查，可能失去原本期望的最终holdout。

新实现：训练/验证/holdout长度为显式注册配置；数据不足立即停止，不给不完整研究加“完成验证”标签。`test_dataset_integrity_and_fail_closed_split` 验证。

### G. 需要把检验向量的配对键带进门槛本身

证据：上游 `gate.Gate.evaluate` 按scores数组位置配对，主要检查长度；与公平基准中显式 episode/seed 配对的做法不同。若调用者给出顺序不同但等长的数组，gate本身没有seed/episode键可核对。这是接口契约缺口，不表示当前所有调用都错配。

新实现：每一条结果携带方法、路径、episode键；重复、缺失或失败单元导致证据不足，不按零收益填补，也不静默删去。统计聚类单位是一整条市场路径，避免把同一行情的搜索随机性当作额外市场样本。

### H. 同一评估预算应由统一状态机执行

证据：上游 random arm 默认k=8，grid arm遍历整个目录，Agent循环由迭代与提议数控制。记录预算是必要步骤，但比较时仍要说明实际评估数是否相等，不能仅凭方法名称作公平性结论。

新实现：三个方法都由同一个工具和账本强制每episode尝试K次；失败和重复不返还额度；冻结时再次检查实际次数。没有把目录前K叫成全网格穷举。

### I. 严格输入优于启发式猜列

证据：runner的 `_get_close` 在找不到close别名时会采用第一个能转换为数值的列；这适合容错，但可能把别的字段用于回测。上游 `user_ingest` 已提供更严格路径，应区分直接runner调用与受验证导入。

新实现：本版CSV固定要求date/open/close/available_at；非法价格、重复日期、无时区、延迟公开数据均停止。没有“首个数字列就是价格”的回退。

## 4. 新项目为什么不是改个名字

Alpha Evidence Lab不复刻上游的全部UI、web research、工具编排或记忆图。它实现一条更小且一致的实验链：

1. 搜索前注册问题、参数目录、拆分、预算、成本、主指标和代码/数据摘要。
2. 三种可解释的离线搜索策略在相同开发数据与预算上选择候选。
3. 三种方法一起冻结后，才批量揭晓holdout。
4. 预注册净收益主指标，按市场路径配对重采样，校正两个方法比较。
5. 保存全部尝试、失败和历史验证证据；检查事件链与SQL状态投影。
6. 导出中文HTML、CSV、完整JSON及文件哈希检查点。

这是工程与研究协议改进，不是已测得比上游收益更高。我们没有做两个完整平台的同数据收益对决，也没有宣称实现了上游仍在研究的自主自我改进。

## 5. 统计与安全边界

- 上游已有maxT形式门槛；初版使用配对路径簇 bootstrap 加 Bonferroni；2026-11-24 修订为精确簇符号翻转加 Bonferroni，明确独立与符号对称假设，不宣称统计上普遍优于 maxT。
- 本版主指标是净收益，不冒充DSR；夏普仅为描述指标。
- 默认人工数据支持工程测试，不支持市场有效性结论。
- 五条路径是最低工程门槛，不是任何真实数据都足够的普遍统计保证；真实路径独立性须另证。
- 本地预注册没有可信第三方时间戳；事件链无法防止拥有全部文件权限的人重写整个记录。
- 提议工具没有holdout引用，但这不是针对本地恶意Python代码的安全沙箱。
- 运行时为新写代码；上游MIT快照只供阅读，保留原作者许可，未导入或执行。

## 6. 下一步应怎样验证真实价值

先固定真实数据供应商、单资产执行假设和时间区间；加入充足且有独立性依据的测试路径；在不反复查看最终结果的前提下比较无记忆、真实记忆和打乱记忆政策。若接LLM，需要冻结模型版本与候选协议，记录实际调用，并对等评估预算和模型成本分别作比较。只有前瞻或严格无污染的实验才能支持“模型提升研究质量”的结论。
