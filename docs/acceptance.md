# 独立项目MVP验收

日期：2026-11-24。项目：Alpha Evidence Lab。

## 精读证据

固定版本：AgentQuant `830376cbca073225c955c29b7f61f4b19977ebd6`。保存29个完整参考文件快照，约6,089行，原MIT许可证保留。重点跟踪提议、评估、成本、暖机、holdout、记忆、gate及对应测试；未声称完整审计上游全部代码，也未执行其整套测试。精读发现与新实现对应关系见 `AgentQuant_精读与改进.md`。

## 验证结果

环境：Python3.11.18、numpy2.4.6、Pydantic2.14.3。

| 检查 | 结果 |
|---|---|
| ruff check . | 通过；上游参考快照排除，不修改其代码 |
| ruff format --check . | 通过 |
| mypy src | 10个源码文件通过 |
| pytest | 34 passed，无警告 |
| 新运行时覆盖率 | 97.75%，门槛80% |
| pip check | 无依赖冲突 |
| 端到端人工演示 | 6条路径、42批冻结、378次尝试、126个测试结果单元（默认配置；首个 episode 不入主检验） |
| alpha-lab verify | 事件链、SQL状态投影、交付物哈希通过 |

测试覆盖下一开盘与费用手算、初始亏损、完整窗口可得时点、非法输入、目录/预算约束、失败与重复计数、全方法冻结、一次揭晓、SQL/事件篡改、记忆时点/路径隔离、按ID配对、失败结果不填零、独立路径门槛、保护性回撤阈值、holdout扰动不改变搜索，以及CSV/CLI/HTML转义。

## 演示结果的真实含义

默认人工演示中，random与memory方法相对ordered的平均测试收益差为正，但两个Bonferroni校正p值分别为0.087和0.100，均未达到注册的0.05门槛，因此两者均标记 `not_promoted`。这只是合成样本上的工程演示，不能推断真实市场效果。

总体gate的 `evaluated` 表示完成计算，不表示通过门槛。报告保留未晋级结果，没有为获得漂亮结论重新挑种子或改阈值。

## 交付物

运行 `alpha-lab demo --output output/final` 生成以下文件（输出目录不提交到仓库）：

- `output/final/report.html`：最终中文报告。
- `output/final/registration.json`：搜索前的配置/数据/代码指纹。
- `output/final/audit.sqlite`：尝试、冻结、揭晓与事件链。
- `output/final/attempts.csv`、`holdout_results.csv`、`study.json`、`checkpoint.json`。
- `output/report-preview.png`：独立临时Chrome配置渲染并查看的排版截图，表格与文字未发现溢出。

## 未完成的功能

真实LLM提议/成本评测、自动提示词修改、强化学习、国内行情适配、完整A股执行/税费/公司行动、公开预注册与外部签名、实盘、分布式恢复、安全沙箱均不在本版范围。

项目检验研究Agent的选择过程；不宣称收益击败上游，不宣称复现上游完整平台，也不把合成样本当成有效投资策略。

仅供研究，不构成投资建议。
