# SDK 真实数据验收（2026-11-26）

已通过相邻 quant 项目的可选 Tushare SDK 适配器生成 research_bundle_v1，并使用原有哈希导入、时间簇精确检验和审计流程运行。alpha 不保存卡号，也不需要安装 Tushare。

实际输入：平安银行2020–2024年1,212条日线，包含经校验的复权因子及涨跌停价格。原始未复权 OHLC 与 BaoStock 的1,212个日期键全部一致。数据包位于 `../quant-research-lab/output/sdk_bundle_20261126`。

实际输出：`output/sdk_alpha_20261126/report.html`。共63次尝试、7批冻结，事件链、状态投影及交付物哈希校验通过。主检验采用6个时间簇，首个 episode 排除为暖机段；random 和 memory 都返回 insufficient_evidence，不作为有效策略或增量优势的证据。

免费 BaoStock 流程保持默认。SDK 数据归档、缓存、可选依赖及复现命令见相邻 quant 项目的 `docs/tushare_sdk.md`。数据仍是单资产样本，历史股票池和公司行动完整性没有得到认证。
