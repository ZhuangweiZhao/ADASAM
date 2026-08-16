# Research Log

## 2026-08-16 - LoveDA benchmark context recorded

Date: 2026-08-16

Question: How do unified LoveDA results compare with the supplied literature table?

Experiment: Recorded eight supplied literature rows and compared them with four
internal 512 x 512, 100%-label, seed-42 baselines.

Observation: MobileSAM full finetune reaches 52.35% mIoU, approximately matching
UNetFormer (52.40%) and Hi-ResNet (52.50%), while trailing the supplied maximum
(U-Net MaxViT-S, 56.16%) by 3.81 points. It leads the supplied rows on background
and barren IoU but is notably weak on water and forest.

Decision: Use the table only as external context until citations and protocols are
verified. Keep internal results at screening status until three seeds are complete.

Evidence Link: `docs/LOVEDA_BENCHMARK_COMPARISON.md`, `EXPERIMENT_MANIFEST.md`

只记录可复用的研究事实，不记录长篇讨论。每天使用以下模板：

```text
Date:
Question:
Experiment:
Observation:
Decision:
Evidence Link:
```

## 2026-08-14 — Evidence registration + Vaihingen survival completion

Date: 2026-08-14  
Question: 已有云端/本地运行结果中，哪些具备可登记的证据包？  
Experiment: 汇总 `runs/云服务器`（150 runs + 36 aggregates，`analysis/export_cloud_results_summary.py`）；核对 NEU-Seg 108-run 3 种子矩阵、LoveDA 语义预算/保留率/layer 组、Vaihingen 自适应保留 3 种子、iSAID 单 run；更新 `EXPERIMENT_MANIFEST.md`。  
Observation: NEU-Seg 矩阵完备（basic 增强是最大单因素，@1% +4.8~12.9pt）；LoveDA 语义预算 K=3>2>1 单调但单种子；magnitude 保留率 0.25–1.00 精度与 FPS 均无差异（零结果）；Vaihingen adaptive 3 种子在 mIoU（+0.9~1.2pt）与 Small IoU（+0.020~0.029）一致占优，seed42 仅因 0.08pt 未过 1pt 闸门而判 REDESIGN。  
Decision: 全部登记为 screening（缺存档 checkpoint / 多种子 / 效率协议记录，不得作为论文结果引用）；提交 Vaihingen/LoRA/自适应代码；移除 `thirdparty/SAM-RSP` gitlink 并加入 gitignore。  
Evidence Link: `EXPERIMENT_MANIFEST.md`, `runs/云服务器/云服务器结果汇总.csv`, `runs/vaihingen_adaptive_survival*/survival_summary.json`

## 2026-08-10 — Documentation consolidation

Date: 2026-08-10  
Question: 当前文档是否准确反映 8 月以来的主线实现？  
Experiment: 对照当前代码、训练入口和 `84dfdef..e3b5171` 提交范围，审查 README、方法设计、技术文档和实验清单。  
Observation: 根 README 与实验清单仍主要描述早期实例分割/V3 路线；当前主线已是 NEU-Seg、LoveDA、iSAID 上的标签高效语义分割，并已实现多尺度适配、prompt/prototype、边界分支和 SCSR 系列实验。仓库尚未登记完整的 paper-ready 多种子证据包。  
Decision: 以标签高效语义分割为当前入口；旧 few-shot/instance 内容保留为历史资料；所有收益继续标记为待验证；新增统一文档索引和项目进展总结。  
Evidence Link: `README.md`, `docs/PROJECT_UPDATE_2026-08-10.md`, `EXPERIMENT_MANIFEST.md`, commit `e3b5171`

## 2026-07-27

Date: 2026-07-27  
Question: Phase 1 的下一项可交付成果是什么？  
Experiment: 冻结 Research Plan v1.0，建立 Literature Landscape v0.1 和 Phase 1 产物模板。  
Observation: 管理文档已足够，下一步应转向文献、baseline、protocol 和 evidence。  
Decision: 开始 L01，暂停新模型设计。  
Evidence Link: `research_issues.csv`

## 2026-07-27 — L01 Preliminary Result

Question: Prototype family 已经解决了哪些 support representation 问题？  
Experiment: 第一轮 Prototype family literature mapping。  
Observation: Prototype、Multi-Prototype、Memory、Pixel Matching 已有成熟或相邻路线；目前材料尚未显示 duplicate support 和 support noise 被系统作为主要研究对象。  
Decision: 不将“尚未发现”写成“现有方法没有”；L01 标记为 Partial，下一步优先完成 L02 Multi-Prototype landscape。  
Evidence Link: `docs/literature_landscape.md`, `representation_matrix.csv`, `evidence_chain.csv`

## 2026-07-27 — L02 Scope Revision

Question: Multi-Prototype 是否已经隐式实现 Information Weighting？  
Experiment: 将 L02 从 literature summary 改为 information-flow equivalence analysis。  
Observation: 当前尚无足够证据区分 prototype allocation、prototype selection 与 information weighting。  
Decision: 先比较 Selection Target 和显式优化目标；加入 K02 Novelty Collapse；SIE 继续禁止实现。  
Evidence Link: `docs/literature_landscape.md`, `evidence_chain.csv`, `research_issues.csv`

## 2026-07-27 — L03 First-Round Finding

Question: Memory family 是否已经覆盖 Information Weighting？  
Experiment: 分析 Prototype Memory、Feature Memory、Style Memory、Online Memory 和 MM-Net。  
Observation: Memory 通常写入 prototype、feature 或 style，读取依赖 similarity、attention 或 addressing；当前材料未显示显式 Information Filtering Objective。MM-Net 的 support quality weighting 是与未来 SIE 最接近的边界案例。  
Decision: L03 标记 Partial；K02 保持 Open；新增 L04 Objective Analysis，重点区分 SIE 与 quality-weighted prototype。  
Evidence Link: `docs/literature_landscape.md`, `evidence_chain.csv`, `research_issues.csv`

## 2026-07-27 — L02 Round-2 Finding

Question: Multi-Prototype 是否已经隐式完成 Information Weighting？  
Experiment: 分析 ASGNet、RPGM、TPSN、ProtoFormer 的 selection target、weighting 和 optimization objective。  
Observation: 这些方法主要围绕 prototype 构建、分配、匹配或 attention；当前材料尚未发现独立的 Information Selection Objective、duplicate-aware optimization 或 noise-aware optimization。  
Decision: K02 不触发，状态保持 Open/Insufficient；继续核对优化目标和训练信号，重点防止 SIE 退化为 prototype selection。  
Evidence Link: `docs/literature_landscape.md`, `evidence_chain.csv`

## 2026-07-27 — L02 Round-3 Finding

Question: Multi-Prototype 与未来 SIE 的边界是否已经塌缩？  
Experiment: 继续分析 prototype allocation、prototype weighting、support noise 相关工作和 duplicate support 检索结果。  
Observation: Pattern diversity 已是成熟方向；weighting 已普遍存在，但主要服务 prototype matching/allocation。Support noise 存在相关但不同的半监督/伪标签/不确定性工作；Duplicate Support 仍未发现直接 FSS 研究。  
Decision: K02 继续保持 Open；L02 标记 Partial，下一步进入 L03 Memory equivalence analysis。  
Evidence Link: `docs/literature_landscape.md`, `evidence_chain.csv`, `research_issues.csv`
