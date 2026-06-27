# GenAI Feature Analysis

这个 repo 整理了目前项目中可以公开保存的代码、aggregate results、图表和文字总结。研究主线分为两个相互连接的部分：

1. **Goal 1: User brainstorming process**  
   研究用户如何使用 GenAI 进行 brainstorming，包括 creative-state labels、creative run length、wide/local search、process transitions、user-process groups 和 topic differences。

2. **Goal 2: AI strategy and user outcomes**  
   研究 AI response strategy 如何影响用户下一步 reaction，以及后续 KE / user diversity、satisfaction 等 outcome。Goal 2 会基于 Goal 1 里得到的 user state 和 next-reaction labels 继续推进。

## Repository structure

```text
shared_methodology/
  definitions/                     Creativity / brainstorming definitions
  classification/                  Shared four-class turn-level classifier
  data_and_privacy.md
  methods_overview.md

goal1_user_brainstorm_process/
  src/                             Candidate extraction, calibration, full analysis
  results/                         Aggregate tables only
  figures/                         Public result figures
  reports/                         Main findings, literature map, qualitative patterns

goal2_ai_strategy_outcomes/
  src/                             Strategy-KE and next-reaction analyses
  results/                         Aggregate result tables
  figures/
  reports/
```

repo 中只保留清理后的脚本、aggregate tables、figures 和 summaries。原始 conversations、逐行预测结果、逐行人工审核文件、模型文件和大型中间数据都不上传 GitHub。

## Shared four-class labels

- `strict_brainstorming`: 用户明确探索多个 ideas/options。
- `single_creative_generation`: 用户要求生成一个 creative output。
- `refinement_or_selection`: 用户选择、比较、修改或深化已有方向。
- `non_creative_task`: 不属于上述 creative activities。

classifier 在 **turn level** 运行；conversation-level 指标由同一 conversation 内的 turn labels 聚合得到。

## Current status

Goal 1 已经整理成稳定版本，主报告在：

[`goal1_user_brainstorm_process/reports/goal1_brainstorm_process_report.md`](goal1_user_brainstorm_process/reports/goal1_brainstorm_process_report.md)

Goal 1 的核心发现：

- 全量数据包含 262,718 个 user turns、56,942 个 conversations，其中 36,018 个 conversations 至少有一个 creative turn。
- Creative runs 通常很短：平均 1.86 turns，中位数 1 turn。
- 在 creative conversations 中，56.8% 是 single creative generation only，32.0% 是 wide exploration without observed local refinement。
- 人工校准的 genuine multiple-option paths 显示：initial wide search 后进入 local search 约 16.2%；initial local search 后继续 local 约 48.1%。
- 三类可解释 user process groups：brief creativity within broader tasks、creative-session dominant、extended iterative process。

Goal 2 是下一阶段，核心连接是：

```text
current user state + AI response strategy -> next user reaction -> KE / satisfaction
```

## Reproducibility

```powershell
pip install -r requirements.txt
$env:GENAI_DATA_ROOT="D:\path\to\organized_project_files"
```

Goal 1 的脚本说明和结果索引见 [`goal1_user_brainstorm_process/README.md`](goal1_user_brainstorm_process/README.md)。  
数据隐私规则见 [`shared_methodology/data_and_privacy.md`](shared_methodology/data_and_privacy.md)。

## Interpretation boundary

classifier outputs 用于 descriptive process analysis；人工校准结果用于解释 semantic wide/local transitions。目前结果描述 associations 和 process sequences，不是 randomized causal effects。
