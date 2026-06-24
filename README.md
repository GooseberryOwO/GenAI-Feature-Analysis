# GenAI Feature Analysis

本仓库整理了两个相互连接的研究目标：

1. **Goal 1: User brainstorming process**
   用户如何使用 GenAI 进行 brainstorming，包括 creative activity 的类型、持续时间、wide/local search、过程转换和 topic differences。
2. **Goal 2: AI strategy and user outcomes**
   AI response strategy 如何关联用户下一步 reaction，以及后续 KE、satisfaction 等 conversation outcomes。

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

各文件只保留一个正式版本。原始 conversations、逐行预测数据、人工审核原文、模型文件和大型中间数据不进入 GitHub。

## Shared four-class labels

- `strict_brainstorming`: 明确探索多个 ideas/options。
- `single_creative_generation`: 只要求一个 creative output。
- `refinement_or_selection`: 选择、比较、修改或深化已有方向。
- `non_creative_task`: 不属于上述 creative activity。

Classifier 在 **turn level** 运行；conversation-level 指标由同一 conversation 内的 turn labels 聚合得到。

## Current findings

- Four-class classifier: 5-fold CV accuracy **83.2%**；holdout accuracy **81.5%**。
- 全量 Goal 1：262,718 user turns、56,942 conversations；36,018 conversations 至少含一个 creative turn。
- Creative runs 通常较短：平均 **1.86 turns**，中位数 **1 turn**。
- Creative conversations 中：
  - single creative generation only: **56.8%**
  - wide exploration without observed local refinement: **32.0%**
  - extended iterative process group: **13.1%**
- 人工校准的 genuine multiple-option paths 显示：
  - initial wide search 后转入 local search: **16.2%**
  - initial local search 后继续 local: **48.1%**
- Goal 2 的 preliminary result 指向：late synthesis、context uptake、clarification、comparison 和 selection criteria 与较高 KE 的关联更强。

## Reproducibility

```powershell
pip install -r requirements.txt
$env:GENAI_DATA_ROOT="D:\path\to\organized_project_files"
```

Goal 1 的正式执行顺序和结果解释见
[`goal1_user_brainstorm_process/README.md`](goal1_user_brainstorm_process/README.md)。

数据隐私与未上传内容见
[`shared_methodology/data_and_privacy.md`](shared_methodology/data_and_privacy.md)。

## Interpretation boundary

大规模 classifier 结果用于描述总体 process；人工校准结果用于解释 semantic wide/local transition。DTR-style estimates 和其他模型结果均为控制可观测状态后的 association，不是 randomized causal effects。
