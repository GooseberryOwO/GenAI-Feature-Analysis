# Goal 1: User Brainstorming Process

研究问题：**How do people use GenAI to conduct brainstorming activity?**

## Analysis layers

1. Shared four-class classifier 为每个 user turn 生成 creative-state label。
2. Conversation aggregation 识别 creative runs、过程顺序和 topic differences。
3. Multiple-option candidate extraction 找到 AI 可能给出多个 alternatives 的 turns。
4. Human semantic calibration 区分 genuine alternatives、procedural/factual lists，并校正后续 wide/local reactions。

四分类全量结果与人工校准 wide/local 结果回答不同问题，不应合并为同一个准确率。

## Official scripts

```text
src/
  extract_wide_local_candidates.py        Rule-based candidate extraction only
  build_weighted_multiturn_flow.py        Builds the audit path and weights
  apply_manual_multiturn_calibration.py   Applies the completed semantic audit
  run_goal1_process_analysis.py           Full-data taxonomy, runs, clusters, topics
  summarize_calibrated_goal1.py           Bootstrap CI and calibrated figures
  make_sample_workflow.py                 Workflow figure utility
```

Run:

```powershell
$env:GENAI_DATA_ROOT="D:\path\to\organized_project_files"
python src/apply_manual_multiturn_calibration.py
python src/run_goal1_process_analysis.py
python src/summarize_calibrated_goal1.py
```

`run_goal1_process_analysis.py` also accepts `GENAI_GOAL1_ROOT` for a separate output directory.

## Main outputs

- [`reports/goal1_brainstorm_process_report.md`](reports/goal1_brainstorm_process_report.md): 中文正式结果总结。
- [`reports/goal1_literature_process_map.md`](reports/goal1_literature_process_map.md): literature concepts 与项目变量的对应。
- [`results/manual_calibrated_t1_to_t2_with_ci.csv`](results/manual_calibrated_t1_to_t2_with_ci.csv): 人工校准转移率和 bootstrap 95% CI。
- [`results/process_taxonomy_summary.csv`](results/process_taxonomy_summary.csv): conversation process taxonomy。
- [`results/three_group_cluster_summary.csv`](results/three_group_cluster_summary.csv): 三类可解释用户过程。
- [`results/topic_family_process_summary.csv`](results/topic_family_process_summary.csv): topic-family differences。

Raw conversation text 和 row-level audit data 只保存在本地 private folders。
