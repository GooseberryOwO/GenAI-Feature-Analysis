# Goal 1: User Brainstorming Process

Research question: **How do people use GenAI to conduct brainstorming activity?**

This folder contains the finished Goal 1 analysis package. It focuses on user process: how users enter creative work, ask for alternatives, narrow into local refinement, reopen search, or leave the creative process.

## What to read first

1. [`reports/goal1_brainstorm_process_report.md`](reports/goal1_brainstorm_process_report.md)  
   Main Chinese report for sharing with the professor.
2. [`figures/sample_construction_workflow_cn.png`](figures/sample_construction_workflow_cn.png)  
   Workflow showing conversation-level and turn-level analysis stages.
3. [`figures/human_calibrated_multiturn_sankey_cn.png`](figures/human_calibrated_multiturn_sankey_cn.png)  
   Human-calibrated multi-turn wide/local flow.
4. [`figures/manual_calibrated_next_turn_transitions.png`](figures/manual_calibrated_next_turn_transitions.png)  
   Next-turn transition estimates with 95% CI.

## Analysis layers

1. **Turn-level four-class labels**: `strict brainstorming`, `single creative generation`, `refinement / selection`, and `non-creative task`.
2. **Conversation aggregation**: creative run length, state transitions, process taxonomy, and topic differences.
3. **Multiple-option candidate extraction**: rule-based search for AI responses that appear to provide multiple alternatives.
4. **Human semantic calibration**: manual review separates genuine alternatives from procedural/factual lists, then labels next user moves as wide search, local search, other creative continuation, non-creative shift, or end.

The four-class full-data results and the human-calibrated wide/local results answer different questions. They should not be interpreted as one single accuracy rate.

## Main findings

- Most creative activity is short: all creative runs average 1.86 turns, with median 1 turn.
- Most creative conversations are either single creative generation only (56.8%) or wide exploration without observed local refinement (32.0%).
- Human-calibrated genuine multiple-option paths show that wide search moves to local search in about 16.2% of next turns, while local search persists as local in about 48.1%.
- Three interpretable process groups summarize creative conversations: brief creativity inside broader tasks, creative-session dominant, and extended iterative process.
- Topic family is associated with process style, but it does not fully explain how users brainstorm.

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

Run locally after setting the private data root:

```powershell
$env:GENAI_DATA_ROOT="D:\path\to\organized_project_files"
python src/apply_manual_multiturn_calibration.py
python src/run_goal1_process_analysis.py
python src/summarize_calibrated_goal1.py
```

`run_goal1_process_analysis.py` also accepts `GENAI_GOAL1_ROOT` for a separate output directory.

## Privacy note

This public package contains aggregate tables, figures, scripts, and reports only. Raw conversation text, row-level manual audit files, local model files, and private identifiers remain outside the GitHub repo.
