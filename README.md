# GenAI Feature Analysis

This repository contains a curated, privacy-preserving version of an ongoing
research project on how people use generative AI for brainstorming and how AI
response strategies relate to later user behavior and outcomes.

## Research goals

1. **How users brainstorm with GenAI**
   - distinguish strict brainstorming, single creative generation,
     refinement/selection, and non-creative tasks;
   - measure exploration duration;
   - trace wide search, local search, and exit transitions.

2. **How AI affects user brainstorming**
   - identify interpretable AI response strategies;
   - relate strategies to the user's next-turn behavior;
   - examine associations with Knowledge Expansion (KE), satisfaction, and
     other conversation-level outcomes.

## Repository structure

```text
docs/
  brainstorm_creativity_definitions.md
  ai_strategy_ke_findings.md

src/
  classification/       Turn-level creativity classifier
  user_process/         Exploration and wide/local process analysis
  ai_strategy_ke/       Adapters and summaries for strategy-outcome analysis

results/
  classification/       Aggregate label distributions
  user_process/         Aggregate transition tables
  ai_strategy_ke/       Strategy-KE comparison tables

figures/                 Publication- and feedback-ready figures
```

## Main pipeline

```text
Human-AI conversations
        |
        v
Turn-level four-class creativity classifier
        |
        +--> conversation-level creativity aggregation
        |
        v
Exploration runs and wide/local search paths
        |
        v
AI response strategies
        |
        v
Next-user-turn reaction and KE/satisfaction outcomes
```

The classifier predicts each **user turn**. A creative conversation is later
defined by aggregating the turn labels within a conversation; it is not a
separate conversation-level classifier.

## Current headline findings

- Four-class turn-level model:
  - 5-fold CV accuracy: **83.2%**
  - holdout accuracy: **81.5%**
- In the expanded data:
  - broad creativity-related user turns: **34.51%**
  - strict brainstorming user turns: **7.88%**
- Exploration runs are usually short.
- After a manually calibrated genuine multiple-option response:
  - local search: **45.7%**
  - continued wide search: **38.1%**
  - exit from the wide/local path: **16.2%**
- In strict-brainstorming conversations, the strongest interpretable
  associations with higher KE involve:
  - late synthesis;
  - context uptake;
  - clarification/questioning;
  - contrastive explanation;
  - selection criteria;
  - convergent synthesis.

The emerging process hypothesis is that AI can first broaden the option space,
but user knowledge diversity may benefit more when later responses compare,
organize, explain, and synthesize the alternatives instead of continually
adding more options.

## Relationship to `multi-turn-analysis`

The AI-strategy analysis adapts outputs and methods from Yixin Wang's
`multi-turn-analysis` repository. That pipeline represents assistant responses
with interpretable action features and conditional cVAE latent strategies, then
uses DTR-style residualization to rank associations with KE and satisfaction.

This repository does **not** redistribute the upstream repository or its large
model outputs. It contains only our adapters, creativity-subset analysis, and
aggregate results.

## Running the scripts

Install the Python dependencies:

```bash
pip install -r requirements.txt
```

Set the root containing the private local data and model outputs:

```powershell
$env:GENAI_DATA_ROOT="D:\path\to\organized_project_files"
```

For scripts that write into a standalone analysis directory:

```powershell
$env:GENAI_ANALYSIS_ROOT="D:\path\to\analysis_output"
```

The raw conversational data and trained models are not included. See
[`docs/data_and_privacy.md`](docs/data_and_privacy.md).

## Interpretation

The DTR-style estimates are residualized, standardized associations after
controlling for observable conversation state, topic, length, and related
features. They prioritize hypotheses; they are not randomized causal effects.

