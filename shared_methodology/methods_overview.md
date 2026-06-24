# Methods overview

## Turn-level creativity classification

The primary classifier assigns each user turn to one of four categories:

1. `strict_brainstorming`
2. `single_creative_generation`
3. `refinement_or_selection`
4. `non_creative_task`

The implementation combines word and character TF-IDF features with a
LinearSVC classifier.

## Conversation aggregation

A conversation is broad creativity-related if it contains at least one turn
classified as strict brainstorming, single creative generation, or
refinement/selection. Strict-brainstorming conversations contain at least one
strict brainstorming turn.

## User-process analysis

Consecutive creative turns are grouped into exploration runs. Follow-up
behavior after genuine multiple-option AI responses is coded as:

- **wide search**: asks for additional, different, or more general options;
- **local search**: selects, narrows, modifies, or develops a particular option;
- **exit**: leaves the wide/local path through a non-creative move, another
  creative activity, or conversation termination.

## AI response strategies and KE

AI response strategies include observable features such as clarification,
option generation, divergent ideas, selection criteria, context uptake,
contrastive explanation, and synthesis. Conditional cVAE latent dimensions are
used as complementary strategy representations.

For each strategy feature and outcome, the DTR-style screen residualizes both
the strategy and outcome against observable controls and reports their
correlation. The main outcome analyzed here is conversation-level Knowledge
Expansion (`ke_overall_ke_mean`).

