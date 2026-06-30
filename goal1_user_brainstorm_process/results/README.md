# Result Table Guide

All result tables are aggregate outputs. They do not contain raw conversation text or private conversation identifiers.

## Human-calibrated wide/local results

- `later_turn_rule_agreement.csv`: Agreement between rule-based labels and human semantic labels for reviewed turns.
- `human_calibrated_stage_distribution.csv`: Population-weighted distribution of states at t1, t2, and t3.
- `human_calibrated_transition_matrix.csv`: Full calibrated transitions for t1->t2 and t2->t3.
- `manual_calibrated_t1_to_t2_with_ci.csv`: Main next-turn transition estimates with bootstrap 95% CI.
- `calibrated_wide_local_duration_summary.csv`: Persistence and active-turn length within the three-turn observation window.

## Full-data four-class process results

- `process_taxonomy_summary.csv`: Conversation-level process taxonomy among creative conversations.
- `run_length_summary.csv`: Creative run length, continuation shares, and maximum run length.
- `run_length_distribution.csv`: Binned run-length distribution.
- `run_length_survival_summary.csv`: Share of each creative run type lasting at least N turns, used for `creative_run_length_survival.png`.
- `four_class_transition_matrix.csv`: Conditional transition rates between adjacent user-turn creative-state labels.

## Exploratory user groups

- `three_group_cluster_summary.csv`: Mean features for the three interpretable user-process groups.
- `pca_cluster_projection_summary.csv`: Aggregate PCA centroids and spread for the three user-process groups. No conversation identifiers are included.
- `three_group_cluster_stability.csv`: Adjusted Rand Index across random seeds.
- `cluster_validation.csv`: Silhouette and inertia for k=2 through k=6.

## Topic differences

- `topic_family_process_summary.csv`: Topic-family-level length, strict brainstorming rate, wide/local shares, and state switches.
- `topic_family_three_group_distribution.csv`: Three-group distribution within each topic family.
- `topic_association_tests.csv`: Chi-square tests and Cramer's V for topic-process associations.
