# Result Table Guide

## Human-calibrated wide/local results

- `later_turn_rule_agreement.csv`: rule label 与人工语义 label 的一致率。
- `human_calibrated_stage_distribution.csv`: t1、t2、t3 各状态的 population-weighted distribution。
- `human_calibrated_transition_matrix.csv`: t1→t2 和 t2→t3 的完整 calibrated transitions。
- `manual_calibrated_t1_to_t2_with_ci.csv`: t1→t2 的 weighted estimates 与 bootstrap 95% CI。
- `calibrated_wide_local_duration_summary.csv`: 三轮观察窗口中的 persistence 和 active-turn length。

## Full-data four-class process results

- `process_taxonomy_summary.csv`: creative conversation process taxonomy。
- `run_length_summary.csv`: creative run 的长度、持续比例和最大值。
- `run_length_distribution.csv`: run-length distribution。
- `four_class_transition_matrix.csv`: 四分类相邻 user turns 的 conditional transition rates。

## Exploratory user groups

- `three_group_cluster_summary.csv`: 三个可解释 user-process groups 的平均特征。
- `three_group_cluster_stability.csv`: 不同随机种子的 Adjusted Rand Index。
- `cluster_validation.csv`: k=2–6 的 silhouette 和 inertia。

## Topic differences

- `topic_family_process_summary.csv`: topic family 的长度、strict rate 和 process features。
- `topic_family_three_group_distribution.csv`: 各 topic family 内三群分布。
- `topic_association_tests.csv`: chi-square 与 Cramer's V。

所有表均为 aggregate results，不包含 conversation text 或 conversation identifiers。
