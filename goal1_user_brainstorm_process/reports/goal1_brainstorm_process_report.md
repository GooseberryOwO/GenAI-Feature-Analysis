# Goal 1：用户如何使用 GenAI 进行 brainstorming

## 一句话结论

Goal 1 已经形成一个稳定的 preliminary result：多数 creative conversations 并不是长时间连续 brainstorm，而是短暂生成、短暂扩展，或在 broader task 中插入一小段 creative work；真正从 wide search 进入 local refinement、再反复打开 search space 的过程存在，但比例较低、conversation 明显更长，也更接近 design-like interaction。

## 研究问题与分析单位

研究问题是：**How do people use GenAI to conduct brainstorming activity?**

本轮分析区分两个层次：

- **Turn-level creative state**：对每个 user turn 标注 `strict brainstorming`、`single creative generation`、`refinement / selection`、`non-creative task`。
- **Conversation-level process**：把同一 conversation 的 turn labels 按时间顺序组合，观察 creative run、wide/local transition、process taxonomy、topic difference 和用户过程类型。

全量数据包含 **262,718 个 user turns** 和 **56,942 个 conversations**；其中 **36,018 个 conversations** 至少包含一个 creative turn。

## 方法概览

1. 用已训练的四分类模型给全量数据生成 turn-level creative-state labels。
2. 在 conversation 内聚合 turn labels，识别 creative runs、相邻 state transitions 和 process taxonomy。
3. 从 AI responses 中抽取可能包含多个 options 的 candidate turns。
4. 对 candidate paths 做人工语义校准，区分 genuine multiple-option responses、procedural/factual lists，以及下一轮用户是 wide search、local search、other creative continuation、non-creative shift 还是 conversation end。
5. 对人工校准结果做 population weighting 和 bootstrap 95% CI。

这里需要特别区分：四分类模型用于全量 descriptive analysis；wide/local transition 的正式结论采用人工校准后的结果，因为 rule-based wide/local label 只能作为候选抽取工具。

## Finding 1：creative activity 通常很短

Creative activity 多数是短 run，而不是长时间连续 exploration。

| Run type | Mean length | Median | Share >= 2 turns | Share >= 3 turns | Share >= 5 turns |
|---|---:|---:|---:|---:|---:|
| All creative runs | 1.86 | 1 | 42.0% | 17.2% | 5.3% |
| Broad exploration runs | 1.82 | 1 | 39.9% | 16.1% | 5.0% |
| Strict brainstorming runs | 1.32 | 1 | 19.5% | 5.2% | 1.1% |
| Local refinement runs | 1.20 | 1 | 12.6% | 3.2% | 0.8% |

解释：用户经常在 creative 和 non-creative work 之间快速移动。一个 conversation 里出现 creative turn 并不代表整段 conversation 都在 sustained brainstorming。

对应图：`figures/creative_run_length_summary.png`

## Finding 2：conversation process 以一次性生成和未收窄的 wide exploration 为主

在 36,018 个 creative conversations 中：

| Process taxonomy | Conversations | Share | Mean turns |
|---|---:|---:|---:|
| Single creative generation only | 20,445 | 56.8% | 4.58 |
| Wide exploration without observed local refinement | 11,535 | 32.0% | 5.59 |
| Local refinement without observed wide exploration | 3,007 | 8.3% | 5.81 |
| Wide to local | 462 | 1.3% | 8.68 |
| Local to wide | 315 | 0.9% | 9.71 |
| Iterative reopening | 254 | 0.7% | 16.75 |

解释：最常见的 creative use 是“生成一个东西”或“要更多 alternatives 但没有明显进入某一个 option 的 refinement”。越接近 wide/local 往返，conversation 越长。`iterative reopening` 虽然只有 0.7%，但平均 16.75 turns，可能代表更深入的 creative/design process。

对应图：`figures/process_taxonomy_distribution.png`

## Finding 3：genuine multiple-option response 后，用户不一定继续 wide search

人工校准读取了 **70 个 genuine multiple-option paths** 的后续两轮，共 **140 个 user turns**。规则与人工语义标注一致率为：

| Stage | Agreement |
|---|---:|
| Next user turn | 54.3% |
| Following user turn | 67.1% |

这说明 raw rule-based labels 会把不少 procedural/factual lists 或一般 creative follow-up 误当成 wide/local process。因此正式 transition 采用人工校准后的估计。

在人审确认的 genuine multiple-option paths 中：

| Initial state | Next state | Estimated share |
|---|---|---:|
| Wide search | Wide search | 28.4% |
| Wide search | Local search | 16.2% |
| Wide search | Other creative continuation | 36.9% |
| Wide search | Conversation end | 18.4% |
| Local search | Wide search | 13.7% |
| Local search | Local search | 48.1% |
| Local search | Non-creative shift | 12.9% |
| Local search | Conversation end | 22.1% |

解释：用户一旦进入 local search，更容易继续围绕某个 option refine；但如果处于 wide search，下一步经常不是直接收窄到 local，而是继续 broad creative subtask 或结束。这一结果更适合提出 process hypotheses，而不是当作精确因果估计，因为人工样本量还不大，95% CI 较宽。

对应图：`figures/manual_calibrated_next_turn_transitions.png`、`figures/rule_vs_human_agreement.png`、`figures/human_calibrated_multiturn_sankey_cn.png`

## Finding 4：可以归纳出三类用户过程

为了概括 conversation-level behavior，用 turn shares、creative run count、longest creative run、creative-state switches、creative start position 和 conversation length 做 KMeans，并固定为三个可解释群体。

| User-process group | Conversations | Share | Main pattern |
|---|---:|---:|---|
| Brief creativity within broader tasks | 19,992 | 55.5% | creativity 只是 broader task 中的一小段；non-creative turns 约 57.6% |
| Creative-session dominant | 11,309 | 31.4% | conversation 主要围绕 creative output；single-generation share 约 80.5% |
| Extended iterative process | 4,717 | 13.1% | 平均 13.96 turns，约 3 个 creative runs，state switches 更多 |

Cluster 在 10 个 random seeds 下较稳定，Adjusted Rand Index 为 **0.972-1.000**。不过 k=5 在测试范围内 silhouette 更高，所以 k=3 是为了理论解释和沟通而采用的简化结构。

对应图：`figures/three_group_cluster_profiles.png`

## Finding 5：topic 与过程有关，但不是决定因素

Topic family 与三类 user process 的关系显著，Cramer's V = **0.245**；topic family 与 process taxonomy 的 Cramer's V = **0.190**。这说明 topic 有实际关联，但不能单独解释用户如何 brainstorming。

几个明显 pattern：

- Education/research 的 strict-brainstorming conversation rate 较高，为 **47.8%**。
- Translation/language 为 **44.8%**。
- Marketing/branding 为 **42.0%**，并且平均 creative-state switches 最高。
- Creative writing 和 visual design/media 更偏向 creative-session-dominant，而不是 broader-task 中的短暂 creativity。

对应图：`figures/topic_three_group_heatmap.png`

## Qualitative patterns

- **Single generation**：用户要一个 creative artifact，例如标题、文案、邮件、故事段落，之后只做轻微修改或直接结束。
- **Wide without local**：用户持续要求更多 names、titles、captions、translations、design alternatives，但没有明确选择某一个 option。
- **Wide to local**：用户先要多个 alternatives，随后点名一个 option，加入 constraints，要求比较、改写或 implementation。
- **Local to wide**：用户先细化已有方向，发现不满意或出现新 constraints 后，重新要求 alternatives。
- **Iterative reopening**：用户在 alternatives 和 refinement 之间往返，最接近反复探索的 design process。
- **Local without wide**：用户带着已有 idea 进入，主要要求 critique、improvement、expansion 或 operationalization。

## 目前限制

1. 四分类模型适合全量 descriptive analysis，但不是专门训练的 semantic wide/local classifier。
2. 人工校准样本经过 stratified population weighting，但样本量仍小，因此 CI 较宽。
3. 三群聚类是 exploratory/descriptive，不代表天然存在三个离散用户类型。
4. Topic family 来自现有 domain labels 归并，`Other` 类仍较大。
5. 当前结果描述 association 和 process sequence，不构成 AI strategy 对用户行为的因果效应。

## 可以给老师看的主文件

建议主发这份 report，并附 2-3 张最能讲故事的图：

- `reports/goal1_brainstorm_process_report.md`
- `figures/sample_construction_workflow_cn.png`
- `figures/human_calibrated_multiturn_sankey_cn.png`
- `figures/manual_calibrated_next_turn_transitions.png`

如果老师想复核细节，再补充 `results/manual_calibrated_t1_to_t2_with_ci.csv` 和 `results/process_taxonomy_summary.csv`。

## 连接到 Goal 2

Goal 1 已经提供了用户的 current state 和 next user reaction labels。下一步 Goal 2 可以把 AI response strategy 放在状态转移之前，研究：

`current user state + AI strategy -> next user reaction -> KE / satisfaction`

重点问题包括：什么 AI strategy 会让用户继续 wide exploration、转入 local refinement、重新打开 search space，或离开 creative process；以及这些中间 reaction 是否解释 AI strategy 与最终 KE/user diversity 的关系。
