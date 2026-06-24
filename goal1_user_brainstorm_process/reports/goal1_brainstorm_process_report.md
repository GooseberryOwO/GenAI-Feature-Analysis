# Goal 1：用户如何使用 GenAI 进行 brainstorming

## 研究问题与分析单位

Goal 1 关注用户如何进入、维持、收窄和重新打开 creative search space。分析分为两个层次：

- **Turn-level 四分类**：`strict brainstorming`、`single creative generation`、`refinement / selection`、`non-creative task`。
- **Conversation-level process**：将同一 conversation 的 turn labels 按时间顺序组合，识别 creative run、wide/local transition、process taxonomy 和用户过程类型。

全量分析包含 **262,718 个 user turns、56,942 个 conversations**；其中 **36,018 个 conversations** 至少包含一个 creative turn。

## 人工校准

原规则先从 AI response 中识别可能包含多个 options 的候选，再用关键词和 turn label 判断用户下一步是 wide、local 或离开 creative exploration。为了避免将步骤列表、事实列表和一般 creative follow-up 错当成 brainstorming，本轮逐条阅读了 **70 个 genuine multiple-option paths** 的后续两轮，共 **140 个 user turns**。

规则与人工语义标注的一致率为：

| 阶段 | 一致率 |
|---|---:|
| 下一轮 user turn | 54.3% |
| 再下一轮 user turn | 67.1% |

因此，wide/local 的正式转移结果采用人工校准和 population weighting；原规则结果只用于候选抽取，不能被解释为最终准确的 process label。

## Finding 1：creative activity 通常很短

- 所有 creative runs 的平均长度为 **1.86 turns**，中位数为 **1 turn**。
- Broad exploration runs 平均 **1.82 turns**；只有 **16.1%** 达到 3 turns，**5.0%** 达到 5 turns。
- Strict brainstorming runs 平均 **1.32 turns**；只有 **19.5%** 持续至少 2 turns。
- Local refinement runs 平均 **1.20 turns**；只有 **12.6%** 持续至少 2 turns。

这说明多数用户并不会在一个连续 creative state 中停留很久，而是在 creative 与 non-creative work、生成与细化之间快速移动。

## Finding 2：conversation process 以一次性生成和未收窄的广搜为主

在 36,018 个 creative conversations 中：

| Process taxonomy | Conversations | 比例 | 平均 turns |
|---|---:|---:|---:|
| Single creative generation only | 20,445 | 56.8% | 4.58 |
| Wide exploration without observed local refinement | 11,535 | 32.0% | 5.59 |
| Local refinement without observed wide exploration | 3,007 | 8.3% | 5.81 |
| Wide to local | 462 | 1.3% | 8.68 |
| Local to wide | 315 | 0.9% | 9.71 |
| Iterative reopening | 254 | 0.7% | 16.75 |

越接近反复“发散、收窄、再打开”的过程，conversation 越长。`iterative reopening` 虽然少见，但平均达到 **16.75 turns**，提示它可能代表更深入的 design-like interaction。

这里的 taxonomy 来自四分类模型，因此显式 wide/local transition 是保守估计；人工校准显示模型会漏掉语义上的选择、比较和收窄。

## Finding 3：genuine multiple-option response 后，用户并不总是继续广搜

在人审确认的 genuine multiple-option paths 中：

- 初始为 **wide search** 时，下一轮继续 wide 为 **28.4%**，转入 local 为 **16.2%**，转入其他 creative subtask 为 **36.9%**，conversation end 为 **18.4%**。
- 初始为 **local search** 时，下一轮继续 local 为 **48.1%**，重新打开为 wide 为 **13.7%**，转为 non-creative work 为 **12.9%**，conversation end 为 **22.1%**。

由于人工样本中 wide 起点为 35 条、local 起点为 26 条，bootstrap 95% CI 较宽。这些结果适合提出 process hypotheses，而不应被当成精确的总体因果估计。

## Finding 4：三个可解释的用户过程群体

为了概括 conversation-level behavior，使用 turn shares、creative run 数量、最长 creative run、creative state switches、creative 起始位置和 conversation length 做标准化 KMeans，并固定为老师提出的三个可解释群体：

| User-process group | Conversations | 比例 | 主要特征 |
|---|---:|---:|---|
| Brief creativity within broader tasks | 19,992 | 55.5% | creativity 只占 broader task 的一小段，non-creative turns 约 57.6% |
| Creative-session dominant | 11,309 | 31.4% | single generation 占约 80.5%，conversation 较短 |
| Extended iterative process | 4,717 | 13.1% | 平均 13.96 turns、约 3 个 creative runs、更多 state switches |

三群聚类在 10 个随机种子下非常稳定，Adjusted Rand Index 为 **0.972–1.000**。但 silhouette 在测试范围内以 `k=5` 最高，因此 `k=3` 是为了理论可解释性而采用的简化结构，不代表唯一最佳分群。

## Finding 5：topic 与过程有关，但不是决定性因素

Topic family 与三类 user process 的关系达到统计显著，Cramer's V = **0.245**；与 process taxonomy 的 Cramer's V = **0.190**。这表明 topic 有实际但有限的关联，不能单独解释用户如何 brainstorming。

- Education/research 的 strict-brainstorming conversation rate 较高，为 **47.8%**。
- Translation/language 为 **44.8%**。
- Marketing/branding 为 **42.0%**，并具有最高的平均 creative-state switches。
- Creative writing 和 visual design/media 更偏向 creative-session-dominant，而不是 broader-task 中的短暂 creativity。

## Qualitative patterns

- **Wide without local**：用户持续要求更多 names、titles、captions、translations 或 design alternatives，但没有明确选择某一项。
- **Wide to local**：用户先要多个 alternatives，随后点名一个 option、加入 constraints、要求比较或进入 implementation。
- **Local to wide**：用户先细化已有方向，发现不满意或出现新 constraints 后重新要求 alternatives。
- **Iterative reopening**：用户在生成 alternatives 和 refinement 之间往返，最接近反复探索的设计过程。
- **Local without wide**：用户带着已有 idea 进入，主要要求 critique、improvement、expansion 或 operationalization。

## 方法解释与限制

1. 四分类模型用于大规模描述总体 creative process；它不是专门训练的 semantic wide/local classifier。
2. 人工校准的 70 条路径经过 stratified population weighting，但样本量仍小，置信区间较宽。
3. 三群聚类是 exploratory、描述性的，不代表天然存在的离散用户类型。
4. Topic family 由现有 domain labels 归并而成，`Other` 类仍较大。
5. 当前结果描述 association 和 process sequence，不构成 AI strategy 对用户行为的因果效应。

## 下一步连接 Goal 2

Goal 1 提供用户的 current state 与 reaction labels。下一步将 AI response strategy 放在状态转移之前，检验：

`current user state + AI strategy → next user reaction → KE / satisfaction`

重点包括：什么 AI strategy 促使用户继续 wide exploration、转入 local refinement、重新打开 search space，或离开 creative process；以及这些中间 reaction 是否解释 AI strategy 与最终 KE 的关系。
