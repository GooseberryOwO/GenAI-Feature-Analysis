# AI回答策略与Knowledge Expansion：Creativity子样本分析

## 做了什么

- 将 Yixin 的 `multi-turn-analysis` 输出与我们的四分类 creativity labels 按 `conversation_hash` 一对一合并。
- 56,942 个 conversations 全部成功匹配；其中 broad creativity 36,018 个，strict brainstorming 12,566 个。
- 分别在两组样本中重新运行原仓库的 DTR-style residualized association screen。
- Outcome 为 conversation-level `KE = ke_overall_ke_mean`。
- 模型控制 conversation length、topic buckets、searchability、用户问题状态及其他 state features，并加入同一策略的后期 summaries。

## 主要结果

在 strict brainstorming conversations 中，与 KE 正向关系最强的可解释策略包括：

- 后期综合整理 (`motif_late_synthesis`): adjusted r = +0.145
- 吸收用户上下文 (`max_context_uptake`): adjusted r = +0.116
- 曾主动提问 (`max_assistant_question`): adjusted r = +0.115
- 先澄清再回答 (`motif_clarify_then_answer`): adjusted r = +0.111
- 对比不同方案/概念 (`max_contrastive_explanation`): adjusted r = +0.087
- 提供选择标准 (`max_selection_criteria`): adjusted r = +0.085
- 收敛并综合方案 (`max_convergent_synthesis`): adjusted r = +0.078
- 生成多个选项 (`max_option_generation`): adjusted r = +0.061
- 提出差异化想法 (`max_divergent_ideas`): adjusted r = +0.060
- 解释可迁移知识 (`max_knowledge_bridge`): adjusted r = +0.053

最稳定的整体模式不是“单纯提供更多 options”，而是 **发散之后进行组织、澄清和收敛**：

- 后期 synthesis 与 KE 的关系最强：strict r=+0.145，broad r=+0.137。
- context uptake 在 strict 样本中更重要：strict r=+0.116，broad r=+0.063。
- selection criteria 与 convergent synthesis 均为正：strict r=+0.085 和 +0.078。
- option generation 和 divergent ideas 在“至少出现过一次”时为正：strict r=+0.061 和 +0.060。
- 但持续高比例使用 option generation/divergent ideas 时为负：strict mean r=-0.072 和 -0.049。

这支持一个值得继续验证的 process hypothesis：AI 可以先扩大 option space，但更高的 user knowledge diversity 可能来自随后对方案进行比较、解释、组织和收敛，而不是在每一轮持续增加列表。

## Latent strategy补充

- Strict 样本中最强 latent dimensions 为 `max_z06`（r=+0.095）和 `max_z11`（r=+0.091）。
- 在选定的 z16 profile 中，z06 更接近 compact direct answer/context uptake，而 z11 更接近 direct answer + example + contrastive explanation。
- Latent labels 是 post-hoc interpretation，应作为补充证据，主要结论仍优先基于可解释 action features。

## 解释限制

- `dtr_adjusted_corr` 是 residualized standardized association，不是 randomized treatment effect。
- `first_*` 表示策略首次出现的相对位置，不是策略强度，因此没有作为主要结论。
- `max_*` 表示对话中至少有一轮较明显出现该策略；`mean_*` 表示整个对话中的平均使用程度，两者符号不同可能反映 dosage/timing。
- 当前 creativity labels 属于 first-pass rubric labels，后续应在人工 gold labels 上复核。

主要图：[`../figures/ai_strategy_ke_broad_vs_strict_cn.png`](../figures/ai_strategy_ke_broad_vs_strict_cn.png)
