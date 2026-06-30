# Goal 1 visualization package

这份 visual package 的目的，是把 Goal 1 report 里的主要 findings 转成更适合 meeting / slides 讨论的图。核心结果和数字没有改变，只是更换为更清晰的 visual representation。

## Recommended reading order

1. `figures/goal1_finding_summary_board.png`  
   一页概括 Goal 1 的核心结果：creative runs 很短，single generation 和 wide without local 是主要模式，真正 wide->local 和 extended process 比例较低。

2. `figures/process_taxonomy_flow_story.png`  
   用 flow-style 图展示 creative conversations 如何分布到不同 process taxonomy。这个图适合回答“用户 brainstorming 的主要流程类型是什么”。

3. `figures/creative_run_length_survival.png`  
   用 survival-style line chart 展示 creative activity 通常在 1-2 turns 内结束。这个图比单纯 bar chart 更适合表达“持续多久”。

4. `figures/manual_transition_story_ci.png`  
   展示人工校准后的 wide/local next-turn transition，并保留 bootstrap 95% CI。这个图适合说明用户在 genuine multiple-option response 之后并不总是继续 broad option search。

5. `figures/pca_user_process_clusters.png`  
   PCA projection of conversation-level process features。不同颜色对应三类 user-process groups，可以直观看到三类过程在低维空间中的相对位置。

6. `figures/topic_process_bubble_map.png`  
   bubble chart 展示 topic family 与 brainstorming intensity / creative-state switching 的关系。x 轴是 strict brainstorming conversation rate，y 轴是 mean creative-state switches，bubble size 是 conversation count，颜色是 mean turns。

## Main visual story

Goal 1 的 visual story 可以这样讲：

1. 先用 summary board 给出总体结论：常见模式是短暂 creative use，复杂的 iterative search 比较少。
2. 再用 taxonomy flow 展示 creative conversations 主要流向 single generation 和 wide without local。
3. 用 run-length survival 说明 creative activity 为什么说“短”。
4. 用人工校准 transition 图说明 wide/local 的动态过程。
5. 用 PCA projection 说明三类 user-process groups 是可解释的。
6. 最后用 topic bubble map 补充 topic differences，但强调 topic 不是唯一解释因素。
