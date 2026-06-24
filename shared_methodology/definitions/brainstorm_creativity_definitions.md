# Brainstorm / Creativity Definition Notes for Classification

This document records the literature-based definitions and interpretation notes for our brainstorm / creativity classifier. The goal is to keep a traceable reference file that can later support the paper's method section, coding rubric, and references.

## 1. Why This Definition Work Matters

Our current classifier should not be interpreted too narrowly as a strict `brainstorming` classifier unless the training labels are explicitly defined that way. In the literature, brainstorming is usually closer to **divergent idea generation**, meaning users generate multiple ideas/options/solutions before evaluation or selection. However, creativity is broader: a user may request one creative artifact, such as a slogan, story, title, image prompt, or social media post, and that can still be creative even if it is not strict brainstorming.

Therefore, for our project we should distinguish:

- **Strict brainstorming**: user explores multiple possible ideas/options/solutions.
- **Single creative generation**: user asks for one creative output.
- **Refinement / selection**: user edits, evaluates, selects, or improves generated ideas.
- **Non-creative task**: factual, explanatory, closed-form, coding/debugging, translation, or administrative task.

This distinction directly addresses the question raised in the meeting: if a user only wants one answer, should that count as brainstorm? Literature suggests: it can count as broad creativity-related behavior, but not necessarily strict brainstorming.

## 2. Literature-Based Definitions

### 2.1 Brainstorming as Divergent Idea Generation

Classic brainstorming literature, starting from Osborn's *Applied Imagination*, treats brainstorming as a method for producing many ideas before judging them. The common principles are:

- defer judgment/evaluation
- seek quantity of ideas
- encourage freewheeling or unusual ideas
- combine and improve ideas

For our coding, this supports a stricter definition: a prompt is `strict_brainstorming` when the user asks ChatGPT to produce multiple ideas, options, alternatives, names, designs, approaches, directions, or solutions.

Examples:

- "Give me 10 ideas for..."
- "What are some possible names for..."
- "List different ways to..."
- "Suggest several design options..."
- "Give me alternatives / variations..."

### 2.2 Divergent Thinking and Creativity Measurement

Guilford's creativity work and later divergent-thinking measures such as the Alternative Uses Task and Torrance Tests of Creative Thinking emphasize generating multiple possible responses. Common scoring dimensions include:

- **fluency**: number of ideas
- **flexibility**: variety of categories
- **originality**: novelty/unusualness
- **elaboration**: amount of detail/development

This helps us operationalize brainstorming in a measurable way. Strict brainstorming is not only about whether the content is creative; it is also about whether the user is exploring a space of possible ideas.

### 2.3 Creativity as Novelty + Usefulness

Runco and Jaeger (2012) summarize the standard definition of creativity as involving both originality/novelty and effectiveness/usefulness/appropriateness. This is broader than brainstorming.

For our classifier, this means a prompt like "write a slogan for my coffee shop" may be creativity-related because it requests a novel and useful artifact, but it is not necessarily strict brainstorming unless the user asks for multiple slogan options or alternatives.

### 2.4 Divergent vs. Convergent Creative Processes

OECD's creative-thinking framework distinguishes divergent-exploratory tasks from convergent-integrative tasks. Divergent tasks ask people to generate different ideas; convergent/integrative tasks involve evaluating, improving, selecting, or synthesizing ideas.

This is useful for our transition analysis:

- brainstorm -> non-brainstorm may often mean generation -> refinement / evaluation / implementation
- non-brainstorm -> brainstorm may often mean context setup -> creative exploration

So a topic transition is not always a complete topic shift; it may be a workflow-stage shift within the same broader project.

### 2.5 LLM Creativity and Diversity

Recent LLM creativity papers suggest that generative AI can improve individual creative outputs but may reduce collective diversity because outputs become more similar across users. This connects strongly to our core information-diversity research question.

For our project, the interesting behavioral question becomes:

- Does ChatGPT help users explore more diverse idea spaces?
- Or does it anchor users toward similar answers?
- Do users continue exploring when the AI gives multiple options?
- Do users move into refinement more quickly when the AI gives one polished answer?

This also suggests future model features: whether the AI response contains one answer vs. multiple options, and whether that affects the user's next turn.

## 3. Proposed Coding Rubric

| Label | Definition | Include | Exclude |
|---|---|---|---|
| `strict_brainstorming` | User asks for multiple possible ideas/options/solutions before choosing | ideas, options, alternatives, several versions, names, topics, designs, strategies | one final answer only |
| `single_creative_generation` | User asks for one creative artifact | write a story, slogan, post, email, image prompt, title | factual answer, pure grammar correction |
| `refinement_or_selection` | User edits, improves, evaluates, or chooses among generated ideas | make shorter, compare options, choose best one, revise tone, make more creative | initial generation request |
| `non_creative_task` | Closed or non-creative request | factual QA, coding/debugging, translation, summarization, administrative task | creative writing, ideation, design options |
| `unclear` | Not enough context to decide | ambiguous short turns | clearly classifiable turns |

## 4. How This Improves The Classifier

### Current interpretation

The existing classifier is better described as a **broad creativity-related classifier**, not necessarily a strict brainstorming classifier. It likely captures many creative/generative prompts, including single-output requests.

### Recommended next step

Use a manually reviewed sample to assign the finer labels above. Then we can either:

1. Keep the existing binary classifier but rename its target as `broad_creativity_related`.
2. Train a second-stage classifier that separates:
   - strict brainstorming
   - single creative generation
   - refinement/selection
   - non-creative task
3. Report two rates:
   - broad creativity-related rate
   - strict brainstorming rate

This would make the method more defensible in the paper and directly answer the concern that "single-answer creative requests" should not automatically be treated as strict brainstorming.

## 5. Interesting Points To Report

1. **Brainstorming is narrower than creativity.** Literature usually links brainstorming to divergent idea generation, while creativity can include a single novel/useful artifact.

2. **The classifier target needs to be named carefully.** If our labels include single creative generation, then the model is not purely detecting brainstorming; it is detecting broader creativity-related ChatGPT use.

3. **Transitions may represent workflow stages, not topic shifts.** Brainstorm -> non-brainstorm may mean idea generation -> refinement/implementation; non-brainstorm -> brainstorm may mean context setup -> ideation.

4. **LLM outputs may affect exploration.** If the AI gives one polished answer, users may quickly refine or accept it; if it gives multiple options, users may stay longer in exploration. This can become a future quantitative analysis.

## 6. References To Cite Later

### Classical brainstorming / ideation

- Osborn, A. F. (1953). *Applied Imagination: Principles and Procedures of Creative Problem-Solving*. Charles Scribner's Sons.
- Diehl, M., & Stroebe, W. (1987). Productivity loss in brainstorming groups: Toward the solution of a riddle. *Journal of Personality and Social Psychology*, 53(3), 497-509.
- Nijstad, B. A., & Stroebe, W. (2006). How the group affects the mind: A cognitive model of idea generation in groups. *Personality and Social Psychology Review*, 10(3), 186-213.
- Paulus, P. B., & Brown, V. R. (2007). Toward more creative and innovative group idea generation: A cognitive-social-motivational perspective of brainstorming. *Social and Personality Psychology Compass*, 1(1), 248-265.

### Creativity / divergent thinking definition

- Guilford, J. P. (1950). Creativity. *American Psychologist*, 5(9), 444-454.
- Runco, M. A., & Jaeger, G. J. (2012). The standard definition of creativity. *Creativity Research Journal*, 24(1), 92-96.
- Torrance, E. P. (1966). *Torrance Tests of Creative Thinking: Norms-Technical Manual*. Personnel Press.
- Alabbasi, A. M. A., Paek, S. H., Kim, D., & Cramond, B. (2022). What do educators need to know about the Torrance Tests of Creative Thinking: A comprehensive review. *Frontiers in Psychology*, 13, 1000385.
- OECD. (2025). *Seven Questions about Creativity and Creative Thinking*.

### LLM / generative AI creativity

- Doshi, A. R., & Hauser, O. P. (2024). Generative AI enhances individual creativity but reduces the collective diversity of novel content. *Science Advances*, 10(28), eadn5290.
- Kumar, H., Vincentius, J., Jordan, E., & Anderson, A. (2025). Human creativity in the age of LLMs: Randomized experiments on divergent and convergent thinking. *Proceedings of CHI 2025*.
- Lee, B. C., & Chung, J. (2024). An empirical investigation of the impact of ChatGPT on creativity. *Nature Human Behaviour*, 8, 1906-1914.
- Chatterji, A., Cunningham, T., Deming, D., Hitzig, Z., Ong, S., Shan, C., & Wadman, K. (2025). *How People Use ChatGPT*. NBER Working Paper No. 34255.

## 7. Local Files

PDF folder:

相关文献 PDF 保存在项目的私有文献目录中，未纳入公开仓库。

Related files:

- `literature_search_brainstorm_definition.md`
- `pdf_download_summary.md`
- `brainstorm_creativity_definition_notes.md`
