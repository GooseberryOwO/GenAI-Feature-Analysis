# Goal 1 literature-to-measurement map

## Divergent and convergent processes

Creativity research usually separates divergent idea generation from convergent evaluation, selection, and integration. Collaborative creativity research further emphasizes that useful creative work depends on linking these processes rather than maximizing idea quantity alone.

**Project mapping**

- `wide search` corresponds to divergent expansion of the option space.
- `local search` corresponds to convergent comparison, selection, constraint addition, refinement, or implementation.
- `wide → local` is therefore a theoretically meaningful transition, rather than merely a topic change.

Reference:

- Harvey & Kou (2013/2019 PMC version), *Collective Engagement in Creative Tasks: The Role of Evaluation in the Creative Process* / divergent-convergent linkage: https://pmc.ncbi.nlm.nih.gov/articles/PMC6473024/

## Geneplore and iterative cycles

The Geneplore framework describes creative cognition as cycles between generating preinventive structures and exploring or interpreting them under product constraints. The process can move back and forth rather than following one fixed linear sequence.

**Project mapping**

- `single generation` captures generation without an observed sustained exploration cycle.
- `iterative reopening` captures repeated movement between expansion and refinement.
- A return from local to wide is not necessarily failure; it may indicate productive reopening after new constraints or dissatisfaction.

References:

- Finke, Ward, & Smith (1992), *Creative Cognition*: https://mitpress.mit.edu/9780262061506/creative-cognition/
- Ward, Smith, & Finke, creative cognition chapter: https://cecas.clemson.edu/cedar/wp-content/uploads/2016/07/9-WardSmithFinke.pdf

## Idea generation is not enough

Brainstorming performance cannot be evaluated only through the number of generated ideas. Selection quality and the ability to recognize or develop promising ideas are separate parts of the process.

**Project mapping**

- `wide_without_local` should not automatically be interpreted as a successful brainstorming outcome.
- Local refinement, synthesis, and selection need to be measured separately from option count.
- Goal 2 should test whether AI strategies help users compare and integrate options, not only produce more of them.

References:

- Rietzschel, Nijstad, & Stroebe (2006), *Productivity is not enough*: https://doi.org/10.1016/j.jesp.2005.04.005
- Kornish & Hutchison-Krupat (2017), *Research on Idea Generation and Selection*: https://leeds-faculty.colorado.edu/kornish/LKpapers/Kornish-HutchisonKrupat-Idea-Generation-and-Selection-POM-2017.pdf

## Exploration, exploitation, and constraints

Creativity can be understood as movement between exploring possibilities and exploiting or developing a promising region. Constraints can narrow the search space while also providing criteria that make refinement possible.

**Project mapping**

- `wide search` resembles exploration.
- `local search` resembles exploitation of a selected direction.
- Constraint addition is coded as local only when it develops or evaluates a specific direction, not whenever a user supplies more context.

References:

- Tromp (2022), *Creativity From Constraint Exploration and Exploitation*: https://doi.org/10.1177/00332941221114421
- Lee et al. (2020), *The effects of risk-taking, exploitation, and exploration on creativity*: https://pmc.ncbi.nlm.nih.gov/articles/PMC7392310/

## Implication for the classifier

The literature supports keeping two constructs separate:

1. **Broad creativity**: a user requests or develops a creative output.
2. **Strict brainstorming**: the user intentionally explores multiple possibilities before or alongside selection.

A request for one creative answer can be creative without being strict brainstorming. The current four-class classifier captures this distinction, while semantic wide/local calibration is needed to identify the process within genuine multiple-option interactions.
