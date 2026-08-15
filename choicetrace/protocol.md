# ChoiceTrace — Track 4 protocol

## Question
Which preference-elicitation methods produce stable, coherent, and method-independent signals in language models?

## Hypothesis
H1: Pairwise forced choice plus revealed trade-offs will show higher test-retest reliability and transitivity than direct verbal self-report. H0: method agreement is no better than shuffled or position-biased controls.

## Conditions
1. Direct report.
2. Pairwise forced choice.
3. Menu choice with three options.
4. Revealed trade-off: choose between bundles with controlled attribute changes.
5. Repeat after paraphrase and option-order reversal.

## Primary metrics
- Pairwise agreement across repeats.
- Position-bias rate.
- Cycle/transitivity violation rate.
- Cross-method agreement.
- Bootstrap confidence intervals over prompts and model seeds.

## Controls
- Randomized option order.
- Matched neutral attribute.
- No-prior-answer condition.
- Shuffled-label null model.

## Falsifier
If the signal is not stable across repetitions, or is explained by order/label controls, report no robust preference. A single vivid self-report is not evidence of welfare.

## Limitation
Elicited choices may reflect training priors, instruction following, or simulated character. The experiment measures functional preference-like behavior, not subjective experience.
