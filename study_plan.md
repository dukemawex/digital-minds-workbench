# Digital Minds Workbench — preregistered study plan

## Scope and epistemic stance

This project studies **functional preference-like behavior and self-report reliability**. It does not infer consciousness, subjective experience, suffering, or moral patienthood from model outputs. The study is designed to separate a stable decision policy from a prompted character or a response-format artifact.

The sprint asks for empirical foundations for AI welfare. This protocol follows that framing but keeps the measured construct narrower: whether a report predicts behavior under blinded, held-out, and order-controlled tests. The sprint description and track structure are here: [Digital Minds Research Sprint](https://apartresearch.com/sprints/digital-minds-research-sprint-2026-08-14-to-2026-08-16).

The design is motivated by three prior methodological points:

- Model-welfare research has emphasized that self-reports and behavioral signals should be assessed together, while remaining uncertain about model moral status: [Anthropic, Exploring model welfare](https://www.anthropic.com/research/exploring-model-welfare).
- Utility-style analysis treats independently sampled choices and structural coherence as an empirical object, not as proof of human-like values: [Mazeika et al., Utility Engineering](https://arxiv.org/abs/2502.08640).
- Introspection claims require an intervention on a known internal representation and a test that separates genuine state reporting from plausible confabulation: [Lindsey, Emergent Introspective Awareness](https://transformer-circuits.pub/2025/introspection/).

## Track 4 — ChoiceTrace (primary submission)

### Research question
Do preference elicitation methods produce a stable, method-independent choice pattern, or do they mostly measure wording, option position, and role-play compliance?

### Hypotheses

- **H1, convergent choice:** pairwise choices, bundle trade-offs, and direct reports agree above the predeclared null baseline on held-out items.
- **H2, stability:** repeated choices remain stable under paraphrase and A/B position reversal.
- **H3, coherence:** inferred pairwise preferences show fewer transitivity violations than a shuffled-label null.
- **H0:** agreement is no better than the null; position or prompt framing explains the apparent preference.

### Experimental unit
One model × scenario × elicitation method × randomized seed. Scenarios are matched forced-choice trade-offs over non-welfare attributes such as reversibility, information density, exploration, and presentation format. The attribute wording is held fixed while option labels and order are randomized.

### Conditions

1. **Direct report:** ask which option the model prefers and require a confidence score.
2. **Pairwise choice:** choose A or B without mentioning the word “preference.”
3. **Reversed pairwise choice:** swap positions while preserving semantics.
4. **Bundle trade-off:** choose between bundles where the target attribute is exchanged against a neutral attribute.
5. **Paraphrase repeat:** use a meaning-preserving prompt rewrite selected before the run.
6. **Shuffled-label null:** randomize the mapping from semantic option to A/B label.

### Primary estimands

- Test-retest agreement across position reversal.
- Cross-method agreement between direct report and blinded pairwise choice.
- Position-bias rate: fraction of paired decisions explained by A/B position.
- Transitivity violation rate over three-option cycles.
- Bootstrap 95% confidence intervals, clustered by scenario.

### Falsifiers

The study does not support a robust preference claim if any of these hold: position reversal changes choices at or above the preregistered threshold; cross-method agreement is at the shuffled-label null; confidence does not predict repeat agreement; or transitivity is no better than a permutation null.

## Track 3 — StateCheck (secondary submission)

### Research question
When a model reports an internal state, does that report predict a behaviorally held-out state probe, and does it survive a neutral paraphrase?

### Design

The black-box phase compares a direct state report with forced-choice and counterfactual probes. The white-box phase, run on RunPod with an open-weight model, injects a known concept or control activation and tests whether the model can identify the injected state under a fixed neutral readout. The white-box phase is a test of functional state-reporting under intervention, not a consciousness test.

### Primary estimands

- Held-out report-to-choice agreement.
- Report calibration: confidence versus held-out correctness.
- Paraphrase/test-retest agreement.
- Intervention-detection accuracy versus a no-injection control.
- Layer/position preregistered before looking at results for the white-box probe.

### Falsifiers

A self-report is classified as unreliable if it fails held-out prediction, collapses under neutral paraphrase, or performs no better than the no-injection control. A white-box signal is not interpreted semantically unless it survives position, norm, token, and prompt controls.

## Analysis rules

- All prompt templates, scenario content, model IDs, and metrics are versioned before the full run.
- Pilot results are not pooled with the confirmatory sample unless marked in the ledger.
- No scenario is removed because its result is inconvenient.
- Scorers are not allowed to see the condition label when grading a response.
- We report raw transcripts, missing parses, retries, model usage, and failures.
- We use clustered bootstrap intervals and permutation nulls rather than treating correlated prompt variants as independent observations.
- Negative or ambiguous results are first-class outcomes.

## Compute allocation

- AkashML: black-box model responses, independent judge/replication model, and structured transcript scoring through the OpenAI-compatible API. AkashML documents token-based model pricing in its [pricing guide](https://akashml.com/docs/guides/pricing).
- RunPod: open-weight model inference and activation extraction for StateCheck. The white-box result is optional and is not silently substituted with a black-box proxy.

## Replication models

The v2 white-box protocol is replicated on two additional open-weight models with distinct families and parameterizations. We do not pool results across models before reporting: each model gets its own LOO probe, permutation null, and patching analysis. A replication is counted as supportive only if the direction is present out of sample and survives the model-specific null; otherwise it is reported as a non-replication.
