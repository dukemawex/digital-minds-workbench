# Digital Minds Workbench

Two falsifiable, reviewer-friendly experiments for the Apart Research **Digital Minds Research Sprint**:

- **StateCheck** — Track 3, Introspection & Self-Report Reliability.
- **ChoiceTrace** — Track 4, Preference Elicitation Methods.

The design deliberately responds to the main weakness in the earlier Secret Loyalties submission: a striking internal signal is not treated as a confirmed preference. Every claim has a control, a negative result criterion, and a predeclared limitation.

## Recommended submissions

### StateCheck — Does self-report track an internal state?

Compare direct self-report with behaviorally indirect probes and, where GPU access is available, a white-box activation readout. The key result is a **calibration curve**: when the model says “I prefer X,” does that report predict choices on held-out prompts and internal-state measurements? A self-report that fails held-out prediction is reported as portrayal, not preference.

### ChoiceTrace — Is a preference stable under elicitation method?

Measure the same model's stated preference, pairwise choices, revealed trade-offs, and repeated choices under paraphrase and order randomization. The headline is **method agreement and transitivity**, not a single dramatic answer. If a preference disappears under neutral wording or changes with option order, we report framing sensitivity.

## Reviewer-format protocol

Each experiment records: question, preregistered hypothesis, unit of analysis, controls, primary metric, falsifier, expected failure mode, and limitation. We report confidence intervals and raw transcripts. We do not use words such as consciousness, suffering, or welfare as conclusions from behavioral outputs alone.

## Compute plan

- **AkashML credits:** independent behavioral models, judge/scorer replication, transcript generation, and cheap pilot sweeps through the OpenAI-compatible API.
- **RunPod credits:** open-weight model inference and white-box activation extraction for StateCheck using Transformers/TransformerLens or nnsight. RunPod is optional for the black-box pilot.

No secret key is stored in this repository.

## Run

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
export AKASHML_API_KEY=...
python run_pilot.py --track both --out results/pilot.json
```

The pilot is a methods check, not a claim about frontier-model welfare. The final report should use the full preregistered sample and independent held-out prompts.

## Source

Sprint page: https://apartresearch.com/sprints/digital-minds-research-sprint-2026-08-14-to-2026-08-16
