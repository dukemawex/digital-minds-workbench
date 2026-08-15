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

## Larger controlled study

The larger run uses 18 preregistered neutral scenarios × 3 AkashML models × 2 tracks. Analysis is clustered by scenario/model rather than treating every prompt response as independent. The run records model usage, raw transcripts, explicit parse failures, and model-stratified results.

```bash
python run_pilot.py --track both --scenarios 18 --workers 3 --out results/large-study.json
python analyze_large.py --input results/large-study.json --output results/large-analysis.json
```

The primary claims are report-to-held-out agreement for StateCheck and semantic stability after option reversal for ChoiceTrace. Both are functional reliability measures; neither is evidence of consciousness or welfare.

## RunPod white-box StateCheck

`whitebox_statecheck.py` is the next-stage experiment. It runs an open-weight model, records residual-stream activations at preregistered layers, trains a leave-scenarios-out linear readout, and compares that readout with direct reports. It is a functional representation test, not a consciousness or welfare test.

```bash
python whitebox_statecheck.py --model Qwen/Qwen2.5-0.5B-Instruct --layers 4,8,12 --out results/whitebox-statecheck.json
```

Run this on a RunPod GPU. The output reports held-out accuracy/AUC and direct-report agreement per layer, with raw model metadata and the exact split seed.
