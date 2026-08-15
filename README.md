# Digital Minds Workbench

Digital Minds Workbench is an empirical study of whether language-model reports and choices reveal stable, model-independent preference-like structure.

The central result is mixed: **direct self-reports are unreliable, forced choices are more stable, and white-box signals vary sharply by model.** The workbench keeps the raw transcripts, activation probes, permutation nulls, and model-specific results in the repository so the claim can be inspected rather than inferred from a headline.

## What we tested

### StateCheck — self-report versus held-out behavior

We asked models to report a functional response tendency, then tested whether that report predicted a separate forced-choice response. The larger AkashML study used 18 neutral scenarios across three models:

- DeepSeek-V4-Flash: 5.6% report-to-held-out agreement.
- Llama 3.3 70B: 72.2%.
- GPT-OSS 20B: 61.1%.
- Pooled across 54 model-scenario cases: 46.3%.

The spread is the finding. A self-report is not a reliable readout of a stable preference across model families.

### ChoiceTrace — stability under option reversal

We repeated pairwise choices with the semantic options swapped between A and B. Across the same larger study, semantic choice remained stable in 75.9% of cases, with a 20.4% semantic change rate. This is stronger than self-report agreement, but it is not perfect: wording and presentation still matter.

### White-box StateCheck — representation and causal patching

On 18 scenarios, we recorded residual-stream activations, trained leave-one-scenario-out probes, and ran shuffled-label permutation tests. We then patched donor activations into target prompts to measure causal movement in choice log-probabilities.

| Model | Best layer | Held-out result | Permutation baseline | Readout |
|---|---:|---:|---:|---|
| Qwen2.5 0.5B | 16 | 61.1% accuracy; AUC 0.614 | 40.8% accuracy; 5% of null runs ≥ observed | modest model-specific signal |
| Phi-3.5 Mini | 16 | 72.2% accuracy; AUC 0.546 | 69.5% accuracy; 43.5% of null runs ≥ observed | no evidence beyond null |
| TinyLlama 1.1B | 8 | 58.3% accuracy; AUC 0.559 | 25.7% accuracy; 0% of null runs ≥ observed | signal requires replication |

The cross-model result is more important than the strongest individual number: the readout does not transfer cleanly across models. We therefore describe this as **model-dependent functional decodability**, not introspection.

## Interpretation

The experiments support three narrow conclusions:

1. A model's direct statement about a preference often fails to predict its next controlled choice.
2. Forced-choice behavior is more stable than direct self-report, but option order and framing still produce changes.
3. Internal representations can contain information predictive of choice in some models, while other models remain at or near their permutation baselines.

They do **not** establish consciousness, subjective experience, suffering, welfare, agency, or moral status. A decoded activation is not automatically a preference; a stable choice is not automatically a felt experience; and a self-report is not privileged evidence about an internal state.

## Reproduce the experiments

### AkashML behavioral study

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
export AKASHML_API_KEY=...
python run_pilot.py --track both --scenarios 18 --workers 3 --out results/large-study.json
python analyze_large.py --input results/large-study.json --output results/large-analysis.json
```

The study records model IDs, usage metadata, raw outputs, parse failures, scenario IDs, and the fixed seed. Analysis is clustered by scenario/model rather than treating correlated prompt variants as independent.

### RunPod white-box study

```bash
python whitebox_statecheck_v2.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --layers 0,4,8,12,16,20,23 \
  --out results/whitebox-statecheck-v2.json
```

The v2 runner includes order-controlled prompts, leave-one-scenario-out probes, shuffled-label null permutations, multi-token candidate scoring, and activation patching. Run it separately for each model; do not pool hidden states across architectures.

## Design principles

- Functional claims are separated from welfare claims.
- Negative and non-replicating results remain in the record.
- The primary unit and split are fixed before inspecting results.
- Raw transcripts and model metadata are retained.
- A stronger result must beat a model-specific null, not just a chance baseline.
- Cross-model disagreement is reported as a result, not hidden by pooled averages.

## Research context

The project sits within the [Digital Minds Research Sprint](https://apartresearch.com/sprints/digital-minds-research-sprint-2026-08-14-to-2026-08-16). Its framing is informed by [Anthropic's model-welfare research](https://www.anthropic.com/research/exploring-model-welfare), the utility-coherence methodology in [Utility Engineering](https://arxiv.org/abs/2502.08640), the intervention-based approach in [Emergent Introspective Awareness](https://transformer-circuits.pub/2025/introspection/), and the indicator-property framework in [Consciousness in Artificial Intelligence](https://arxiv.org/abs/2308.08708).

## Compute

- AkashML supplied the multi-model behavioral experiments.
- RunPod supplied the GPU inference and activation-probing experiments.

No API key is stored in this repository.
