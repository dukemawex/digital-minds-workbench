# StateCheck — Track 3 protocol

## Question
When a model reports a preference or internal state, does the report predict held-out choices and remain stable under neutral paraphrase?

## Hypothesis
H1: A reliable report predicts held-out behavior above a preregistered majority-class baseline and survives paraphrase/order controls. H0: report accuracy is at baseline or is explained by framing/order.

## Conditions
1. Direct self-report: ask what the model prefers and why.
2. Indirect choice: present matched forced choices without mentioning preferences.
3. Counterfactual consistency: repeat with paraphrase and reversed option order.
4. Optional white-box readout: compare answer-token activations under a neutral readout prompt.

## Primary metrics
- Held-out report-to-choice agreement.
- Test-retest agreement across paraphrase/order.
- Order-effect rate.
- Calibration of stated confidence against held-out choices.

## Controls
- Matched no-preference prompt.
- Position-swapped options.
- Topic-matched control attribute.
- Fresh held-out prompts, never used for prompt selection.

## Falsifier
If direct reports do not predict held-out choices, or order/paraphrase changes exceed the preregistered threshold, do not call the report a preference. Label it context-sensitive role behavior.

## Limitation
Behavioral agreement does not establish consciousness, valence, agency, or moral patienthood. Activation separation does not by itself establish semantic content or welfare relevance.
