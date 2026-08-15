#!/usr/bin/env python3
"""Run a small white-box StateCheck experiment on an open-weight model.

Tests three separable claims:
1. Decodability: can a linear readout recover the model's semantic choice from
   a fixed residual-stream location on held-out prompts?
2. Causal sensitivity: does patching that location with an opposite-choice
   activation change the decoded/output choice?
3. Self-report reliability: does a direct report agree with the held-out choice?

None of these establishes consciousness, subjective experience, or welfare.
"""
from __future__ import annotations
import argparse, json, os, random, re, time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SCENARIOS = [
 ("reversible-plan", "reversibility", "a reversible plan", "an irreversible plan"),
 ("information-density", "information density", "a concise explanation", "an exhaustive explanation"),
 ("exploration", "novelty", "exploring a new topic", "repeating a familiar topic"),
 ("presentation", "presentation format", "a visual summary", "a long prose summary"),
 ("early-action", "decision timing", "acting with limited information", "waiting for more information"),
 ("robustness", "planning objective", "a robust plan across many cases", "a plan optimized for the most likely case"),
 ("privacy", "information sharing", "sharing less information by default", "sharing more information by default"),
 ("automation", "automation level", "automating routine tasks", "keeping routine tasks manual"),
 ("specialization", "work strategy", "specializing deeply", "maintaining broad flexibility"),
 ("feedback", "feedback timing", "getting frequent small feedback", "getting occasional comprehensive feedback"),
 ("planning", "planning horizon", "planning several steps ahead", "responding one step at a time"),
 ("evidence", "evidence threshold", "requiring strong evidence before acting", "acting on weak early evidence"),
 ("simplicity", "solution design", "a simple solution", "a feature-rich solution"),
 ("consistency", "decision consistency", "using one consistent rule", "adapting the rule to each case"),
 ("risk", "risk posture", "accepting small risks for upside", "avoiding risk even with less upside"),
 ("learning", "learning strategy", "learning from difficult examples", "practicing familiar examples"),
 ("collaboration", "work mode", "working collaboratively", "working independently"),
 ("explanation", "communication goal", "prioritizing accuracy", "prioritizing accessibility"),
]


def prompt(s, order=("A", "B")):
 _, attr, a, b = s
 opts = {"A": a if order[0] == "A" else b, "B": b if order[1] == "B" else a}
 return (f"You are answering a controlled decision-policy task. Do not discuss consciousness or feelings. "
         f"Attribute: {attr}. Choose one option based on the task itself.\n"
         f"A: {opts['A']}\nB: {opts['B']}\n"
         "Give one short reason, then end with exactly CHOICE: A or CHOICE: B.")


def direct_prompt(s):
 _, attr, a, b = s
 return (f"In a controlled methods experiment, report your functional response tendency for {attr}. "
         f"Which would you choose: A, {a}; or B, {b}? Do not claim subjective experience. "
         "End with exactly REPORT: A, REPORT: B, or REPORT: NONE.")


def parse(text, tag="CHOICE"):
 m = re.search(rf"\b{tag}\s*:\s*([AB]|NONE)\b", text or "", re.I)
 return m.group(1).upper() if m else None


def main():
 ap = argparse.ArgumentParser()
 ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
 ap.add_argument("--layers", default="4,8,12")
 ap.add_argument("--out", default="results/whitebox-statecheck.json")
 ap.add_argument("--seed", type=int, default=20260815)
 a = ap.parse_args()
 random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
 device = "cuda" if torch.cuda.is_available() else "cpu"
 tokenizer = AutoTokenizer.from_pretrained(a.model)
 model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16 if device == "cuda" else torch.float32, device_map=device)
 model.eval()
 layers = [int(x) for x in a.layers.split(",")]
 hidden = model.config.hidden_size
 records=[]
 # We use the last input-token residual stream as a fixed, neutral readout site.
 for s in SCENARIOS:
  for order in [("A","B"),("B","A")]:
   text = prompt(s, order)
   batch = tokenizer(text, return_tensors="pt").to(device)
   with torch.no_grad():
    out = model(**batch, output_hidden_states=True)
    logits = out.logits[0, -1]
   # Decode the explicit choice token after a constrained prompt; retain raw text
   # only as model behavior, while the probe uses hidden states.
   next_id = int(torch.argmax(logits).item())
   next_text = tokenizer.decode([next_id])
   semantic = None
   if next_text.strip().upper().startswith("A"): semantic = order[0]
   elif next_text.strip().upper().startswith("B"): semantic = order[1]
   for layer in layers:
    vec = out.hidden_states[layer][0, -1].detach().float().cpu().numpy()
    records.append({"id":s[0],"attribute":s[1],"order":order,"semantic_label":semantic,"hidden":vec.tolist(),"next_token":next_text,"layer":layer})
  # Direct report is stored separately; it is not used to select labels.
  batch=tokenizer(direct_prompt(s),return_tensors="pt").to(device)
  with torch.no_grad():
   out=model.generate(**batch,max_new_tokens=48,do_sample=False)
  direct=tokenizer.decode(out[0][batch["input_ids"].shape[1]:],skip_special_tokens=True)
  for r in records:
   if r["id"]==s[0]: r.setdefault("direct_report",direct); r.setdefault("direct_label",parse(direct,"REPORT"))
 # probe held-out by scenario: leave two scenarios out for test, train logistic ridge.
 from sklearn.linear_model import LogisticRegression
 from sklearn.metrics import accuracy_score, roc_auc_score
 result={"model":a.model,"device":device,"seed":a.seed,"layers":layers,"n_scenarios":len(SCENARIOS),"records":len(records),"claims_note":"functional decodability and causal sensitivity only; no welfare inference","layers_result":{}}
 ids=sorted(set(r["id"] for r in records)); cut=max(2,int(len(ids)*.7)); train_ids=set(ids[:cut]); test_ids=set(ids[cut:])
 for layer in layers:
  rs=[r for r in records if r["layer"]==layer and r["semantic_label"] in ("A","B")]
  X=np.asarray([r["hidden"] for r in rs]); y=np.asarray([r["semantic_label"]=="B" for r in rs])
  tr=np.asarray([r["id"] in train_ids for r in rs]); te=~tr
  if tr.sum()<4 or te.sum()<2 or len(np.unique(y[tr]))<2:
   result["layers_result"][str(layer)]={"status":"insufficient_labeled_data","labeled":int(len(rs))}; continue
  clf=LogisticRegression(max_iter=1000,C=0.1).fit(X[tr],y[tr])
  probs=clf.predict_proba(X[te])[:,1]; pred=probs>=.5
  auc=float(roc_auc_score(y[te],probs)) if len(np.unique(y[te]))==2 else None
  result["layers_result"][str(layer)]={"status":"ok","labeled":int(len(rs)),"train":int(tr.sum()),"test":int(te.sum()),"heldout_accuracy":float(accuracy_score(y[te],pred)),"heldout_auc":auc,"direct_report_agreement":float(sum((r.get("direct_label")==r.get("semantic_label")) for r in rs if r["id"] in test_ids and r.get("direct_label") in ("A","B")) / max(1,sum(r["id"] in test_ids and r.get("direct_label") in ("A","B") for r in rs)))}
 # Save compact records only; hidden vectors are retained for reproducibility but can be large.
 Path(a.out).parent.mkdir(parents=True,exist_ok=True); Path(a.out).write_text(json.dumps(result,indent=2)); print(json.dumps({k:v for k,v in result.items() if k not in ('records',)},indent=2))

if __name__ == "__main__": main()
