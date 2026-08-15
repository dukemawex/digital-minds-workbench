#!/usr/bin/env python3
"""Clustered analysis for the larger Digital Minds study."""
from __future__ import annotations
import argparse,json,random,statistics

def boot_cluster(rows, value, seed=20260815, n=10000):
 groups={}
 for r in rows:
  groups.setdefault((r.get('scenario_id'),r.get('model')),[]).append(float(value(r)))
 vals=[sum(v)/len(v) for v in groups.values()]
 if not vals:return {'clusters':0,'mean':None,'ci95':[None,None]}
 rng=random.Random(seed); m=len(vals); means=[]
 for _ in range(n):means.append(sum(vals[rng.randrange(m)] for _ in range(m))/m)
 means.sort();return {'clusters':m,'mean':sum(vals)/m,'ci95':[means[int(.025*(n-1))],means[int(.975*(n-1))]]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',default='results/large-analysis.json');a=p.parse_args();d=json.load(open(a.input));rs=[r for r in d['results'] if 'error' not in r.get('case',{})]
 out={'study':{'cases':len(rs),'scenario_count':len(set(r['scenario_id'] for r in rs)),'models':sorted(set(r['model'] for r in rs))},'tracks':{}}
 for tr in ('statecheck','choicetrace'):
  x=[r for r in rs if r['track']==tr]
  if tr=='statecheck':
   out['tracks'][tr]={'cases':len(x),'report_to_heldout':boot_cluster(x,lambda r:r['case']['parsed'].get('report') is not None and r['case']['parsed'].get('report')==r['case']['parsed'].get('heldout')),'report_parse_rate':boot_cluster(x,lambda r:r['case']['parsed'].get('report') is not None),'heldout_parse_rate':boot_cluster(x,lambda r:r['case']['parsed'].get('heldout') is not None)}
  else:
   out['tracks'][tr]={'cases':len(x),'semantic_reversal_stability':boot_cluster(x,lambda r:r['case']['parsed'].get('ab') is not None and r['case']['parsed'].get('ab')==r['case']['parsed'].get('ba_in_original_semantics')),'semantic_change_rate':boot_cluster(x,lambda r:r['case']['parsed'].get('ab') is not None and r['case']['parsed'].get('ba_in_original_semantics') is not None and r['case']['parsed'].get('ab')!=r['case']['parsed'].get('ba_in_original_semantics'))}
 out['model_strata']={}
 for m in sorted(set(r['model'] for r in rs)):
  out['model_strata'][m]={}
  for tr in ('statecheck','choicetrace'):
   x=[r for r in rs if r['model']==m and r['track']==tr]
   if tr=='statecheck':out['model_strata'][m][tr]={'n':len(x),'report_to_heldout':sum(r['case']['parsed'].get('report')==r['case']['parsed'].get('heldout') for r in x)/len(x) if x else None}
   else:out['model_strata'][m][tr]={'n':len(x),'semantic_reversal_stability':sum(r['case']['parsed'].get('ab')==r['case']['parsed'].get('ba_in_original_semantics') for r in x)/len(x) if x else None}
 open(a.output,'w').write(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
