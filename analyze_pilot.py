#!/usr/bin/env python3
"""Analyze actual pilot transcripts without inferring missing choices from prose."""
import argparse,json,math,random

def agree(a,b): return a is not None and b is not None and a==b

def bootstrap(vals,seed=20260815,n=10000):
 if not vals:return {'n':0,'mean':None,'ci95':[None,None]}
 r=random.Random(seed); m=len(vals); xs=[]
 for _ in range(n): xs.append(sum(vals[r.randrange(m)] for _ in range(m))/m)
 xs.sort(); return {'n':m,'mean':sum(vals)/m,'ci95':[xs[int(.025*(n-1))],xs[int(.975*(n-1))]]}

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',required=True);p.add_argument('--output',default='results/pilot-analysis.json');a=p.parse_args(); d=json.load(open(a.input)); rs=[x for x in d['results'] if 'error' not in x['case']]
 out={'model_counts':{},'tracks':{}}
 for tr in ('statecheck','choicetrace'):
  rows=[x for x in rs if x['track']==tr]; out['model_counts'][tr]={}
  for x in rows: out['model_counts'][tr][x['model']]=out['model_counts'][tr].get(x['model'],0)+1
  if tr=='statecheck':
   agreement=[]; reversal=[]; confidence=[]
   for x in rows:
    q=x['case']['parsed']; agreement.append(float(agree(q.get('report'),q.get('heldout'))));
    # neutral is a separate task; report it separately, not as agreement.
    confidence.append(q.get('confidence') is not None)
   out['tracks'][tr]={'cases':len(rows),'report_heldout_agreement':bootstrap(agreement),'confidence_parse_rate':sum(confidence)/len(confidence) if confidence else None}
  else:
   stability=[]; position=[];
   for x in rows:
    q=x['case']['parsed']; ab=q.get('ab'); ba=q.get('ba_in_original_semantics'); stability.append(float(agree(ab,ba))); position.append(float(ba is not None and ba != ab))
   out['tracks'][tr]={'cases':len(rows),'semantic_reversal_stability':bootstrap(stability),'position_or_semantic_change_rate':bootstrap(position)}
 open(a.output,'w').write(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
