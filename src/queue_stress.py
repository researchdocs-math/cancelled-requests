"""Independent FCFS queueing stress test of frozen service-time-trained delays.
Completion audits retain the loser or launch a post-response backup. Queueing
changes response potentials; this experiment is outside the estimator theorem.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import heapq,argparse,json,time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
from core import *

def simulate(x,y,arrival,server,h,audit,lag):
 n=len(x);q=[deque(),deque()];busy=[None,None]; heap=[];serial=0;last_activity=0.0
 state=np.zeros((n,2),dtype=np.int8);beg=np.zeros((n,2));spent=np.zeros((n,2));finish=np.full(n,np.nan);auditflag=np.zeros(n,dtype=bool)
 def event(t,kind,i,j=0):
  nonlocal serial
  serial+=1;heapq.heappush(heap,(t,serial,kind,i,j))
 def start(s,t):
  if busy[s] is not None:return
  while q[s]:
   i,j=q[s].popleft()
   if state[i,j]!=1:continue
   state[i,j]=2;beg[i,j]=t;busy[s]=(i,j);event(t+(x[i] if j==0 else y[i]),'done',i,j);break
 def launch(i,j,t):
  if state[i,j]!=0:return
  s=server[i] if j==0 else 1-server[i]
  state[i,j]=1;q[s].append((i,j));start(s,t)
 def cancel(i,j,t):
  nonlocal last_activity
  if state[i,j] not in (1,2):return
  s=server[i] if j==0 else 1-server[i]
  if state[i,j]==2:
   spent[i,j]+=t-beg[i,j];busy[s]=None;last_activity=max(last_activity,t)
  state[i,j]=4;start(s,t)
 for i,t in enumerate(arrival):event(t,'arrival',i)
 while heap:
  t,_,kind,i,j=heapq.heappop(heap)
  if kind=='arrival':
   launch(i,0,t)
   if h<TAU:event(t+h,'hedge',i)
  elif kind=='hedge':
   if np.isnan(finish[i]):launch(i,1,t)
  elif kind=='cancel':cancel(i,j,t)
  elif kind=='done':
   if state[i,j]!=2:continue
   s=server[i] if j==0 else 1-server[i]
   if busy[s]!=(i,j):raise AssertionError('busy mismatch')
   spent[i,j]+=t-beg[i,j];state[i,j]=3;busy[s]=None;last_activity=max(last_activity,t)
   if np.isnan(finish[i]):
    finish[i]=t;other=1-j
    if audit[i]:
     auditflag[i]=True
     if state[i,other]==0:launch(i,other,t)
    else:
     if state[i,other] in (1,2):event(t+lag,'cancel',i,other)
   start(s,t)
 if np.isnan(finish).any():raise AssertionError('unfinished requests')
 lat=finish-arrival;sl=slice(min(2000,n//10),None)
 return dict(miss=np.mean(lat[sl]>D),p99=np.quantile(lat[sl],.99),latency=lat[sl].mean(),work=spent[sl].sum(axis=1).mean(),backup=spent[sl,1].mean(),audit=auditflag[sl].mean(),utilization=spent.sum()/2/max(last_activity,arrival[-1],1e-12),last_latency=lat[-1000:].mean())

def one(arg):
 scenario,seed,rho,lag,n=arg
 xt,yt=generate(25000,seed+1000000,scenario);o,*_=record(xt,yt,40.);p=propensities(o)
 rr=np.random.default_rng(seed+921);a=rr.random(len(xt))<p
 z,*_=audit_estimates(o,a,p,np.column_stack([xt[a],yt[a]]));chosen=float(H[np.argmin(z.mean(axis=0))])
 x,y=generate(n,seed,scenario);r=np.random.default_rng(seed+9121)
 # Offered utilization calibrated with an independent 25,000-request sample.
 arrival=np.cumsum(r.exponential(xt.mean()/(2*rho),n));server=r.integers(0,2,n);u=r.random(n)
 out=[]
 for method,h,prob in [('No hedge',300.,0.),('Fixed 40',40.,0.),('Immediate',0.,0.),('CARE frozen',chosen,0.),('CARE + audits',chosen,.02)]:
  m=simulate(x,y,arrival,server,h,u<prob,lag)
  out.append(dict(scenario=scenario,seed=seed,rho=rho,lag=lag,method=method,delay=h,n=n,**m))
 return out
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=8);ap.add_argument('--out',default='results');ap.add_argument('--quick',action='store_true');args=ap.parse_args();start=time.time()
 jobs=[(s,800000+k,r,l,6000 if args.quick else 30000) for s in ['independent','correlated','common_shock'] for k in range(2 if args.quick else 12) for r in [.3,.6,.8,.95] for l in [0.,5.]]
 rows=[]
 with ProcessPoolExecutor(args.workers) as ex:
  for i,r in enumerate(ex.map(one,jobs)):
   rows.extend(r)
   if i%24==0:print('Queue replicate',i+1,'/',len(jobs),flush=True)
 out=Path(args.out);out.mkdir(exist_ok=True,parents=True);pd.DataFrame(rows).to_csv(out/'queue.csv',index=False)
 (out/'queue_manifest.json').write_text(json.dumps(dict(seconds=time.time()-start,jobs=len(jobs),workers=args.workers),indent=2));print('Queue seconds',time.time()-start)
