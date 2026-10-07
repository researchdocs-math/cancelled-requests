"""Seeded Monte Carlo experiments, parallelized across independent replicates."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import argparse,json,time,platform,sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from collections import deque
import numpy as np
import pandas as pd
from core import *
SCENARIOS=['independent','correlated','common_shock','asymmetric','pareto','switching']
POLICIES=['No hedge','Fixed 40','Completed p90','KM independent','Bandit','Audit only','CARE','Full information']

def policy_run(args):
 scenario,seed,n,block,rate,lam=args
 x,y=generate(n,seed,scenario)
 allrows=[]; curves=[]
 for method in POLICIES:
  rng=np.random.default_rng(seed+88113)
  hist=deque(maxlen=8); raw=deque(maxlen=8); armhist=deque(maxlen=8)
  records=[]; h=40.
  for j,start in enumerate(range(0,n,block)):
   xx=x[start:start+block];yy=y[start:start+block]; m=len(xx)
   if method=='No hedge':hs=np.full(m,TAU)
   elif method=='Fixed 40':hs=np.full(m,40.)
   elif method=='Bandit':
    if not armhist:hs=rng.choice(H,m)
    else:
     aa=np.concatenate([v[0] for v in armhist]); ff=np.concatenate([v[1] for v in armhist])
     scores=np.array([ff[aa==v].mean() if np.any(aa==v) else 0. for v in H])
     hs=np.full(m,H[np.argmin(scores)]); explore=rng.random(m)<.1;hs[explore]=rng.choice(H,explore.sum())
   else:hs=np.full(m,h)
   f,l,b=outcome(xx,yy,hs,lam);o,_,launched,primary=record(xx,yy,hs)
   p=np.zeros(m);a=np.zeros(m,dtype=bool);extra=np.zeros(m)
   if method in ['CARE','Audit only']:
    p=propensities(o,rate,lam);a=rng.random(m)<p
    # Hidden paired durations cross the estimator interface only on audited rows.
    z,ipw,_,_=audit_estimates(o,a,p,np.column_stack([xx[a],yy[a]]),lam)
    hist.append((z if method=='CARE' else ipw).mean(axis=0))
    h=H[np.argmin(np.mean(hist,axis=0))]
    extra[a]=xx[a]+yy[a]-l[a]-b[a]
   elif method=='Full information':
    hist.append(outcome(xx[:,None],yy[:,None],H,lam)[0].mean(axis=0));h=H[np.argmin(np.mean(hist,axis=0))]
   elif method=='Completed p90':
    raw.append(xx[primary]);vals=np.concatenate(raw)
    h=np.clip(np.quantile(vals,.9),0,TAU) if len(vals) else 40.
   elif method=='KM independent':
    raw.append(o);h=H[np.argmin(km_scores(raw,lam))]
   elif method=='Bandit':armhist.append((hs,f))
   rows=np.column_stack([l,b,extra,a,launched,f,hs]); records.append(rows)
   curves.append(dict(scenario=scenario,seed=seed,method=method,block=j,miss=np.mean(l>D),loss=f.mean(),audit=a.mean(),audit_work=extra.mean(),work=(l+b+extra).mean(),delay=hs.mean()))
  r=np.concatenate(records)
  allrows.append(dict(scenario=scenario,seed=seed,method=method,n=n,miss=np.mean(r[:,0]>D),p95=np.quantile(r[:,0],.95),p99=np.quantile(r[:,0],.99),latency=r[:,0].mean(),backup=r[:,1].mean(),audit_work=r[:,2].mean(),audit=r[:,3].mean(),launch=r[:,4].mean(),loss=r[:,5].mean(),operational_loss=np.mean(r[:,5]+lam*r[:,2]/D),work=np.mean(r[:,0]+r[:,1]+r[:,2]),primary_mean=x.mean(),rate=rate,lam=lam))
 return allrows,curves

def estimation(args):
 scenario,seed,n,rate=args
 x,y=generate(n,seed,scenario);o,*_=record(x,y,40.)
 p=propensities(o,rate);a=np.random.default_rng(seed+87113).random(n)<p
 z,ipw,lo,hi=audit_estimates(o,a,p,np.column_stack([x[a],y[a]]))
 truth=outcome(x[:,None],y[:,None],H)[0].mean(axis=0)
 known=(hi==lo)
 sn=lo.copy()
 for j in range(len(H)):
  unknown=~known[:,j]; selected=unknown&a
  if selected.any():
   fs=outcome(x[selected],y[selected],H[j])[0]
   sn[unknown,j]=np.sum(fs/p[selected])/np.sum(1/p[selected])
  else:sn[unknown,j]=(lo[unknown,j]+hi[unknown,j])/2
 allscores={'Self-normalized audit':sn.mean(axis=0),'CARE':z.mean(axis=0),'Audit only':ipw.mean(axis=0),'Midpoint':((lo+hi)/2).mean(axis=0),'Upper bound':hi.mean(axis=0),'KM independent':km_scores([o])}
 out=[]
 for k,s in allscores.items():
  out.append(dict(scenario=scenario,seed=seed,n=n,method=k,audits=int(a.sum()),audit_rate=a.mean(),rmse=np.sqrt(np.mean((s-truth)**2)),max_error=np.max(np.abs(s-truth)),regret=truth[np.argmin(s)]-truth.min(),width=np.mean(hi-lo)))
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['pilot','full','estimation','sensitivity'],default='full');ap.add_argument('--workers',type=int,default=8);ap.add_argument('--out',default='results');args=ap.parse_args()
 out=Path(args.out);out.mkdir(parents=True,exist_ok=True);start=time.time()
 if args.mode in ['pilot','full']:
  seeds=range(4) if args.mode=='pilot' else range(32)
  n=16000 if args.mode=='pilot' else 80000
  jobs=[(s,(1000 if args.mode=='pilot' else 100000)+seed,n,1000,.12,.12) for s in SCENARIOS for seed in seeds]
  rows=[];curves=[]
  with ProcessPoolExecutor(args.workers) as ex:
   for k,(r,c) in enumerate(ex.map(policy_run,jobs)):
    rows.extend(r);curves.extend(c)
    if k%12==0:print('Completed policy replicates',k+1,'/',len(jobs),flush=True)
  pd.DataFrame(rows).to_csv(out/'policies.csv',index=False);pd.DataFrame(curves).to_csv(out/'curves.csv',index=False)
 elif args.mode=='estimation':
  jobs=[(s,50000+seed,n,.12) for s in SCENARIOS[:-1] for n in [2000,10000,50000] for seed in range(96)]
  rows=[]
  with ProcessPoolExecutor(args.workers) as ex:
   for k,r in enumerate(ex.map(estimation,jobs)):
    rows.extend(r)
    if k%100==0: print('Completed estimator replicates',k+1,'/',len(jobs),flush=True)
  pd.DataFrame(rows).to_csv(out/'estimation.csv',index=False)
 else:
  jobs=[(s,70000+seed,40000,1000,rate,lam) for s in ['independent','correlated','common_shock'] for rate in [.03,.06,.12,.24] for lam in [.06,.12,.24] for seed in range(12)]
  rows=[]
  with ProcessPoolExecutor(args.workers) as ex:
   for k,(r,c) in enumerate(ex.map(policy_run,jobs)):
    rows.extend(r)
    if k%36==0:print('Completed sensitivity replicates',k+1,'/',len(jobs),flush=True)
  pd.DataFrame(rows).to_csv(out/'sensitivity.csv',index=False)
 (out/f'{args.mode}_manifest.json').write_text(json.dumps(dict(mode=args.mode,seconds=time.time()-start,workers=args.workers,python=sys.version,numpy=np.__version__,os=platform.system(),machine=platform.machine(),jobs=len(jobs)),indent=2))
 print('Finished in',round(time.time()-start,2),'seconds',flush=True)
if __name__=='__main__':main()
