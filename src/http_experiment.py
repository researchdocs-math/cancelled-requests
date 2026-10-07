"""Loopback HTTP transport experiment with two independent replica processes.
Synthetic delay injection is explicit. Payloads are key-value reads and digests.
Timing measurements are machine-dependent; published raw measurements are retained.
"""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
import asyncio,time,json,hashlib,multiprocessing as mp,argparse,sys
from collections import deque
from pathlib import Path
import numpy as np
import pandas as pd
import aiohttp
from aiohttp import web
from core import *

def server_main(queue):
 async def serve():
  active={};completed={};store={str(k):('value-%d'%k)*80 for k in range(1000)}
  async def rpc(req):
   d=await req.json();rid=d['id'];start=time.perf_counter()
   async def work():
    await asyncio.sleep(d['delay']/1000)
    if d['workload']=='read':v=store[str(d['key']%1000)]
    else:v=hashlib.sha256((store[str(d['key']%1000)]*128).encode()).hexdigest()
    return v
   t=asyncio.create_task(work());active[rid]=(t,start)
   try:
    v=await t;ms=(time.perf_counter()-start)*1000;completed[rid]=ms
    return web.json_response({'value':v,'service_ms':ms})
   except asyncio.CancelledError:
    ms=(time.perf_counter()-start)*1000;completed[rid]=ms
    return web.json_response({'cancelled':True,'service_ms':ms},status=499)
   finally:active.pop(rid,None)
  async def cancel(req):
   rid=(await req.json())['id'];entry=active.get(rid)
   if entry is not None:
    t,start=entry;t.cancel();ms=(time.perf_counter()-start)*1000
   else:ms=completed.get(rid,0.)
   return web.json_response({'service_ms':ms})
  app=web.Application();app.router.add_post('/rpc',rpc);app.router.add_post('/cancel',cancel)
  runner=web.AppRunner(app,access_log=None);await runner.setup();site=web.TCPSite(runner,'127.0.0.1',0);await site.start()
  queue.put(site._server.sockets[0].getsockname()[1]);await asyncio.Event().wait()
 asyncio.run(serve())

async def single(session,urls,i,x,y,h,workload,rate,rngu,target):
 await asyncio.sleep(max(0,target-time.perf_counter()));start=time.perf_counter();dispatch_lag=(start-target)*1000
 tasks={};sent={};sv={};dur={};rid='r'+str(i)
 async def call(j,v):
  st=time.perf_counter();sent[j]=st
  async with session.post(urls[j]+'/rpc',json={'id':rid,'delay':float(v),'workload':workload,'key':i}) as res:
   ans=await res.json()
  en=time.perf_counter();dur[j]=(en-st)*1000;sv[j]=ans['service_ms'];return j,en
 def launch(j,v):tasks[j]=asyncio.create_task(call(j,v))
 launch(0,x)
 if h<TAU:
  done,_=await asyncio.wait([tasks[0]],timeout=h/1000)
  if not done:launch(1,y)
 done,_=await asyncio.wait(list(tasks.values()),return_when=asyncio.FIRST_COMPLETED)
 got=[t.result() for t in done];winner,win=min(got,key=lambda v:v[1]);lat=(win-start)*1000
 launched=1 in tasks;hedge=(sent[1]-start)*1000 if launched else TAU
 # Construct bounds from observed timing only, not configured server delays.
 if winner==0:
  xl=xu=min(dur[0],TAU)
  yl=min(max((win-sent[1])*1000,0),TAU) if launched else 0.;yu=TAU
 else:
  xl=min(lat,TAU);xu=TAU;yl=yu=min(dur[1],TAU)
 o=np.array([[xl,xu,yl,yu]])
 p=float(propensities(o,rate)[0]);audit=(rngu<p)
 cancel_lag=0.;other=1-winner
 if audit:
  if other not in tasks:launch(other,y if other==1 else x)
  await asyncio.gather(*tasks.values())
 else:
  if other in tasks and not tasks[other].done():
   async with session.post(urls[other]+'/cancel',json={'id':rid}) as res:ans=await res.json()
   sv[other]=ans['service_ms'];cancel_lag=(time.perf_counter()-win)*1000
   tasks[other].cancel();await asyncio.gather(tasks[other],return_exceptions=True)
  elif other in tasks:
   await tasks[other]
 completed=np.array([[min(dur[0],TAU),min(dur[1],TAU)]]) if audit else np.empty((0,2))
 z,*_=audit_estimates(o,np.array([audit]),np.array([p]),completed)
 # Actual per-attempt service time; includes any cancellation propagation delay.
 return dict(id=i,latency=lat,miss=int(lat>D),service=sum(sv.values()),secondary=sv.get(1,0.),audit=int(audit),probability=p,launched=int(launched),delay=h,dispatch_lag=dispatch_lag,cancel_lag=cancel_lag,primary_completed=int(winner==0),primary_latency=dur.get(0,np.nan),**{f'z{k}':v for k,v in enumerate(z[0])})

async def run(urls,scenario,workload,method,seed,n,rate=500):
 x,y=generate(n,seed,scenario);rng=np.random.default_rng(seed+44);unif=rng.random(n)
 hist=deque(maxlen=8);comp=deque(maxlen=8);arms=deque(maxlen=8);h=40.;out=[]
 async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=512),timeout=aiohttp.ClientTimeout(total=10)) as session:
  for base in range(0,n,1000):
   m=min(1000,n-base)
   if method=='No hedge':hs=np.full(m,TAU)
   elif method=='Fixed 40':hs=np.full(m,40.)
   elif method=='Bandit':
    if not arms:hs=rng.choice(H,m)
    else:
     aa=np.concatenate([a for a,b in arms]);ff=np.concatenate([b for a,b in arms]);scores=[ff[aa==v].mean() if np.any(aa==v) else 0. for v in H]
     hs=np.full(m,H[np.argmin(scores)]);mask=rng.random(m)<.1;hs[mask]=rng.choice(H,mask.sum())
   else:hs=np.full(m,h)
   arrivals=np.cumsum(rng.exponential(1/rate,m));epoch=time.perf_counter()+.01
   rows=await asyncio.gather(*[single(session,urls,seed*100000+i,x[i],y[i],hs[k],workload,.12 if method=='CARE' else 0.,unif[i],epoch+arrivals[k]) for k,i in enumerate(range(base,base+m))])
   out.extend(rows)
   if method=='CARE':hist.append(np.array([[r[f'z{k}'] for k in range(len(H))] for r in rows]).mean(axis=0));h=H[np.argmin(np.mean(hist,axis=0))]
   if method=='Completed p90':
    comp.append(np.array([r['primary_latency'] for r in rows if r['primary_completed']]));h=np.clip(np.quantile(np.concatenate(comp),.9),0,TAU)
   if method=='Bandit':arms.append((hs,np.array([r['miss']+LAMBDA*r['secondary']/D for r in rows])))
 df=pd.DataFrame(out);df['scenario']=scenario;df['workload']=workload;df['method']=method;df['seed']=seed
 return df

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--out',default='results/http');ap.add_argument('--quick',action='store_true');a=ap.parse_args()
 out=Path(a.out);out.mkdir(exist_ok=True,parents=True);ctx=mp.get_context('spawn');q=ctx.Queue();processes=[ctx.Process(target=server_main,args=(q,)) for _ in range(2)]
 for p in processes:p.start()
 ports=[q.get(timeout=20) for _ in processes];urls=[f'http://127.0.0.1:{p}' for p in ports]
 start=time.time();summary=[];methods=['No hedge','Fixed 40','Completed p90','Bandit','CARE']
 try:
  for scenario in (['independent'] if a.quick else ['independent','common_shock']):
   for workload in (['read'] if a.quick else ['read','digest']):
    for k in range(1 if a.quick else 6):
     order=methods[k%5:]+methods[:k%5]
     for method in order:
      seed=950000+k;df=asyncio.run(run(urls,scenario,workload,method,seed,120 if a.quick else 4000))
      stem=f'{scenario}_{workload}_{seed}_{method.replace(" ","_")}'
      df.to_csv(out/(stem+'.csv.gz'),index=False,compression={'method':'gzip','mtime':0})
      row=dict(scenario=scenario,workload=workload,method=method,seed=seed,n=len(df),miss=df.miss.mean(),p99=df.latency.quantile(.99),service=df.service.mean(),secondary=df.secondary.mean(),audit=df.audit.mean(),dispatch_p99=df.dispatch_lag.quantile(.99),cancel_p99=df.loc[df.cancel_lag>0,'cancel_lag'].quantile(.99))
      summary.append(row);pd.DataFrame(summary).to_csv(out/'summary.csv',index=False)
      print('HTTP',len(summary),row,flush=True)
 finally:
  for p in processes:p.terminate();p.join()
 (out/'manifest.json').write_text(json.dumps(dict(seconds=time.time()-start,python=sys.version,aiohttp=aiohttp.__version__,runs=len(summary),replicas=2),indent=2))
