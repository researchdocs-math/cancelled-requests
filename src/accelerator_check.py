"""Optional accelerator cross-check; primary scientific estimates remain float64."""
import argparse,json,time
from pathlib import Path
import numpy as np
from core import *
def main():
 import torch
 ap=argparse.ArgumentParser();ap.add_argument('--out',default='results/accelerator.json');a=ap.parse_args()
 device='mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
 rows=[]
 for s in ['independent','correlated','common_shock','asymmetric','pareto']:
  x,y=generate(1000000,4545,s);ref=outcome(x[:,None],y[:,None],H)[0].mean(axis=0)
  st=time.perf_counter();tx=torch.tensor(x,dtype=torch.float32,device=device)[:,None];ty=torch.tensor(y,dtype=torch.float32,device=device)[:,None];th=torch.tensor(H,dtype=torch.float32,device=device)[None,:]
  L=torch.minimum(tx,th+ty);B=torch.minimum(ty,torch.clamp(tx-th,min=0));f=(L>D).float()+LAMBDA*B/D
  got=f.mean(dim=0).cpu().numpy();elapsed=time.perf_counter()-st
  err=float(np.max(np.abs(ref-got)))
  if err>1e-5:raise AssertionError((s,err))
  rows.append(dict(scenario=s,requests=len(x),candidate_evaluations=len(x)*len(H),max_mean_error=err,accelerator_seconds=elapsed))
 p=Path(a.out);p.parent.mkdir(exist_ok=True,parents=True);p.write_text(json.dumps(dict(backend=device,torch=torch.__version__,dtype='float32',cpu_reference_dtype='float64',tolerance=1e-5,results=rows),indent=2));print(p.read_text())
if __name__=='__main__':main()
