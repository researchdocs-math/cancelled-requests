"""Reproduce numerical experiments and reanalyze recorded measurements."""
import argparse, hashlib, json, os, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
ENV = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))

def command(*args):
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, env=ENV, check=True)

def compare_csv(expected, actual):
    a, b = pd.read_csv(expected), pd.read_csv(actual)
    assert list(a.columns) == list(b.columns), (expected.name, 'columns')
    assert a.shape == b.shape, (expected.name, a.shape, b.shape)
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]):
            np.testing.assert_allclose(a[c], b[c], rtol=1e-9, atol=1e-9, equal_nan=True, err_msg=f'{expected.name}: {c}')
        else:
            assert a[c].equals(b[c]), (expected.name, c)

def ci(v):
    v = np.asarray(v)
    half = float(t.ppf(.975, len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)))
    return {'mean': float(v.mean()), 'lower': float(v.mean()-half), 'upper': float(v.mean()+half), 'replicates': len(v)}

def reanalyze():
    p = ROOT / 'results'
    e, a, q, ht = [pd.read_csv(p / f) for f in ['estimation.csv', 'policies.csv', 'queue.csv', 'http/summary.csv']]
    est = []
    for name, g in e[e.n == 50000].groupby('scenario'):
        means = g.groupby('method').rmse.mean()
        est.append({'scenario': name, 'mean_rmse': means.to_dict(), 'CARE_reduction_vs_audit_only_percent': float(100*(1-means['CARE']/means['Audit only'])), 'audit_percent': float(100*g[g.method=='CARE'].audit_rate.mean())})
    adaptive = []
    rng = np.random.default_rng(81622)
    for name, g in a.groupby('scenario'):
        v = g.pivot(index='seed', columns='method', values='operational_loss')
        difference = (v['CARE']-v['Bandit']).to_numpy()
        bootstrap = np.mean(rng.choice(difference, (10000, len(difference)), replace=True), axis=1)
        adaptive.append({'scenario': name, 'CARE_minus_bandit': ci(difference), 'bootstrap_95_interval': np.quantile(bootstrap,[.025,.975]).tolist(), 'policy_means': g.groupby('method')[['miss','p99','work','audit','operational_loss']].mean().to_dict(orient='index')})
    http = []
    for (scenario,payload,method), g in ht.groupby(['scenario','workload','method']):
        http.append({'scenario': scenario, 'payload': payload, 'method': method, **{c: ci(g[c]) for c in ['miss','p99','service','audit']}})
    queues = []
    for key, g in q.groupby(['scenario','rho','lag','method']):
        queues.append(dict(zip(['scenario','offered_load','cancel_lag','method'],key),p99=ci(g.p99),work=ci(g.work)))
    return {'estimation_at_50000': est, 'adaptive': adaptive, 'http': http, 'queue': queues}

def verify():
    checked = 0
    for line in (ROOT / 'checksums.sha256').read_text().splitlines():
        digest, rel = line.split('  ',1)
        assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == digest, f'Integrity mismatch: {rel}'
        checked += 1
    command('-m','unittest','discover','-s','tests','-v')
    sys.path.insert(0,str(ROOT/'src'))
    import core
    config = json.loads((ROOT/'config.json').read_text())
    np.testing.assert_array_equal(core.H,config['candidate_delays'])
    assert core.D==config['deadline'] and core.TAU==config['attempt_cap'] and core.LAMBDA==config['default_work_price']
    summary = pd.read_csv(ROOT/'results/http/summary.csv')
    assert len(summary)==config['http']['runs']
    total = 0
    for row in summary.to_dict('records'):
        stem = f"{row['scenario']}_{row['workload']}_{row['seed']}_{row['method'].replace(' ','_')}"
        d = pd.read_csv(ROOT/'results/http'/f'{stem}.csv.gz')
        assert len(d)==row['n']==config['http']['requests_per_run']
        total += len(d)
        actual = dict(miss=d.miss.mean(),p99=d.latency.quantile(.99),service=d.service.mean(),secondary=d.secondary.mean(),audit=d.audit.mean(),dispatch_p99=d.dispatch_lag.quantile(.99),cancel_p99=d.loc[d.cancel_lag>0,'cancel_lag'].quantile(.99))
        for key, value in actual.items():
            np.testing.assert_allclose(value,row[key],rtol=1e-9,atol=1e-9,equal_nan=True,err_msg=stem+': '+key)
        assert d.id.is_unique
        assert np.all(d.service>=0) and np.all(d.secondary>=0)
        assert np.all(d.service+1e-9>=d.secondary)
        assert np.all((d.audit==0)|(d.probability>0))
    output = {'integrity_files_checked': checked, 'http_runs_reconstructed': len(summary), 'http_requests_reconstructed': total, 'numerical_results': reanalyze()}
    (ROOT/'verification.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(f'PASS: {checked} hashes; ten tests; {total:,} measured requests in {len(summary)} runs. Numerical comparisons: verification.json')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['verify','simulation','http','accelerator'],default='verify')
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if args.workers<1:parser.error('--workers must be positive')
    if args.mode=='verify':verify()
    elif args.mode=='simulation':
        target=ROOT/'recomputed';target.mkdir(exist_ok=True)
        for mode in ['full','estimation','sensitivity']:
            command('src/run_experiments.py','--mode',mode,'--workers',args.workers,'--out',target)
        command('src/queue_stress.py','--workers',args.workers,'--out',target)
        for filename in ['policies.csv','curves.csv','estimation.csv','sensitivity.csv','queue.csv']:
            compare_csv(ROOT/'results'/filename,target/filename)
            print('PASS:',filename,flush=True)
    elif args.mode=='http':command('src/http_experiment.py','--out','recomputed/http')
    else:command('src/accelerator_check.py','--out','recomputed/accelerator.json')

if __name__=='__main__':main()
