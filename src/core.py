"""Cancellation-aware policy evaluation for capped, exogenous paired latencies.
All durations are milliseconds. Controllers receive interval records, never hidden
potential times, except on explicitly randomized completion audits.
"""
import numpy as np
H=np.array([0.,10.,20.,30.,40.,60.,80.,300.])
D=80.; TAU=300.; LAMBDA=.12

def outcome(x,y,h,lam=LAMBDA):
    l=np.minimum(x,h+y)
    b=np.minimum(y,np.maximum(x-h,0.))
    return (l>D).astype(float)+lam*b/D,l,b

def record(x,y,h):
    l=np.minimum(x,h+y)
    launched=x>h
    primary=x<=h+y
    xl=np.where(primary,x,np.nextafter(l,np.inf))
    xu=np.where(primary,x,TAU)
    yl=np.where(launched,np.where(primary,np.nextafter(l-h,np.inf),y),0.)
    yu=np.where(launched&~primary,y,TAU)
    xl=np.minimum(xl,TAU);yl=np.minimum(yl,TAU)
    return np.column_stack([xl,xu,yl,yu]),l,launched,primary

def bounds(o,lam=LAMBDA):
    lo=outcome(o[:,0,None],o[:,2,None],H[None,:],lam)[0]
    hi=outcome(o[:,1,None],o[:,3,None],H[None,:],lam)[0]
    return lo,hi

def audit_estimates(o,a,p,complete,lam=LAMBDA):
    lo,hi=bounds(o,lam); mid=(lo+hi)/2
    z=mid.copy(); ipw=np.zeros_like(mid)
    ix=np.flatnonzero(a)
    if len(ix):
        f=outcome(complete[:,0,None],complete[:,1,None],H[None,:],lam)[0]
        z[ix]+=(f-mid[ix])/p[ix,None]
        ipw[ix]=f/p[ix,None]
    # On zero-width records the loss is fully known and no audit is necessary.
    known=(hi==lo)
    ipw[known]=mid[known]
    return z,ipw,lo,hi

def propensities(o,rate=.12,lam=LAMBDA):
    lo,hi=bounds(o,lam)
    w=np.max(hi-lo,axis=1)
    return np.minimum(.5,rate*w)

def generate(n,seed,scenario):
    rng=np.random.default_rng(seed)
    z=rng.standard_normal((n,3)); u=rng.random((n,3))
    # Independent request pairs across rows; dependence within a pair is explicit.
    if scenario=='independent':
        x=18*np.exp(.45*z[:,0]); y=18*np.exp(.45*z[:,1])
        x*=np.where(u[:,0]<.10,7.,1.); y*=np.where(u[:,1]<.10,7.,1.)
    elif scenario=='correlated':
        x=30*np.exp(.95*z[:,0]); y=30*np.exp(.95*(.9*z[:,0]+np.sqrt(1-.9**2)*z[:,1]))
    elif scenario=='common_shock':
        shock=np.where(u[:,2]<.10,8.,1.)
        x=18*np.exp(.4*z[:,0])*shock; y=18*np.exp(.4*z[:,1])*shock
    elif scenario=='asymmetric':
        x=18*np.exp(.45*z[:,0])*np.where(u[:,0]<.16,8.,1.)
        y=32*np.exp(.65*z[:,1])*np.where(u[:,1]<.03,6.,1.)
    elif scenario=='pareto':
        x=12*(1-u[:,0])**(-1/1.4); y=12*(1-u[:,1])**(-1/1.4)
    elif scenario=='switching':
        # Four equal regimes: independent, common shock, reversed asymmetry, recovery.
        x=np.empty(n); y=np.empty(n)
        for k,ix in enumerate(np.array_split(np.arange(n),4)):
            xx,yy=generate(len(ix),seed+10000*(k+1),['independent','common_shock','asymmetric','independent'][k])
            if k==2: xx,yy=yy,xx
            x[ix]=xx; y[ix]=yy
    else: raise ValueError(scenario)
    return np.minimum(x,TAU),np.minimum(y,TAU)

def km_grid(times,events,grid):
    # Product limit; unobserved remaining mass is assigned to the administrative cap.
    order=np.argsort(times,kind='stable'); t=times[order]; e=events[order]
    u,first,count=np.unique(t,return_index=True,return_counts=True)
    deaths=np.add.reduceat(e.astype(float),first); risk=len(t)-first
    s=np.cumprod(1-deaths/risk)
    idx=np.searchsorted(u,grid,side='right')-1
    return np.where(idx>=0,s[np.maximum(idx,0)],1.)

def km_scores(history,lam=LAMBDA):
    o=np.concatenate(history)
    grid=np.linspace(0,TAU,601)
    sx=km_grid(o[:,0],o[:,0]==o[:,1],grid)
    seen=o[:,2]>0
    if seen.sum()<20:return np.full(len(H),1.)
    sy=km_grid(o[seen,2],o[seen,2]==o[seen,3],grid)
    # Marginal independence is an explicit baseline assumption, not a property of CARE.
    scores=[]
    for h in H:
        miss=np.interp(D,grid,sx)*(np.interp(D-h,grid,sy) if h<D else 1.)
        t=grid[grid<=TAU-h]
        cost=np.trapz(np.interp(h+t,grid,sx)*np.interp(t,grid,sy),t) if len(t)>1 else 0.
        scores.append(miss+lam*cost/D)
    return np.array(scores)
