import unittest
import numpy as np
from core import *
class CoreTests(unittest.TestCase):
 def test_interval_contains_loss(self):
  for s in ['independent','correlated','common_shock','asymmetric','pareto']:
   x,y=generate(10000,73,s)
   for h in H:
    o,_,_,_=record(x,y,h); lo,hi=bounds(o)
    f=outcome(x[:,None],y[:,None],H[None,:])[0]
    self.assertTrue(np.all(f>=lo-1e-10)); self.assertTrue(np.all(f<=hi+1e-10))
 def test_cap_ties(self):
  o,*_=record(np.array([TAU]),np.array([TAU]),0.)
  self.assertTrue(np.all(o[:,0]<=o[:,1])); self.assertTrue(np.all(o[:,2]<=o[:,3]))
 def test_cost_and_completion(self):
  x=np.array([10.,100.,100.]);y=np.array([20.,10.,200.]);h=40.
  _,l,b=outcome(x,y,h)
  np.testing.assert_array_equal(l,[10,50,100]);np.testing.assert_array_equal(b,[0,10,60])
  self.assertTrue(np.all(x+y-l-b>=0))
 def test_no_hidden_truth_access(self):
  x,y=generate(100,1,'independent');o,*_=record(x,y,40)
  a=np.zeros(100,dtype=bool);p=propensities(o)
  z,_,lo,hi=audit_estimates(o,a,p,np.empty((0,2)))
  np.testing.assert_allclose(z,(lo+hi)/2)
 def test_exact_conditional_unbiasedness(self):
  x,y=generate(300,5,'correlated');o,*_=record(x,y,40);lo,hi=bounds(o);m=(lo+hi)/2
  p=propensities(o);f=outcome(x[:,None],y[:,None],H)[0]
  active=p>0
  expect=(1-p[active,None])*m[active]+p[active,None]*(m[active]+(f[active]-m[active])/p[active,None])
  np.testing.assert_allclose(expect,f[active],atol=1e-13)
 def test_nonidentification(self):
  oa,*_=record(np.array([60.]),np.array([10.]),40.)
  ob,*_=record(np.array([200.]),np.array([10.]),40.)
  np.testing.assert_array_equal(oa,ob)
  self.assertNotEqual(outcome(np.array([60.]),np.array([10.]),300.)[0][0],outcome(np.array([200.]),np.array([10.]),300.)[0][0])
if __name__=='__main__':unittest.main()
