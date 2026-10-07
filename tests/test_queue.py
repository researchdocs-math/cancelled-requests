import unittest
import numpy as np
from queue_stress import simulate

class QueueTests(unittest.TestCase):
 def test_single_request_cancellation(self):
  r=simulate(np.array([100.]),np.array([10.]),np.array([0.]),np.array([0]),40.,np.array([False]),0.)
  self.assertEqual(r['latency'],50.);self.assertEqual(r['work'],60.)
 def test_audit_charges_unseen_attempt(self):
  r=simulate(np.array([10.]),np.array([200.]),np.array([0.]),np.array([0]),300.,np.array([True]),0.)
  self.assertEqual(r['latency'],10.);self.assertEqual(r['work'],210.);self.assertAlmostEqual(r['utilization'],.5)
 def test_cancellation_propagation(self):
  r=simulate(np.array([100.]),np.array([10.]),np.array([0.]),np.array([0]),40.,np.array([False]),5.)
  self.assertEqual(r['latency'],50.);self.assertEqual(r['work'],65.)
 def test_mm1_reference(self):
  rng=np.random.default_rng(739);n=100000;rho=.6;mean=20.
  x=rng.exponential(mean,n);y=rng.exponential(mean,n)
  arrival=np.cumsum(rng.exponential(mean/(2*rho),n));server=rng.integers(0,2,n)
  r=simulate(x,y,arrival,server,300.,np.zeros(n,dtype=bool),0.)
  self.assertLess(abs(r['latency']-mean/(1-rho))/(mean/(1-rho)),.06)
  self.assertLess(abs(r['utilization']-rho),.02)
if __name__=='__main__':unittest.main()
