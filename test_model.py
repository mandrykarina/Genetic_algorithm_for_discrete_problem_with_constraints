import unittest
import numpy as np
from model import generate,metrics,repair,canonical,mutate,solve

class Tests(unittest.TestCase):
    def setUp(self):self.g=generate()
    def test_known_optimum(self):
        self.assertEqual(metrics(np.array(self.g['known_coloring']),self.g),(5,0))
        edges={tuple(e) for e in self.g['edges']}
        self.assertTrue(all((i,j) in edges for i in range(5) for j in range(i+1,5)))
    def test_repair_property(self):
        rng=np.random.default_rng(71)
        for _ in range(100):self.assertEqual(metrics(repair(rng.integers(0,5,30),self.g),self.g)[1],0)
    def test_canonical(self):np.testing.assert_array_equal(canonical([8,8,3,6,3]),[0,0,1,2,1])
    def test_conflict(self):self.assertGreater(metrics(np.zeros(30),self.g)[1],0)
    def test_mutation_shape(self):
        for mode in ['reset','copy']:
            x=mutate(np.arange(30),np.random.default_rng(1),1,mode);self.assertEqual(x.shape,(30,));self.assertTrue(np.all(x>=0))
    def test_budget_determinism(self):
        c=dict(population=12,generations=4,penalty=31,mutation_probability=.15,crossover_probability=.9)
        a=solve(c,self.g,17);b=solve(c,self.g,17)
        np.testing.assert_array_equal(a[1],b[1]);self.assertEqual(a[0]['evaluations'],60);self.assertEqual(a[0]['conflicts'],0);self.assertTrue(np.all(np.diff(a[2])<=0))
if __name__=='__main__':unittest.main()
