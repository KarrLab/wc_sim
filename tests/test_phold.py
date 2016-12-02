import os
import unittest
import warnings
import random

from argparse import Namespace

from examples.phold import RunPhold
from wc_sim.core.simulation_engine import SimulationEngine

class TestMultiAlgorithm(unittest.TestCase):

    def setUp(self):
        warnings.simplefilter("ignore")        

    def run_phold(self, seed, end_time):
        args = Namespace(end_time=end_time, frac_self_events=0.3, num_phold_procs=10, seed=seed)
        random.seed(seed)
        SimulationEngine.reset()
        return(RunPhold.main(args))

    def test_phold_reproducibility(self):
    
        num_events1=self.run_phold(123, 10)
        num_events2=self.run_phold(123, 10)
        self.assertEqual(num_events1, num_events2)
    
        num_events2=self.run_phold(173, 10)
        self.assertNotEqual(num_events1, num_events2)
