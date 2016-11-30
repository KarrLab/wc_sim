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

    def run_phold(self):
        seed=123
        args = Namespace(end_time=3.0, frac_self_events=0.3, num_phold_procs=3, seed=seed)
        random.seed(seed)
        SimulationEngine.reset()
        return(RunPhold.main(args))

    def test_phold(self):
        num_events1=self.run_phold()
        num_events2=self.run_phold()
        self.assertEqual(num_events1, num_events2)
