""" Test RunResults

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-05-20
:Copyright: 2018, Karr Lab
:License: MIT
"""

import os
import getpass
import unittest
import shutil
import tempfile
from argparse import Namespace
import warnings
import pandas
import numpy

from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm.simulation import Simulation
from wc_sim.multialgorithm.run_results import RunResults


class TestRunResults(unittest.TestCase):

    def setUp(self):
        # use stored checkpoints and metadata from simulation of 2_species_1_reaction model
        self.RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'results_dir')
        self.results_dir = tempfile.mkdtemp()
        self.results_copy = os.path.join(self.results_dir, 'results_copy')
        shutil.copytree(self.RESULTS_DIR, self.results_copy)
        self.checkpoint_period = 10
        self.max_time = 100

    def tearDown(self):
        shutil.rmtree(self.results_dir)

    def test_run_results(self):

        run_results_1 = RunResults(self.results_copy)
        # after run_results file created
        run_results_2 = RunResults(self.results_copy)
        for component in RunResults.COMPONENTS:
            component_data = run_results_1.get(component)
            self.assertTrue(run_results_1.get(component).equals(run_results_2.get(component)))


        expected_times = pandas.Float64Index(numpy.linspace(0, self.max_time, 1 + self.max_time/self.checkpoint_period))
        for component in ['populations', 'aggregate_states', 'random_states']:
            component_data = run_results_1.get(component)
            self.assertTrue(component_data.index.equals(expected_times))

        # total population is invariant
        populations = run_results_1.get('populations')
        pop_sum = populations.sum(axis='columns')
        for time in expected_times:
            self.assertEqual(pop_sum[time], pop_sum[0.])

        metadata = run_results_1.get('metadata')
        self.assertEqual(metadata['simulation']['time_max'], self.max_time)

    def test_run_results_errors(self):

        run_results = RunResults(self.results_copy)
        with self.assertRaisesRegexp(MultialgorithmError, "component '.*' is not an element of "):
            run_results.get('not_a_component')
