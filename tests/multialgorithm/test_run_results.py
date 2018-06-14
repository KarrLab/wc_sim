""" Test RunResults

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-05-20
:Copyright: 2018, Karr Lab
:License: MIT
"""

import unittest
import shutil
import tempfile
import pandas
import numpy

from wc_lang.core import SpeciesType
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm.simulation import Simulation
from wc_sim.multialgorithm.run_results import RunResults
from wc_sim.multialgorithm.make_models import MakeModels


class TestRunResults(unittest.TestCase):

    def setUp(self):
        # create stored checkpoints and metadata
        self.temp_dir = tempfile.mkdtemp()
        # create and run simulation
        model = MakeModels.make_test_model('2 species, 1 reaction')
        simulation = Simulation(model)
        self.checkpoint_period = 10
        self.max_time = 100
        _, self.results_dir = simulation.run(end_time=self.max_time, results_dir=self.temp_dir,
            checkpoint_period=self.checkpoint_period)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_run_results(self):

        run_results_1 = RunResults(self.results_dir)
        # after run_results file created
        run_results_2 = RunResults(self.results_dir)
        for component in RunResults.COMPONENTS:
            component_data = run_results_1.get(component)
            self.assertTrue(run_results_1.get(component).equals(run_results_2.get(component)))

        expected_times = pandas.Float64Index(numpy.linspace(0, self.max_time, 1 + self.max_time/self.checkpoint_period))
        for component in ['populations', 'observables', 'aggregate_states', 'random_states']:
            component_data = run_results_1.get(component)
            self.assertFalse(component_data.empty)
            self.assertTrue(component_data.index.equals(expected_times))

        # total population is invariant
        populations = run_results_1.get('populations')
        pop_sum = populations.sum(axis='columns')
        for time in expected_times:
            self.assertEqual(pop_sum[time], pop_sum[0.])

        metadata = run_results_1.get('metadata')
        self.assertEqual(metadata['simulation']['time_max'], self.max_time)

    def test_run_results_errors(self):

        run_results = RunResults(self.results_dir)
        with self.assertRaisesRegexp(MultialgorithmError, "component '.*' is not an element of "):
            run_results.get('not_a_component')
