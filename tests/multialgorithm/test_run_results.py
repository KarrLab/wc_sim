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

from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm.__main__ import SimController
from wc_sim.multialgorithm.run_results import RunResults


class TestRunResults(unittest.TestCase):

    def setUp(self):
        self.CHECKPOINTS_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'checkpoints_dir')
        self.checkpoints_dir = tempfile.mkdtemp()
        self.checkpoints_copy = os.path.join(self.checkpoints_dir, 'checkpoints_copy')
        shutil.copytree(self.CHECKPOINTS_DIR, self.checkpoints_copy)
        self.args = Namespace(
            model_file='filename',
            end_time=10,
            checkpoint_period=3,
            checkpoints_dir=self.checkpoints_dir,
            fba_time_step=5
        )
        self.metadata = SimController.create_metadata(self.args)

    def tearDown(self):
        shutil.rmtree(self.checkpoints_dir)

    def test_run_results(self):

        run_results_1 = RunResults(self.checkpoints_copy, self.metadata)
        # after run_results file created
        run_results_2 = RunResults(self.checkpoints_copy)
        for component in RunResults.COMPONENTS:
            component_data = run_results_1.get(component)
            self.assertTrue(run_results_1.get(component).equals(run_results_2.get(component)))

        expected_times = pandas.Float64Index([0., 3., 6., 9.])
        for component in ['populations', 'aggregate_states', 'random_states']:
            component_data = run_results_1.get(component)
            self.assertTrue(component_data.index.equals(expected_times))

        # total population is invariant
        populations = run_results_1.get('populations')
        pop_sum = populations.sum(axis='columns')
        for time in expected_times:
            self.assertEqual(pop_sum[time], pop_sum[0.])

        metadata = run_results_1.get('metadata')
        self.assertEqual(metadata['simulation']['time_max'], 10.)
        self.assertEqual(metadata['author']['username'], getpass.getuser())

    def test_run_results_errors(self):

        with self.assertRaises(MultialgorithmError):
            RunResults(self.checkpoints_copy)
        run_results = RunResults(self.checkpoints_copy, self.metadata)
        with self.assertRaisesRegexp(MultialgorithmError, "component '.*' is not an element of "):
            run_results.get('not_a_component')
