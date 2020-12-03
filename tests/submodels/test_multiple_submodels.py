"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2020-12-03
:Copyright: 2020, Karr Lab
:License: MIT
"""

from capturer import CaptureOutput
import os
import pandas
import shutil
import tempfile
import unittest

import wc_sim


class TestMultipleSubmodels(unittest.TestCase):

    TEST_MODEL = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'test_model.xlsx')
    # todo: make work on CircleCI
    EXAMPLE_WC_LANG_MODEL = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'wc_lang', 'tests', 'fixtures',
                                         'example-model.xlsx')

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.models = [self.TEST_MODEL, self.EXAMPLE_WC_LANG_MODEL]

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def run_simulation(self, simulation, max_time=10):
        checkpoint_period = min(1, max_time)
        with CaptureOutput(relay=True):
            simulation_rv = simulation.run(max_time=max_time,
                                           results_dir=tempfile.mkdtemp(dir=self.test_dir),
                                           checkpoint_period=checkpoint_period,
                                           dfba_time_step=2)
        self.assertTrue(0 < simulation_rv.num_events)
        self.assertTrue(os.path.isdir(simulation_rv.results_dir))
        run_results = wc_sim.RunResults(simulation_rv.results_dir)

        for component in wc_sim.RunResults.COMPONENTS:
            self.assertTrue(isinstance(run_results.get(component), (pandas.DataFrame, pandas.Series)))

    def test_simulation_model_in_file(self):
        for model in self.models:
            print(f"\nRunning '{model}'")
            self.run_simulation(wc_sim.Simulation(model), max_time=20)
