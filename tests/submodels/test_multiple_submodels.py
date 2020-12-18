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

from wc_lang import util
from wc_lang.io import Reader
from wc_sim.testing.transformations import SetStdDevsToZero
from wc_sim.testing.utils import create_run_directory
import wc_lang
import wc_sim


class TestMultipleSubmodels(unittest.TestCase):

    TEST_MODEL = os.path.join(os.path.dirname(__file__), '..', 'fixtures', 'test_model.xlsx')
    EXAMPLE_WC_MODEL = os.path.join(os.path.dirname(__file__), 'fixtures', '4_submodel_MP_model.xlsx')

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.model = Reader().run(self.EXAMPLE_WC_MODEL, validate=True)[wc_lang.Model][0]

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def run_simulation(self, simulation, max_time=10):
        checkpoint_period = min(1, max_time)
        with CaptureOutput(relay=True):
            simulation_rv = simulation.run(max_time=max_time,
                                           results_dir=create_run_directory(),
                                           checkpoint_period=checkpoint_period,
                                           dfba_time_step=2)
        self.assertTrue(0 < simulation_rv.num_events)
        self.assertTrue(os.path.isdir(simulation_rv.results_dir))
        run_results = wc_sim.RunResults(simulation_rv.results_dir)

        for component in wc_sim.RunResults.COMPONENTS:
            self.assertTrue(isinstance(run_results.get(component), (pandas.DataFrame, pandas.Series)))

    @unittest.skip("just SDs == 0")
    def test_run_submodel_types(self):
        print(f"\nRunning '{self.EXAMPLE_WC_MODEL}'")
        self.run_simulation(wc_sim.Simulation(self.EXAMPLE_WC_MODEL), max_time=20)

    def test_run_submodel_types_sds_eq_0(self):
        print(f"\nRunning w SDs == 0 '{self.EXAMPLE_WC_MODEL}'")
        print(util.get_model_summary(self.model))
        SetStdDevsToZero().run(self.model)
        self.run_simulation(wc_sim.Simulation(self.model), max_time=10)
