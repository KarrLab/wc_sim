""" Test simulation

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-05-26
:Copyright: 2018, Karr Lab
:License: MIT
"""

from capturer import CaptureOutput
from copy import copy
import os
import pandas
import shutil
import tempfile
import time
import unittest

from de_sim.checkpoint import Checkpoint, AccessCheckpoints
from de_sim.simulation_engine import SimulationEngine
from wc_sim import sim_config
from wc_sim.multialgorithm_errors import MultialgorithmError
from wc_sim.run_results import RunResults
from wc_sim.simulation import Simulation
from wc_sim.testing.make_models import MakeModel

TOY_MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', '2_species_1_reaction.xlsx')


class TestSimulation(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.results_dir = tempfile.mkdtemp(dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def run_simulation(self, simulation, time_max=100):
        checkpoint_period = min(10, time_max)
        with CaptureOutput(relay=False):
            num_events, results_dir = simulation.run(time_max=time_max, results_dir=self.results_dir,
                                                     checkpoint_period=checkpoint_period)
        self.assertTrue(0 < num_events)
        self.assertTrue(os.path.isdir(results_dir))
        run_results = RunResults(results_dir)

        for component in RunResults.COMPONENTS:
            self.assertTrue(isinstance(run_results.get(component), (pandas.DataFrame, pandas.Series)))

    def test_prepare(self):
        model = MakeModel.make_test_model('2 species, 1 reaction, with rates given by reactant population',
                                          transform_prep_and_check=False)
        model.id = 'illegal id'
        simulation = Simulation(model)
        with self.assertRaises(MultialgorithmError):
            simulation._prepare()

    def test_simulation_model_in_file(self):
        self.run_simulation(Simulation(TOY_MODEL_FILENAME), time_max=5)

    def test_simulation_model_in_memory(self):
        model = MakeModel.make_test_model('2 species, 1 reaction', transform_prep_and_check=False)
        self.run_simulation(Simulation(model))

    def test_simulation_errors(self):
        with self.assertRaisesRegex(MultialgorithmError,
                                    'model must be a `wc_lang Model` or a pathname for a model'):
            Simulation(3)

    def test_simulate_wo_output_files(self):
        with CaptureOutput(relay=False):
            num_events, results_dir = Simulation(TOY_MODEL_FILENAME).run(time_max=5)
        self.assertTrue(0 < num_events)
        self.assertEqual(results_dir, None)

    def test_run(self):
        with CaptureOutput(relay=False):
            num_events, results_dir = Simulation(TOY_MODEL_FILENAME).run(time_max=5,
                                                                         results_dir=self.results_dir,
                                                                         checkpoint_period=1)

        # check time, and simulation config in checkpoints
        access_checkpoints = AccessCheckpoints(results_dir)
        for time in access_checkpoints.list_checkpoints():
            ckpt = access_checkpoints.get_checkpoint(time=time)
            self.assertEqual(time, ckpt.time)
            self.assertTrue(ckpt.random_state != None)

        with self.assertRaisesRegex(MultialgorithmError, 'cannot be simulated .* it contains no submodels'):
            Simulation(TOY_MODEL_FILENAME).run(time_max=5,
                                               results_dir=tempfile.mkdtemp(dir=self.test_dir),
                                               checkpoint_period=1,
                                               submodels_to_skip=['test_submodel'])

    def test_reseed(self):
        # different seeds must make different results
        seeds = [17, 19]
        results = {}
        run_results = {}
        for seed in seeds:
            tmp_results_dir = tempfile.mkdtemp()
            with CaptureOutput(relay=False):
                num_events, results_dir = Simulation(TOY_MODEL_FILENAME).run(time_max=5,
                                                                             results_dir=tmp_results_dir,
                                                                             checkpoint_period=5, seed=seed)
            results[seed] = {}
            results[seed]['num_events'] = num_events
            run_results[seed] = RunResults(results_dir)
            shutil.rmtree(tmp_results_dir)
        self.assertNotEqual(results[seeds[0]]['num_events'], results[seeds[1]]['num_events'])
        self.assertFalse(run_results[seeds[0]].get('populations').equals(run_results[seeds[1]].get('populations')))

        # a given seed must must always make the same result
        seed = 117
        results = {}
        run_results = {}
        for rep in range(2):
            tmp_results_dir = tempfile.mkdtemp()
            with CaptureOutput(relay=False):
                num_events, results_dir = Simulation(TOY_MODEL_FILENAME).run(time_max=5,
                                                                             results_dir=tmp_results_dir,
                                                                             checkpoint_period=5, seed=seed)
            results[rep] = {}
            results[rep]['num_events'] = num_events
            run_results[rep] = RunResults(results_dir)
            shutil.rmtree(tmp_results_dir)
        self.assertEqual(results[0]['num_events'], results[1]['num_events'])
        self.assertTrue(run_results[0].get('populations').equals(run_results[1].get('populations')))
        for component in RunResults.COMPONENTS:
            # metadata differs, because it includes timestamp
            if component != 'metadata':
                self.assertTrue(run_results[0].get(component).equals(run_results[1].get(component)))

    def test_get_simulation_engine(self):
        model = MakeModel.make_test_model('2 species, 1 reaction, with rates given by reactant population')
        simulation = Simulation(model)
        self.assertEqual(simulation.get_simulation_engine(), None)
        with CaptureOutput(relay=False):
            simulation.run(time_max=5)
        self.assertTrue(isinstance(simulation.get_simulation_engine(), SimulationEngine))

    def test_provide_event_counts(self):
        model = MakeModel.make_test_model('2 species, 1 reaction, with rates given by reactant population')
        simulation = Simulation(model)
        self.assertTrue('execute run() to obtain event counts' in simulation.provide_event_counts())
        with CaptureOutput(relay=False):
            simulation.run(time_max=100)
        self.assertTrue('Event type' in simulation.provide_event_counts())
