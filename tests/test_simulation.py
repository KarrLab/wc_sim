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
import pstats
import shutil
import tempfile
import time
import unittest

from de_sim.checkpoint import Checkpoint, AccessCheckpoints
from de_sim.config import core
from de_sim.simulator import Simulator
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

    def run_simulation(self, simulation, max_time=100):
        checkpoint_period = min(10, max_time)
        with CaptureOutput(relay=False):
            simulation_rv = simulation.run(max_time=max_time, results_dir=self.results_dir,
                                            checkpoint_period=checkpoint_period)
        self.assertTrue(0 < simulation_rv.num_events)
        self.assertTrue(os.path.isdir(simulation_rv.results_dir))
        run_results = RunResults(simulation_rv.results_dir)

        for component in RunResults.COMPONENTS:
            self.assertTrue(isinstance(run_results.get(component), (pandas.DataFrame, pandas.Series)))

    def test_simulation_model_in_file(self):
        self.run_simulation(Simulation(TOY_MODEL_FILENAME), max_time=5)

    def test_simulation_model_in_memory(self):
        model = MakeModel.make_test_model('2 species, 1 reaction', transform_prep_and_check=False)
        self.run_simulation(Simulation(model))

    def test_simulation_errors(self):
        with self.assertRaisesRegex(MultialgorithmError,
                                    'model must be a `wc_lang Model` or a pathname for a model'):
            Simulation(3)

    def test_simulate_wo_output_files(self):
        with CaptureOutput(relay=False):
            simulation_rv = Simulation(TOY_MODEL_FILENAME).run(max_time=5)
        self.assertTrue(0 < simulation_rv.num_events)
        self.assertEqual(simulation_rv.results_dir, None)

    def test_run(self):
        with CaptureOutput(relay=False):
            simulation_rv = Simulation(TOY_MODEL_FILENAME).run(max_time=2,
                                                               results_dir=self.results_dir,
                                                               checkpoint_period=1)

        # check time, and simulation config in checkpoints
        access_checkpoints = AccessCheckpoints(simulation_rv.results_dir)
        for time in access_checkpoints.list_checkpoints():
            ckpt = access_checkpoints.get_checkpoint(time=time)
            self.assertEqual(time, ckpt.time)
            self.assertTrue(ckpt.random_state != None)

        # test performance profiling and verbose to stdout
        with CaptureOutput(relay=False) as capturer:
            simulation_rv = Simulation(TOY_MODEL_FILENAME).run(max_time=2,
                                                               profile=True,
                                                               verbose=True)
            expected_patterns =['function calls',
                                'filename:lineno\(function\)',
                                'Simulated \d+ events',
                                'Caching statistics']
            for pattern in expected_patterns:
                self.assertRegex(capturer.get_text(), pattern)
            self.assertTrue(isinstance(simulation_rv.profile_stats, pstats.Stats))

        with self.assertRaisesRegex(MultialgorithmError, 'cannot be simulated .* it contains no submodels'):
            Simulation(TOY_MODEL_FILENAME).run(max_time=5,
                                               results_dir=tempfile.mkdtemp(dir=self.test_dir),
                                               checkpoint_period=1,
                                               submodels_to_skip=['test_submodel'])

    def test_object_memory_use(self):
        # test memory use profile to measurements file
        results_dir = tempfile.mkdtemp(dir=self.test_dir)
        # set object_memory_change_interval=50 because the simulation has about 200 events
        Simulation(TOY_MODEL_FILENAME).run(max_time=2,
                                           results_dir=results_dir,
                                           checkpoint_period=1,
                                           object_memory_change_interval=50)
        expected_patterns =['Memory use changes by SummaryTracker',
                            '# objects']
        measurements_file = core.get_config()['de_sim']['measurements_file']
        measurements_pathname = os.path.join(results_dir, measurements_file)
        measurements = open(measurements_pathname, 'r').read()
        for pattern in expected_patterns:
            self.assertRegex(measurements, pattern)

    def test_reseed(self):
        # different seeds must make different results
        seeds = [17, 19]
        results = {}
        run_results = {}
        for seed in seeds:
            tmp_results_dir = tempfile.mkdtemp()
            with CaptureOutput(relay=False):
                simulation_rv = Simulation(TOY_MODEL_FILENAME).run(max_time=5,
                                                                   results_dir=tmp_results_dir,
                                                                   checkpoint_period=5, seed=seed)
            results[seed] = {}
            results[seed]['num_events'] = simulation_rv.num_events
            run_results[seed] = RunResults(simulation_rv.results_dir)
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
                simulation_rv = Simulation(TOY_MODEL_FILENAME).run(max_time=5,
                                                                   results_dir=tmp_results_dir,
                                                                   checkpoint_period=5, seed=seed)
            results[rep] = {}
            results[rep]['num_events'] = simulation_rv.num_events
            run_results[rep] = RunResults(simulation_rv.results_dir)
            shutil.rmtree(tmp_results_dir)
        self.assertEqual(results[0]['num_events'], results[1]['num_events'])
        self.assertTrue(run_results[0].get('populations').equals(run_results[1].get('populations')))
        for component in RunResults.COMPONENTS:
            # metadata differs, because it includes timestamp
            if component != 'metadata':
                self.assertTrue(run_results[0].get(component).equals(run_results[1].get(component)))

    def test_get_simulator(self):
        model = MakeModel.make_test_model('2 species, 1 reaction, with rates given by reactant population')
        simulation = Simulation(model)
        self.assertEqual(simulation.get_simulator(), None)
        with CaptureOutput(relay=False):
            simulation.run(max_time=5)
        self.assertTrue(isinstance(simulation.get_simulator(), Simulator))

    def test_provide_event_counts(self):
        model = MakeModel.make_test_model('2 species, 1 reaction, with rates given by reactant population')
        simulation = Simulation(model)
        self.assertTrue('execute run() to obtain event counts' in simulation.provide_event_counts())
        with CaptureOutput(relay=False):
            simulation.run(max_time=100)
        self.assertTrue('Event type' in simulation.provide_event_counts())
