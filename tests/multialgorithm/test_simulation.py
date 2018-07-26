""" Test simple simulation

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2018-05-26
:Copyright: 2018, Karr Lab
:License: MIT
"""

import os
import unittest
import time
import shutil
import tempfile
import pandas
from copy import copy
from capturer import CaptureOutput

from wc_sim.core import sim_config
from wc_sim.core.sim_metadata import SimulationMetadata
from wc_sim.log.checkpoint import Checkpoint
from wc_lang.core import SpeciesType
from wc_sim.multialgorithm.simulation import Simulation
from wc_sim.multialgorithm.run_results import RunResults
from wc_sim.multialgorithm.make_models import MakeModels
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError

TOY_MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', '2_species_1_reaction.xlsx')


class TestSimulation(unittest.TestCase):

    def setUp(self):
        self.results_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.results_dir)

    def run_simulation(self, simulation):
        with CaptureOutput(relay=False):
            num_events, results_dir = simulation.run(end_time=100, results_dir=self.results_dir,
                checkpoint_period=10)

        # TODO(Arthur): add more specific tests
        self.assertTrue(0<num_events)
        self.assertTrue(os.path.isdir(results_dir))
        run_results = RunResults(results_dir)

        for component in RunResults.COMPONENTS:
            self.assertTrue(isinstance(run_results.get(component), (pandas.DataFrame, pandas.Series)))

    def test_simulation_model_in_file(self):
        self.run_simulation(Simulation(TOY_MODEL_FILENAME))

    def test_simulation_model_in_memory(self):
        model = MakeModels.make_test_model('2 species, 1 reaction', transform_prep_and_check=False)
        self.run_simulation(Simulation(model))

    def test_simulation_errors(self):
        with self.assertRaisesRegexp(MultialgorithmError, 'model must be a wc_lang Model or a pathname'):
            Simulation(2)
        model = MakeModels.make_test_model('2 species, 1 reaction, with rates given by reactant population',
            specie_copy_numbers={'spec_type_0[compt_1]':10, 'spec_type_1[compt_1]':0}, init_vols=[1E-22])
        with self.assertRaisesRegexp(MultialgorithmError,
            'simulation with 1 submodel and total propensities = 0 cannot progress'):
            Simulation(model).run(1000)

    def test_simulate_wo_output_files(self):
        with CaptureOutput(relay=False):
            num_events, results_dir = Simulation(TOY_MODEL_FILENAME).run(end_time=100)
        self.assertTrue(0 < num_events)
        self.assertEqual(results_dir, None)

    def test_simulate(self):
        end_time = 30
        with CaptureOutput(relay=False):
            num_events, results_dir = Simulation(TOY_MODEL_FILENAME).run(end_time=end_time,
                results_dir=self.results_dir, checkpoint_period=10)

        # check time, and simulation config in checkpoints
        for time in Checkpoint.list_checkpoints(results_dir):
            ckpt = Checkpoint.get_checkpoint(results_dir, time=time)
            self.assertEqual(time, ckpt.time)
            self.assertTrue(ckpt.random_state != None)

    def test_reseed(self):
        # different seeds must make different results
        seeds = [17, 19]
        results = {}
        run_results = {}
        for seed in seeds:
            tmp_results_dir = tempfile.mkdtemp()
            with CaptureOutput(relay=False):
                num_events, results_dir = Simulation(TOY_MODEL_FILENAME).run(end_time=20,
                    results_dir=tmp_results_dir, checkpoint_period=5, seed=seed)
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
                num_events, results_dir = Simulation(TOY_MODEL_FILENAME).run(end_time=20,
                    results_dir=tmp_results_dir, checkpoint_period=5, seed=seed)
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


class TestProcessAndValidateArgs(unittest.TestCase):

    def setUp(self):
        self.results_dir = tempfile.mkdtemp()
        self.tmp = os.path.expanduser('~/tmp/test_dir/checkpoints_dir')
        if not os.path.exists(self.tmp):
            os.makedirs(self.tmp)
        self.user_tmp_dir = tempfile.mkdtemp(dir=self.tmp)
        self.simulation = Simulation(TOY_MODEL_FILENAME)
        self.args = dict(
            results_dir=self.results_dir,
            checkpoint_period=10,
            end_time=100,
            time_step=5
        )

    def tearDown(self):
        shutil.rmtree(self.results_dir)
        shutil.rmtree(self.user_tmp_dir)

    def test_create_metadata_1(self):
        with self.assertRaises(MultialgorithmError):
            self.simulation._create_metadata()

        self.simulation.sim_config = sim_config.SimulationConfig(time_max=self.args['end_time'],
            time_step=self.args['time_step']) 
        simulation_metadata = self.simulation._create_metadata()
        for attr in SimulationMetadata.ATTRIBUTES:
            self.assertTrue(getattr(simulation_metadata, attr) is not None)

    def test_create_metadata_2(self):
        # no time_step
        self.simulation.sim_config = sim_config.SimulationConfig(time_max=self.args['end_time']) 
        del self.args['time_step']
        simulation_metadata = self.simulation._create_metadata()
        self.assertEqual(simulation_metadata.simulation.time_step, 1)

    def test_ckpt_dir_processing_1(self):
        # checkpoints_dir does not exist
        self.args['results_dir'] = os.path.join(self.results_dir, 'no_such_dir', 'no_such_sub_dir')
        self.simulation.process_and_validate_args(self.args)
        self.assertTrue(os.path.isdir(self.args['results_dir']))

    def test_ckpt_dir_processing_2(self):
        # checkpoints_dir exists, and is empty
        root_dir = self.args['results_dir']
        self.simulation.process_and_validate_args(self.args)
        # process_and_validate_args creates 1 timestamped sub-dir
        self.assertEqual(len(os.listdir(root_dir)), 1)

    def test_ckpt_dir_processing_3(self):
        # checkpoints_dir is a file
        self.args['results_dir'] = os.path.join(self.args['results_dir'], 'new_file')
        try:
            open(self.args['results_dir'], 'x')
            with self.assertRaises(MultialgorithmError):
                self.simulation.process_and_validate_args(self.args)
        except FileExistsError:
           pass

    def test_ckpt_dir_processing_4(self):
        # timestamped sub-directory of checkpoints-dir already exists
        root_dir = self.args['results_dir']
        self.simulation.process_and_validate_args(self.args)
        # given the chance, albeit small, that the second has advanced and
        # a different timestamped sub-directory is made, try repeatedly to create the error
        # the for loop takes about 0.01 sec
        raised = False
        for i in range(10):
            self.args['results_dir'] = root_dir
            try:
                self.simulation.process_and_validate_args(self.args)
            except:
                raised = True
        self.assertTrue(raised)

    def test_process_and_validate_args1(self):
        original_args = copy(self.args)
        self.simulation.process_and_validate_args(self.args)
        self.assertTrue(self.args['results_dir'].startswith(original_args['results_dir']))

    def test_process_and_validate_args2(self):
        # test files specified relative to home directory
        relative_tmp_dir = os.path.join('~/tmp/', os.path.basename(self.user_tmp_dir))
        self.args['results_dir'] = relative_tmp_dir
        self.simulation.process_and_validate_args(self.args)
        self.assertIn(relative_tmp_dir.replace('~', ''), self.args['results_dir'])

    def test_process_and_validate_args3(self):
        self.args['checkpoint_period'] =7
        with self.assertRaises(MultialgorithmError):
            self.simulation.process_and_validate_args(self.args)

    def test_process_and_validate_args4(self):
        # test no results dir
        del self.args['results_dir']
        original_args = copy(self.args)
        self.simulation.process_and_validate_args(self.args)
        self.assertEqual(self.args, original_args)

    def test_process_and_validate_args5(self):
        # test error detection
        errors = dict(
            end_time = [-3, 0],
            checkpoint_period = [-2, 0, self.args['end_time'] + 1],
            time_step = [-2, 0, self.args['end_time'] + 1],
        )
        for arg, error_vals in errors.items():
            for error_val in error_vals:
                bad_args = copy(self.args)
                bad_args[arg] = error_val
                # need a good, empty checkpoints_dir for each call to process_and_validate_args
                new_tmp_dir = tempfile.mkdtemp()
                bad_args['results_dir'] = new_tmp_dir
                with self.assertRaises(MultialgorithmError):
                    self.simulation.process_and_validate_args(bad_args)
                shutil.rmtree(new_tmp_dir)

    def test_process_and_validate_args6(self):
        del self.args['end_time']
        with self.assertRaises(MultialgorithmError):
            self.simulation.process_and_validate_args(self.args)

    def test_process_and_validate_args7(self):
        del self.args['time_step']
        no_exception = False
        try:
            self.simulation.process_and_validate_args(self.args)
            no_exception = True
        except:
            pass
        self.assertTrue(no_exception)
