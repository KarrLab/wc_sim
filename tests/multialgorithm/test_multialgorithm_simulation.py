"""
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-02-08
:Copyright: 2017-2018, Karr Lab
:License: MIT
"""

import unittest
import os
import math
import re
import numpy as np
import shutil
import tempfile
import cProfile
import pstats
from scipy.constants import Avogadro
import pandas
import copy
import time

from wc_utils.util.dict import DictUtil
from obj_model import utils
from wc_lang.io import Reader, Writer
from wc_lang.core import (Model, Submodel,  SpeciesType, SpeciesTypeType, Species,
                          Reaction, Compartment,
                          SpeciesCoefficient, Parameter,
                          RateLaw, RateLawDirection, RateLawEquation, SubmodelAlgorithm, Concentration,
                          BiomassComponent, BiomassReaction, StopCondition)
from wc_lang.prepare import PrepareModel, CheckModel
from wc_lang.transform import SplitReversibleReactionsTransform
from wc_sim.multialgorithm.make_models import MakeModels
from wc_sim.multialgorithm.model_utilities import ModelUtilities
from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.multialgorithm.config import core as config_core_multialgorithm
from wc_sim.multialgorithm.multialgorithm_checkpointing import (MultialgorithmicCheckpointingSimObj,
    MultialgorithmCheckpoint)
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_sim.multialgorithm.run_results import RunResults

from wc_sim.core.simulation_checkpoint_object import CheckpointSimulationObject
from wc_sim.core.debug_logs import logs, config

config_multialgorithm = config_core_multialgorithm.get_config()['wc_sim']['multialgorithm']

# TODO(Arthur): transcode & eval invariants
class Invariant(object):
    """ Support invariant expressions on species concentrations for model testing

    Attributes:
        original_value (:obj:`str`): the original, readable representation of the invariant
        transcoded (:obj:`str`): a representation of the invariant that's ready to be evaluated
    """

    def __init__(self, original_value):
        """
        Args:
            original_value (:obj:`str`): the original, readable representation of the invariant
        """
        self.original_value = original_value
        self.transcoded = None

    def transcode(self):
        """ Transcode the invariant into a form that can be evaluated
        """
        pass

    def eval(self):
        """ Evaluate the invariant

        Returns:
            :obj:`object`: value returned by the invariant, usually a `bool`
        """
        return True


class TestMultialgorithmSimulation(unittest.TestCase):

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')

    def setUp(self):
        # read and initialize a model
        self.model = Reader().run(self.MODEL_FILENAME, strict=False)
        self.args = dict(fba_time_step=1,
            results_dir=None)
        self.multialgorithm_simulation = MultialgorithmSimulation(self.model, self.args)
        self.results_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.results_dir)

    def test_molecular_weights_for_species(self):
        multi_alg_sim = self.multialgorithm_simulation
        self.assertEqual(multi_alg_sim.molecular_weights_for_species(set()), {})
        expected = {
            'specie_6[c]':6,
            'H2O[c]':18.0152
        }
        self.assertEqual(multi_alg_sim.molecular_weights_for_species(set(expected.keys())),
            expected)

    def test_partition_species(self):
        self.multialgorithm_simulation.partition_species()
        expected_priv_species = dict(
            submodel_1=['specie_1[e]', 'specie_2[e]', 'specie_1[c]'],
            submodel_2=['specie_4[c]', 'specie_5[c]', 'specie_6[c]']
        )
        self.assertEqual(self.multialgorithm_simulation.private_species, expected_priv_species)
        expected_shared_species = set(['specie_2[c]', 'specie_3[c]', 'H2O[e]', 'H2O[c]'])
        self.assertEqual(self.multialgorithm_simulation.shared_species, expected_shared_species)

    def test_dynamic_compartments(self):
        expected_compartments = dict(
            submodel_1=['c', 'e'],
            submodel_2=['c']
        )
        for submodel_id in ['submodel_1', 'submodel_2']:
            submodel = self.model.submodels.get_one(id=submodel_id)
            submodel_dynamic_compartments = self.multialgorithm_simulation.get_dynamic_compartments(submodel)
            self.assertEqual(set(submodel_dynamic_compartments.keys()), set(expected_compartments[submodel_id]))

    def test_static_methods(self):
        initial_species_population = MultialgorithmSimulation.get_initial_species_pop(self.model)
        specie_wo_init_conc = 'specie_3[c]'
        self.assertEqual(initial_species_population[specie_wo_init_conc], 0)
        self.assertEqual(initial_species_population['specie_2[c]'], initial_species_population['specie_4[c]'])
        for concentration in self.model.get_concentrations():
            self.assertGreater(initial_species_population[concentration.species.id()], 0)

        local_species_population = MultialgorithmSimulation.make_local_species_pop(self.model)
        self.assertEqual(local_species_population.read_one(0, specie_wo_init_conc), 0)

    def test_build_simulation(self):
        args = dict(fba_time_step=1,
            results_dir=self.results_dir,
            checkpoint_period=10)
        multialgorithm_simulation = MultialgorithmSimulation(self.model, args)
        simulation_engine, _ = multialgorithm_simulation.build_simulation()
        # 3 objects: 2 submodels, and the checkpointing obj:
        self.assertEqual(len(simulation_engine.simulation_objects.keys()), 3)
        self.assertEqual(type(multialgorithm_simulation.checkpointing_sim_obj),
            MultialgorithmicCheckpointingSimObj)
        self.assertEqual(multialgorithm_simulation.dynamic_model.get_num_submodels(), 2)
        self.assertTrue(callable(simulation_engine.stop_condition))

    def test_stop_condition(self):
        args = dict(fba_time_step=1,
            results_dir=self.results_dir,
            checkpoint_period=10)
        multialgorithm_simulation = MultialgorithmSimulation(self.model, args)
        simulation_engine, _ = multialgorithm_simulation.build_simulation()
        self.assertTrue(callable(simulation_engine.stop_condition))


class TestRunSSASimulation(unittest.TestCase):

    def setUp(self):
        self.results_dir = tempfile.mkdtemp()
        self.args = dict(fba_time_step=1,
            results_dir=self.results_dir,
            checkpoint_period=10)
        self.out_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.results_dir)
        shutil.rmtree(self.out_dir)

    def make_model_and_simulation(self, model_type, num_submodels, specie_copy_numbers=None, init_vols=None):
        # make simple model
        if init_vols is not None:
            if not isinstance(init_vols, list):
                init_vols = [init_vols]*num_submodels
        model = MakeModels.make_test_model(model_type, num_submodels=num_submodels,
            specie_copy_numbers=specie_copy_numbers, init_vols=init_vols)
        multialgorithm_simulation = MultialgorithmSimulation(model, self.args)
        simulation_engine, _ = multialgorithm_simulation.build_simulation()
        return (model, multialgorithm_simulation, simulation_engine)

    def checkpoint_times(self, run_time):
        """ Provide expected checkpoint times for a simulation
        """
        checkpoint_period = self.args['checkpoint_period']
        checkpoint_times = []
        t = 0
        while t <= run_time:
            checkpoint_times.append(t)
            t += checkpoint_period
        return checkpoint_times

    def perform_ssa_test_run(self, model_type, run_time, initial_specie_copy_numbers,
        expected_mean_copy_numbers, delta, num_submodels=1, invariants=None, iterations=3, init_vols=None):
        """ Test SSA by comparing expected and actual simulation copy numbers

        Args:
            model_type (:obj:`str`): model type description
            run_time (:obj:`float`): duration of the simulation run
            initial_specie_copy_numbers (:obj:`dict`): initial specie counts, with IDs as keys and counts as values
            expected_mean_copy_numbers (:obj:`str`): expected final mean specie counts, in same format as
                `initial_specie_copy_numbers`
            delta (:obj:`int`): maximum allowed difference between expected and actual counts
            num_submodels (:obj:`int`): number of submodels to create
            invariants (:obj:`list`, optional): list of invariant relationships, to be tested
            iterations (:obj:`int`, optional): number of simulation runs
            init_vols (:obj:`float`, `list` of `floats`, optional): initial volume of compartment(s)
                if reaction rates depend on concentration, use a smaller volume to increase rates
        """
        # TODO(Arthur): analytically determine the values for delta
        # TODO(Arthur): provide some invariant objects
        invariant_objs = [] if invariants is None else [Invariant(value) for value in invariants]

        final_specie_counts = []
        for i in range(iterations):
            model, multialgorithm_simulation, simulation_engine = self.make_model_and_simulation(
                model_type,
                num_submodels=num_submodels,
                specie_copy_numbers=initial_specie_copy_numbers,
                init_vols=init_vols)
            local_species_pop = multialgorithm_simulation.local_species_population
            simulation_engine.initialize()
            simulation_engine.simulate(run_time)
            final_specie_counts.append(local_species_pop.read(run_time))

        mean_final_specie_counts = dict.fromkeys(list(initial_specie_copy_numbers.keys()), 0)
        for final_specie_count in final_specie_counts:
            for k,v in final_specie_count.items():
                mean_final_specie_counts[k] += v
        for k,v in mean_final_specie_counts.items():
            mean_final_specie_counts[k] = v/iterations
            self.assertAlmostEqual(mean_final_specie_counts[k], expected_mean_copy_numbers[k], delta=delta)
        for invariant_obj in invariant_objs:
            self.assertTrue(invariant_obj.eval())

        # check the checkpoint times
        self.assertEqual(MultialgorithmCheckpoint.list_checkpoints(self.results_dir), self.checkpoint_times(run_time))

    def test_run_ssa_suite(self):
        specie = 'spec_type_0[compt_1]'
        self.perform_ssa_test_run('1 species, 1 reaction',
            run_time=999,       # tests checkpoint history in which the last checkpoint time < run time
            initial_specie_copy_numbers={specie:3000},
            expected_mean_copy_numbers={specie:2000},
            delta=50)
        # species counts, and cell mass and volume steadily decline
        prev_ckpt = None
        for time in MultialgorithmCheckpoint.list_checkpoints(self.results_dir):
            ckpt = MultialgorithmCheckpoint.get_checkpoint(self.results_dir, time=time)
            if prev_ckpt:
                prev_species_pops, prev_observables, prev_aggregate_state = RunResults.get_state_components(prev_ckpt.state)
                species_pops, observables, aggregate_state = RunResults.get_state_components(ckpt.state)
                self.assertTrue(species_pops[specie] < prev_species_pops[specie])
                self.assertTrue(aggregate_state['cell mass'] < prev_aggregate_state['cell mass'])
                self.assertTrue(aggregate_state['cell volume'] < prev_aggregate_state['cell volume'])
            prev_ckpt = ckpt

        self.perform_ssa_test_run('2 species, 1 reaction',
            run_time=1000,
            initial_specie_copy_numbers={'spec_type_0[compt_1]':3000, 'spec_type_1[compt_1]':0},
            expected_mean_copy_numbers={'spec_type_0[compt_1]':2000,  'spec_type_1[compt_1]':1000},
            delta=50)

    def test_runtime_errors(self):
        init_spec_type_0_pop = 2000
        # this model consumes all the reactants, driving propensities to 0:
        with self.assertRaisesRegexp(MultialgorithmError,
            "simulation with 1 SSA submodel and total propensities = 0 cannot progress"):
            self.perform_ssa_test_run('2 species, 1 reaction, with rates given by reactant population',
                run_time=500,
                initial_specie_copy_numbers={'spec_type_0[compt_1]':init_spec_type_0_pop, 'spec_type_1[compt_1]':0},
                expected_mean_copy_numbers={},
                delta=0,
                init_vols=1E-22)

    def test_run_multiple_ssa_submodels(self):
        # 1 submodel per compartment, no transfer reactions
        num_submodels = 3
        init_spec_type_0_pop = 200
        initial_specie_copy_numbers = {}
        expected_mean_copy_numbers = {}
        for i in range(num_submodels):
            compt_idx = i + 1
            specie_0_id = 'spec_type_0[compt_{}]'.format(compt_idx)
            specie_1_id = 'spec_type_1[compt_{}]'.format(compt_idx)
            initial_specie_copy_numbers[specie_0_id] = init_spec_type_0_pop
            initial_specie_copy_numbers[specie_1_id] = 0
            expected_mean_copy_numbers[specie_0_id] = 0
            expected_mean_copy_numbers[specie_1_id] = init_spec_type_0_pop

        self.perform_ssa_test_run('2 species, 1 reaction, with rates given by reactant population',
            num_submodels=num_submodels,
            run_time=1000,
            initial_specie_copy_numbers=initial_specie_copy_numbers,
            expected_mean_copy_numbers=expected_mean_copy_numbers,
            delta=0,
            init_vols=1E-22)

    def suspend_logging(self):
        self.saved_config = copy.deepcopy(config)
        DictUtil.set_value(config, 'level', 'ERROR')

    def restore_logging(self):
        global config
        config = self.saved_config

    def prep_simulation(self, num_ssa_submodels):
        model, multialgorithm_simulation, simulation_engine = self.make_model_and_simulation(
            '2 species, a pair of symmetrical reactions, and rates given by reactant population',
            num_ssa_submodels,
            init_vols=1E-18)
        local_species_pop = multialgorithm_simulation.local_species_population
        simulation_engine.initialize()
        return simulation_engine

    @unittest.skip("performance scaling test; runs slowly")
    def test_performance(self):
        end_sim_time = 100
        min_num_ssa_submodels = 2
        max_num_ssa_submodels = 32
        print()
        print("Performance test of SSA submodel simulation: 2 reactions per submodel; end simulation time: {}".format(end_sim_time))
        unprofiled_perf = ["\n#SSA submodels\t# events\trun time (s)\treactions/s".format()]

        self.suspend_logging()
        num_ssa_submodels = min_num_ssa_submodels
        while num_ssa_submodels <= max_num_ssa_submodels:

            # measure execution time
            simulation_engine = self.prep_simulation(num_ssa_submodels)
            start_time = time.process_time()
            num_events = simulation_engine.simulate(end_sim_time)
            run_time = time.process_time() - start_time
            unprofiled_perf.append("{}\t{}\t{:8.3f}\t{:8.3f}".format(num_ssa_submodels, num_events,
                run_time, num_events/run_time))

            # profile
            simulation_engine = self.prep_simulation(num_ssa_submodels)
            out_file = os.path.join(self.out_dir, "profile_out_{}.out".format(num_ssa_submodels))
            locals = {'simulation_engine':simulation_engine,
                'end_sim_time':end_sim_time}
            cProfile.runctx('num_events = simulation_engine.simulate(end_sim_time)', {}, locals, filename=out_file)
            profile = pstats.Stats(out_file)
            print("Profile for {} simulation objects:".format(num_ssa_submodels))
            profile.strip_dirs().sort_stats('cumulative').print_stats(15)

            num_ssa_submodels *= 4

        print('Performance summary')
        print("\n".join(unprofiled_perf))
        self.restore_logging()

class TestSSaExceptions(unittest.TestCase):

    def setUp(self):
        self.model = MakeModels.make_test_model('2 species, 1 reaction, with rates given by reactant population',
            specie_copy_numbers={'spec_type_0[compt_1]':10, 'spec_type_1[compt_1]':10})

    def test_nan_propensities(self):
        self.model.species_types.get_one(id='spec_type_0').molecular_weight = float('NaN')
        multialgorithm_simulation = MultialgorithmSimulation(self.model, {})
        simulation_engine, _ = multialgorithm_simulation.build_simulation()
        with self.assertRaisesRegexp(AssertionError, "total propensities is 'NaN'"):
            simulation_engine.initialize()

    # TODO(Arthur): test multiple ssa submodels, in shared or different compartments
    # TODO(Arthur): compare SSA submodel with published model
    # TODO(Arthur): test have identify_enabled_reactions() return a disabled reaction & ssa submodel with reactions that cannot run
    # TODO(Arthur): have if self.enabled_reaction(self.reactions[reaction_index]) do else branch
    # TODO(Arthur): review the cement programs
    # TODO(Arthur): update Docker image; use pip 10, pip install wc_sim --process-dependency-links
    # TODO(Arthur): handle concentration units: 2D conc, 3D conc, molecules
    # TODO(Arthur): restore and restart a simulation from a checkpoint
    # TODO(Arthur): use invariants to test saving aggregate values from DynamicModel in checkpoints
    # TODO(Arthur): delete unused parts of CheckpointLogger
    # TODO(Arthur): control pytest warnings

    # TODO(Arthur): catch MultialgorithmErrors from get_specie_concentrations, and elsewhere
    # TODO(Arthur): fit exponential to reaction, with rates given by reactant population
    # TODO(Arthur): perhaps raise warning for high concentration / molecule species like H20 in rate laws
