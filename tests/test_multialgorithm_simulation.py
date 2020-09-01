"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2017-02-08
:Copyright: 2017-2019, Karr Lab
:License: MIT
"""

from datetime import datetime
from scipy.constants import Avogadro
import copy
import cProfile
import numpy as np
import os
import pstats
import shutil
import tempfile
import time
import unittest

from de_sim.simulation_config import SimulationConfig
from wc_lang import Model
from wc_lang.io import Reader
from wc_lang.transform import PrepForWcSimTransform
from wc_onto import onto
from wc_sim.dynamic_components import DynamicModel
from wc_sim.multialgorithm_checkpointing import (MultialgorithmicCheckpointingSimObj,
                                                 MultialgorithmCheckpoint)
from wc_sim.multialgorithm_errors import (DynamicSpeciesPopulationError, MultialgorithmError,
                                          SpeciesPopulationError)
from wc_sim.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.run_results import RunResults
from wc_sim.sim_config import WCSimulationConfig
from wc_sim.simulation import Simulation
from wc_sim.species_populations import LocalSpeciesPopulation
from wc_sim.submodels.dfba import DfbaSubmodel
from wc_sim.submodels.nrm import NrmSubmodel
from wc_sim.submodels.odes import OdeSubmodel
from wc_sim.submodels.ssa import SsaSubmodel
from wc_sim.submodels.testing.deterministic_simulation_algorithm import DsaSubmodel
from wc_sim.testing.make_models import MakeModel
from wc_sim.testing.utils import check_simul_results, verify_independently_solved_model
from wc_utils.util.dict import DictUtil
from wc_utils.util.rand import RandomStateManager


# TODO(Arthur): plots of DSA with mean and instances of SSA monte Carlo
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


class TestMultialgorithmSimulationStatically(unittest.TestCase):

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures', 'test_model.xlsx')

    def setUp(self):
        # read and initialize a model
        self.model = Reader().run(self.MODEL_FILENAME, ignore_extra_models=True)[Model][0]
        for conc in self.model.distribution_init_concentrations:
            conc.std = 0.
        PrepForWcSimTransform().run(self.model)
        de_simulation_config = SimulationConfig(max_time=10)
        self.wc_sim_config = WCSimulationConfig(de_simulation_config, dfba_time_step=1)
        self.multialgorithm_simulation = MultialgorithmSimulation(self.model, self.wc_sim_config)
        self.test_dir = tempfile.mkdtemp()
        self.results_dir = tempfile.mkdtemp(dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_init(self):
        self.model.submodels = []
        with self.assertRaises(MultialgorithmError):
            MultialgorithmSimulation(self.model, self.wc_sim_config)

    def test_prepare_skipped_submodels(self):
        multialgorithm_simulation = MultialgorithmSimulation(self.model, self.wc_sim_config)
        self.assertEqual(multialgorithm_simulation.skipped_submodels(), set())
        submodels_to_skip = ['submodel_1']
        self.wc_sim_config.submodels_to_skip = submodels_to_skip
        multialgorithm_simulation = MultialgorithmSimulation(self.model, self.wc_sim_config)
        self.assertEqual(multialgorithm_simulation.skipped_submodels(), set(submodels_to_skip))

        submodels_to_skip = ['no_such_submodel']
        self.wc_sim_config.submodels_to_skip = submodels_to_skip
        with self.assertRaisesRegex(MultialgorithmError,
                                    "'submodels_to_skip' contains submodels that aren't in the model:"):
            MultialgorithmSimulation(self.model, self.wc_sim_config)

    def test_molecular_weights_for_species(self):
        multi_alg_sim = self.multialgorithm_simulation
        expected = {
            'species_6[c]': float('nan'),
            'H2O[c]': 18.0152
        }
        actual = multi_alg_sim.molecular_weights_for_species(set(expected.keys()))
        self.assertEqual(actual['H2O[c]'], expected['H2O[c]'])
        self.assertTrue(np.isnan(actual['species_6[c]']))

        # add a species_type without a structure
        species_type_wo_structure = self.model.species_types.create(
            id='st_wo_structure',
            name='st_wo_structure')
        cellular_compartment = self.model.compartments.get(**{'id': 'c'})[0]
        species_wo_structure = self.model.species.create(
            species_type=species_type_wo_structure,
            compartment=cellular_compartment)
        species_wo_structure.id = species_wo_structure.gen_id()

        actual = multi_alg_sim.molecular_weights_for_species([species_wo_structure.id])
        self.assertTrue(np.isnan(actual[species_wo_structure.id]))

        # test obtain weights for all species
        actual = multi_alg_sim.molecular_weights_for_species()
        self.assertEqual(actual['H2O[c]'], expected['H2O[c]'])
        self.assertTrue(np.isnan(actual['species_6[c]']))
        self.assertEqual(len(actual), len(self.model.get_species()))

    def test_create_dynamic_compartments(self):
        self.multialgorithm_simulation.create_dynamic_compartments()
        self.assertEqual(set(['c', 'e']), set(self.multialgorithm_simulation.temp_dynamic_compartments))
        for id, dynamic_compartment in self.multialgorithm_simulation.temp_dynamic_compartments.items():
            self.assertEqual(id, dynamic_compartment.id)
            self.assertTrue(0 < dynamic_compartment.init_density)

    def test_prepare_dynamic_compartments(self):
        self.multialgorithm_simulation.initialize_components()
        self.multialgorithm_simulation.initialize_infrastructure()
        for dynamic_compartment in self.multialgorithm_simulation.dynamic_model.dynamic_compartments.values():
            self.assertTrue(dynamic_compartment._initialized())
            self.assertTrue(0 < dynamic_compartment.accounted_mass())
            self.assertTrue(0 < dynamic_compartment.mass())

    def test_init_species_pop_from_distribution(self):
        self.multialgorithm_simulation.create_dynamic_compartments()
        self.multialgorithm_simulation.init_species_pop_from_distribution()
        species_wo_init_conc = ['species_1[c]', 'species_3[c]']
        for species_id in species_wo_init_conc:
            self.assertEqual(self.multialgorithm_simulation.init_populations[species_id], 0)
        for concentration in self.model.get_distribution_init_concentrations():
            self.assertTrue(0 <= self.multialgorithm_simulation.init_populations[concentration.species.id])

        # todo: statistically evaluate sampled population
        # ensure that over multiple runs of init_species_pop_from_distribution():
        # mean(species population) ~= mean(volume) * mean(concentration)

    def test_make_local_species_population(self):
        self.multialgorithm_simulation.create_dynamic_compartments()
        self.multialgorithm_simulation.init_species_pop_from_distribution()
        local_species_population = self.multialgorithm_simulation.make_local_species_population()
        self.assertEqual(local_species_population._molecular_weights,
            self.multialgorithm_simulation.molecular_weights_for_species())

        # test the initial population slopes
        # continuous adjustments are only allowed on species used by continuous submodels
        used_by_continuous_submodels = \
            ['species_1[e]', 'species_2[e]', 'species_1[c]', 'species_2[c]', 'species_3[c]']
        adjustments = {species_id: 0. for species_id in used_by_continuous_submodels}
        self.assertEqual(local_species_population.adjust_continuously(1, adjustments), None)
        not_in_a_reaction = ['H2O[e]', 'H2O[c]']
        used_by_discrete_submodels = ['species_4[c]', 'species_5[c]', 'species_6[c]']
        adjustments = {species_id: 0. for species_id in used_by_discrete_submodels + not_in_a_reaction}
        with self.assertRaises(DynamicSpeciesPopulationError):
            local_species_population.adjust_continuously(2, adjustments)

    def test_set_simultaneous_execution_priorities(self):
        expected_order_of_sim_obj_classes = [SsaSubmodel,
                                             NrmSubmodel,
                                             DsaSubmodel,
                                             DfbaSubmodel,
                                             OdeSubmodel,
                                             MultialgorithmicCheckpointingSimObj]
        self.multialgorithm_simulation.set_simultaneous_execution_priorities()
        # ensure that expected_order_of_sim_obj_classes are arranged in decreasing priority
        for i in range(len(expected_order_of_sim_obj_classes) - 1):
            simulation_object_class = expected_order_of_sim_obj_classes[i]
            next_simulation_object_class = expected_order_of_sim_obj_classes[i+1]
            self.assertLess(simulation_object_class.metadata.class_priority,
                            next_simulation_object_class.metadata.class_priority)

    def test_initialize_components(self):
        self.multialgorithm_simulation.initialize_components()
        self.assertTrue(isinstance(self.multialgorithm_simulation.local_species_population,
                        LocalSpeciesPopulation))

    def test_initialize_infrastructure(self):
        self.multialgorithm_simulation.initialize_components()
        self.multialgorithm_simulation.initialize_infrastructure()
        self.assertTrue(isinstance(self.multialgorithm_simulation.dynamic_model, DynamicModel))

        for dynamic_compartment in self.multialgorithm_simulation.dynamic_model.dynamic_compartments.values():
            self.assertTrue(isinstance(dynamic_compartment.species_population, LocalSpeciesPopulation))

        de_simulation_config = SimulationConfig(max_time=10, output_dir=self.results_dir)
        wc_sim_config = WCSimulationConfig(de_simulation_config, dfba_time_step=1, checkpoint_period=10)
        multialg_sim = MultialgorithmSimulation(self.model, wc_sim_config)
        multialg_sim.initialize_components()
        multialg_sim.initialize_infrastructure()
        self.assertEqual(multialg_sim.checkpointing_sim_obj.checkpoint_dir, self.results_dir)
        self.assertTrue(multialg_sim.checkpointing_sim_obj.access_state_object is not None)
        self.assertTrue(isinstance(multialg_sim.checkpointing_sim_obj, MultialgorithmicCheckpointingSimObj))
        self.assertTrue(isinstance(multialg_sim.dynamic_model, DynamicModel))

    def test_build_simulation(self):
        de_simulation_config = SimulationConfig(max_time=10, output_dir=self.results_dir)
        wc_sim_config = WCSimulationConfig(de_simulation_config, dfba_time_step=1, checkpoint_period=10)
        multialgorithm_simulation = MultialgorithmSimulation(self.model, wc_sim_config)
        simulator, _ = multialgorithm_simulation.build_simulation()
        # 3 objects: 2 submodels, and the checkpointing obj:
        expected_sim_objs = set(['CHECKPOINTING_SIM_OBJ', 'submodel_1', 'submodel_2'])
        self.assertEqual(expected_sim_objs, set(list(simulator.simulation_objects)))
        self.assertEqual(type(multialgorithm_simulation.checkpointing_sim_obj),
                         MultialgorithmicCheckpointingSimObj)
        self.assertEqual(multialgorithm_simulation.dynamic_model.get_num_submodels(), 2)

        # check that submodels receive options
        dfba_options = dict(dfba='fast but inaccurate')
        ssa_options = dict(ssa='accurate but slow')
        options = {'DfbaSubmodel': dict(options=dfba_options),
                   'SsaSubmodel': dict(options=ssa_options)
                  }
        multialgorithm_simulation = MultialgorithmSimulation(self.model, wc_sim_config, options)
        multialgorithm_simulation.build_simulation()
        dfba_submodel = multialgorithm_simulation.dynamic_model.dynamic_submodels['submodel_1']
        ssa_submodel = multialgorithm_simulation.dynamic_model.dynamic_submodels['submodel_2']
        self.assertEqual(dfba_submodel.options, dfba_options)
        self.assertEqual(ssa_submodel.options, ssa_options)

        # test skipped submodel
        submodels_to_skip = ['submodel_2']
        self.wc_sim_config.submodels_to_skip = submodels_to_skip
        ma_sim = MultialgorithmSimulation(self.model, self.wc_sim_config)
        _, dynamic_model = ma_sim.build_simulation()
        expected_dynamic_submodels = set([sm.id for sm in self.model.get_submodels()]) - ma_sim.skipped_submodels()
        self.assertEqual(expected_dynamic_submodels, set(dynamic_model.dynamic_submodels))

        submodel_1 = self.model.submodels.get(id='submodel_1')[0]
        # WC:modeling_framework is not an instance of a modeling framework
        submodel_1.framework = onto['WC:modeling_framework']
        ma_sim = MultialgorithmSimulation(self.model, self.wc_sim_config)
        with self.assertRaisesRegex(MultialgorithmError, 'Unsupported lang_submodel framework'):
            ma_sim.build_simulation()

    def test_get_dynamic_compartments(self):
        expected_compartments = dict(
            submodel_1=['c', 'e'],
            submodel_2=['c']
        )
        self.multialgorithm_simulation.build_simulation()
        for submodel_id in ['submodel_1', 'submodel_2']:
            submodel = self.model.submodels.get_one(id=submodel_id)
            submodel_dynamic_compartments = self.multialgorithm_simulation.get_dynamic_compartments(submodel)
            self.assertEqual(set(submodel_dynamic_compartments.keys()), set(expected_compartments[submodel_id]))

    def test_str(self):
        self.multialgorithm_simulation.create_dynamic_compartments()
        self.multialgorithm_simulation.init_species_pop_from_distribution()
        self.multialgorithm_simulation.local_species_population = \
            self.multialgorithm_simulation.make_local_species_population(retain_history=False)
        self.assertIn('species_1[e]', str(self.multialgorithm_simulation))
        self.assertIn('model:', str(self.multialgorithm_simulation))


class TestMultialgorithmSimulationDynamically(unittest.TestCase):
    """
    Approach:
        Test dynamics:
            Species trajectories
            Property trajectories:
                mass
                volume
                accounted mass
                accounted volume
                density, which should be constant
        Two tests:
            Deterministic: with initial distributions that have standard deviation = 0
            Stochastic: with 0<stds for initial distributions
    """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.results_dir = tempfile.mkdtemp(dir=self.tmp_dir)
        self.models = ['static', 'one_reaction_linear', 'one_rxn_exponential', 'one_exchange_rxn_compt_growth',
                       'stop_conditions']
        de_simulation_config = SimulationConfig(max_time=20, output_dir=tempfile.mkdtemp(dir=self.tmp_dir))
        self.wc_sim_config = WCSimulationConfig(de_simulation_config, checkpoint_period=1)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    @unittest.skip('sbml_io.SbmlWriter fails')
    def test_convert_to_sbml(self):
        from wc_lang.sbml import io as sbml_io
        from wc_lang.io import Reader, Writer
        from wc_sim.testing.utils import read_model_for_test
        for model_name in self.models:
            print(f'converting {model_name}')
            dirname = os.path.join(os.path.dirname(__file__), 'fixtures', 'dynamic_tests')
            model_filename = os.path.join(dirname, f'{model_name}.xlsx')
            integration_framework = 'ordinary_differential_equations'
            model = read_model_for_test(model_filename, integration_framework=f'WC:{integration_framework}')

            # write wc_lang file
            sbml_dirname = os.path.join(dirname, f'{model_name}_sbml')
            if not os.path.isdir(sbml_dirname):
                os.makedirs(sbml_dirname)

            lang_filename = os.path.join(sbml_dirname, f'{model_name}.xlsx')
            print(f'writing {lang_filename}')
            Writer().run(lang_filename, model)
            model_from_file = Reader().run(lang_filename)[Model][0]
            assert model_from_file.validate() is None

            # write SBML file
            sbml_io.SbmlWriter().run(model_from_file, sbml_dirname)

    def test_independently_solved_models(self):
        for model_name in self.models:
            model_filename = os.path.join(os.path.dirname(__file__), 'fixtures', 'dynamic_tests', f'{model_name}.xlsx')
            verify_independently_solved_model(self, model_filename, self.results_dir)

    def test_one_reaction_constant_species_pop(self):
        # test statics
        init_volume = 1E-16
        init_density = 1000
        molecular_weight = 100.
        default_species_copy_number = 10_000
        init_accounted_mass = molecular_weight * default_species_copy_number / Avogadro
        init_accounted_density = init_accounted_mass / init_volume
        expected_initial_values_compt_1 = dict(init_volume=init_volume,
                                               init_accounted_mass=init_accounted_mass,
                                               init_mass= init_volume * init_density,
                                               init_density=init_density,
                                               init_accounted_density=init_accounted_density,
                                               accounted_fraction = init_accounted_density / init_density)
        expected_initial_values = {'compt_1': expected_initial_values_compt_1}
        model = MakeModel.make_test_model('1 species, 1 reaction',
                                          init_vols=[expected_initial_values_compt_1['init_volume']],
                                          init_vol_stds=[0],
                                          density=init_density,
                                          molecular_weight=molecular_weight,
                                          default_species_copy_number=default_species_copy_number,
                                          default_species_std=0)
        multialgorithm_simulation = MultialgorithmSimulation(model, self.wc_sim_config)
        _, dynamic_model = multialgorithm_simulation.build_simulation()
        check_simul_results(self, dynamic_model, None, expected_initial_values=expected_initial_values)


# TODO: clean up this class: move the good parts to test_ssa.py, and discard the rest
class TestRunSSASimulation(unittest.TestCase):

    def setUp(self):
        self.results_dir = tempfile.mkdtemp()
        de_simulation_config = SimulationConfig(max_time=10, output_dir=self.results_dir)
        self.wc_sim_config = WCSimulationConfig(de_simulation_config, dfba_time_step=1, checkpoint_period=10)
        self.out_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.results_dir)
        shutil.rmtree(self.out_dir)

    def make_model_and_simulation(self, model_type, num_submodels, species_copy_numbers=None,
                                  species_stds=None, init_vols=None,
                                  submodel_framework='WC:stochastic_simulation_algorithm'):
        # make simple model
        if init_vols is not None:
            if not isinstance(init_vols, list):
                init_vols = [init_vols]*num_submodels
        model = MakeModel.make_test_model(model_type, num_submodels=num_submodels,
                                          species_copy_numbers=species_copy_numbers,
                                          species_stds=species_stds,
                                          init_vols=init_vols,
                                          submodel_framework=submodel_framework)
        multialgorithm_simulation = MultialgorithmSimulation(model, self.wc_sim_config)
        simulator, _ = multialgorithm_simulation.build_simulation()
        return (model, multialgorithm_simulation, simulator)

    def checkpoint_times(self, run_time):
        """ Provide expected checkpoint times for a simulation
        """
        checkpoint_times = []
        t = 0
        while t <= run_time:
            checkpoint_times.append(t)
            t += self.wc_sim_config.checkpoint_period
        return checkpoint_times

    def perform_ssa_test_run(self, model_type, run_time, initial_species_copy_numbers, initial_species_stds,
                             expected_mean_copy_numbers, delta, num_submodels=1, invariants=None,
                             iterations=3, init_vols=None):
        """ Test SSA by comparing expected and actual simulation copy numbers

        Args:
            model_type (:obj:`str`): model type description
            run_time (:obj:`float`): duration of the simulation run
            initial_species_copy_numbers (:obj:`dict`): initial species counts, with IDs as keys and counts as values
            expected_mean_copy_numbers (:obj:`dict`): expected final mean species counts, in same format as
                `initial_species_copy_numbers`
            delta (:obj:`int`): maximum allowed difference between expected and actual counts
            num_submodels (:obj:`int`): number of submodels to create
            invariants (:obj:`list`, optional): list of invariant relationships, to be tested
            iterations (:obj:`int`, optional): number of simulation runs
            init_vols (:obj:`float`, `list` of `floats`, optional): initial volume of compartment(s)
                if reaction rates depend on concentration, use a smaller volume to increase rates
        """
        # TODO(Arthur): provide some invariant objects
        invariant_objs = [] if invariants is None else [Invariant(value) for value in invariants]

        final_species_counts = []
        for i in range(iterations):
            model, multialgorithm_simulation, simulator = self.make_model_and_simulation(
                model_type,
                num_submodels=num_submodels,
                species_copy_numbers=initial_species_copy_numbers,
                species_stds=initial_species_stds,
                init_vols=init_vols)
            local_species_pop = multialgorithm_simulation.local_species_population
            simulator.initialize()
            simulator.simulate(run_time)
            final_species_counts.append(local_species_pop.read(run_time))

        mean_final_species_counts = dict.fromkeys(list(initial_species_copy_numbers.keys()), 0)
        if expected_mean_copy_numbers:
            for final_species_count in final_species_counts:
                for k, v in final_species_count.items():
                    mean_final_species_counts[k] += v
            for k, v in mean_final_species_counts.items():
                mean_final_species_counts[k] = v/iterations
                if k not in mean_final_species_counts:
                    print(k,  'not in mean_final_species_counts',  list(mean_final_species_counts.keys()))
                if k not in expected_mean_copy_numbers:
                    print(k,  'not in expected_mean_copy_numbers',  list(expected_mean_copy_numbers.keys()))
                self.assertAlmostEqual(mean_final_species_counts[k], expected_mean_copy_numbers[k], delta=delta)
        for invariant_obj in invariant_objs:
            self.assertTrue(invariant_obj.eval())

        # check the checkpoint times
        self.assertEqual(MultialgorithmCheckpoint.list_checkpoints(self.results_dir),
                         self.checkpoint_times(run_time))

    @unittest.skip('debug')
    def test_run_ssa_suite(self):
        species = 'spec_type_0[compt_1]'
        # tests checkpoint history in which the last checkpoint time < run time
        self.perform_ssa_test_run('1 species, 1 reaction',
                                  run_time=999,
                                  initial_species_copy_numbers={species: 3000},
                                  initial_species_stds={species: 0},
                                  expected_mean_copy_numbers={species: 2000},
                                  delta=50)
        # species counts, and cell mass and volume steadily decline
        prev_ckpt = None
        for time in MultialgorithmCheckpoint.list_checkpoints(self.results_dir):
            ckpt = MultialgorithmCheckpoint.get_checkpoint(self.results_dir, time=time)
            if prev_ckpt:
                prev_species_pops, prev_observables, prev_functions, prev_aggregate_state = \
                    RunResults.get_state_components(prev_ckpt.state)
                species_pops, observables, functions, aggregate_state = RunResults.get_state_components(ckpt.state)
                self.assertTrue(species_pops[species] < prev_species_pops[species])
                self.assertTrue(aggregate_state['cell mass'] < prev_aggregate_state['cell mass'])
            prev_ckpt = ckpt

        self.perform_ssa_test_run('2 species, 1 reaction',
            run_time=1000,
            initial_species_copy_numbers={'spec_type_0[compt_1]': 3000, 'spec_type_1[compt_1]': 0},
            initial_species_stds={'spec_type_0[compt_1]': 0, 'spec_type_1[compt_1]': 0},
            expected_mean_copy_numbers={'spec_type_0[compt_1]': 2000,  'spec_type_1[compt_1]': 1000},
            delta=50)

    @unittest.skip('debug')
    def test_run_multiple_ssa_submodels(self):
        # 1 submodel per compartment, no transfer reactions
        num_submodels = 3
        init_spec_type_0_pop = 200
        initial_species_copy_numbers = {}
        initial_species_stds = {}
        expected_mean_copy_numbers = {}
        for i in range(num_submodels):
            compt_idx = i + 1
            species_0_id = 'spec_type_0[compt_{}]'.format(compt_idx)
            species_1_id = 'spec_type_1[compt_{}]'.format(compt_idx)
            initial_species_copy_numbers[species_0_id] = init_spec_type_0_pop
            initial_species_copy_numbers[species_1_id] = 0
            initial_species_stds[species_0_id] = 0
            initial_species_stds[species_1_id] = 0
            expected_mean_copy_numbers[species_0_id] = 0
            expected_mean_copy_numbers[species_1_id] = init_spec_type_0_pop

        self.perform_ssa_test_run('2 species, 1 reaction, with rates given by reactant population',
                                  num_submodels=num_submodels,
                                  run_time=1000,
                                  initial_species_copy_numbers=initial_species_copy_numbers,
                                  initial_species_stds=initial_species_stds,
                                  expected_mean_copy_numbers=expected_mean_copy_numbers,
                                  delta=0,
                                  init_vols=1E-22)

    def prep_simulation(self, num_ssa_submodels, submodel_framework='WC:stochastic_simulation_algorithm'):
        model, multialgorithm_simulation, simulator = self.make_model_and_simulation(
            '2 species, a pair of symmetrical reactions, and rates given by reactant population',
            num_ssa_submodels,
            init_vols=1E-18,
            submodel_framework=submodel_framework)
        local_species_pop = multialgorithm_simulation.local_species_population
        simulator.initialize()
        return simulator

    # @unittest.skip("performance scaling test; runs slowly")
    def test_performance(self):
        end_sim_time = 100
        min_num_ssa_submodels = 2
        max_num_ssa_submodels = 32
        for submodel_framework in ['WC:stochastic_simulation_algorithm', 'WC:next_reaction_method']:

            framework_name = onto[submodel_framework].name.title()
            framework_synonyms = onto[submodel_framework].synonyms
            framework_synonym = next(iter(framework_synonyms)).description

            print(f"Performance test of {framework_name} submodel simulation:\n"
                  f"    2 reactions per submodel; end simulation time: {end_sim_time}")
            unprofiled_perf = [f"\n# {framework_synonym} submodels\t# events\trun time (s)\treactions/s"]

            num_ssa_submodels = min_num_ssa_submodels
            while num_ssa_submodels <= max_num_ssa_submodels:

                # measure execution time
                simulator = self.prep_simulation(num_ssa_submodels,
                                                         submodel_framework=submodel_framework)
                start_time = time.process_time()
                num_events = simulator.simulate(end_sim_time).num_events
                run_time = time.process_time() - start_time
                unprofiled_perf.append("{}\t{}\t{:8.3f}\t{:8.3f}".format(num_ssa_submodels, num_events,
                                                                         run_time, num_events/run_time))

                # profile
                simulator = self.prep_simulation(num_ssa_submodels,
                                                         submodel_framework=submodel_framework)
                out_file = os.path.join(self.out_dir, "profile_out_{}.out".format(num_ssa_submodels))
                locals = {'simulator': simulator,
                          'end_sim_time': end_sim_time}
                cProfile.runctx('simulator.simulate(end_sim_time)', {}, locals, filename=out_file)
                profile = pstats.Stats(out_file)
                print(f"Profile for {num_ssa_submodels} instances of {framework_name}:")
                profile.strip_dirs().sort_stats('cumulative').print_stats(15)

                num_ssa_submodels *= 4

            performance_log = os.path.join(os.path.dirname(__file__), 'perf_results',
                                           'wc_sim_performance_log.txt')
        
            if not os.path.isdir(os.path.dirname(performance_log)):
                os.makedirs(os.path.dirname(performance_log))

            today = datetime.today().strftime('%Y-%m-%d')
            summary = f'Performance summary for {framework_name} on {today}:' + \
                      "\n".join(unprofiled_perf) + "\n"

            with open(performance_log, 'a') as perf_log:
                print(summary, file=perf_log)

            print(f'Performance summary, written to {performance_log}:\n')
            print(summary)

    # TODO(Arthur): test multiple ssa submodels, in shared or different compartments
    # TODO(Arthur): test have identify_enabled_reactions() return a disabled reaction & ssa submodel with reactions that cannot run
    # TODO(Arthur): use invariants to test saving aggregate values from DynamicModel in checkpoints
    # TODO(Arthur): catch MultialgorithmErrors from get_species_counts, and elsewhere
