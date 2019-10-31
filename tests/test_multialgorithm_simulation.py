"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2017-02-08
:Copyright: 2017-2019, Karr Lab
:License: MIT
"""

from collections import defaultdict
from pprint import pprint
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

import obj_tables
from wc_lang import Model
from wc_lang.io import Reader
from wc_lang.transform import PrepForWcSimTransform
from wc_onto import onto
from wc_sim.dynamic_components import DynamicModel
from wc_sim.multialgorithm_checkpointing import (MultialgorithmicCheckpointingSimObj,
                                                 MultialgorithmCheckpoint)
from wc_sim.multialgorithm_errors import MultialgorithmError, SpeciesPopulationError
from wc_sim.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.run_results import RunResults
from wc_sim.simulation import Simulation
from wc_sim.species_populations import LocalSpeciesPopulation
from wc_sim.testing.make_models import MakeModel
from wc_sim.testing.utils import read_model_and_set_all_std_devs_to_0
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
        self.args = dict(dfba_time_step=1,
                         results_dir=None)
        self.multialgorithm_simulation = MultialgorithmSimulation(self.model, self.args)
        self.test_dir = tempfile.mkdtemp()
        self.results_dir = tempfile.mkdtemp(dir=self.test_dir)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

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
        self.assertEqual(set(['c', 'e']), set(self.multialgorithm_simulation.dynamic_compartments))
        for id, dynamic_compartment in self.multialgorithm_simulation.dynamic_compartments.items():
            self.assertEqual(id, dynamic_compartment.id)
            self.assertTrue(0 < dynamic_compartment.init_density)

    def test_prepare_dynamic_compartments(self):
        self.multialgorithm_simulation.create_dynamic_compartments()
        self.multialgorithm_simulation.initialize_species_populations()
        self.multialgorithm_simulation.local_species_population = \
            self.multialgorithm_simulation.make_local_species_population(retain_history=False)
        self.multialgorithm_simulation.prepare_dynamic_compartments()
        for dynamic_compartment in self.multialgorithm_simulation.dynamic_compartments.values():
            self.assertTrue(dynamic_compartment._initialized())
            self.assertTrue(0 < dynamic_compartment.accounted_mass())
            self.assertTrue(0 < dynamic_compartment.mass())

    def test_initialize_species_populations(self):
        self.multialgorithm_simulation.create_dynamic_compartments()
        self.multialgorithm_simulation.initialize_species_populations()
        species_wo_init_conc = ['species_1[c]', 'species_3[c]']
        for species_id in species_wo_init_conc:
            self.assertEqual(self.multialgorithm_simulation.init_populations[species_id], 0)
        for concentration in self.model.get_distribution_init_concentrations():
            self.assertTrue(0 <= self.multialgorithm_simulation.init_populations[concentration.species.id])

        # todo: statistically evaluate sampled population
        # ensure that over multiple runs of initialize_species_populations():
        # mean(species population) ~= mean(volume) * mean(concentration)

    def test_make_local_species_population(self):
        self.multialgorithm_simulation.create_dynamic_compartments()
        self.multialgorithm_simulation.initialize_species_populations()
        local_species_population = self.multialgorithm_simulation.make_local_species_population()
        self.assertEqual(local_species_population._molecular_weights,
            self.multialgorithm_simulation.molecular_weights_for_species())

        # test the initial fluxes
        # continuous adjustments are only allowed on species used by continuous submodels
        used_by_continuous_submodels = \
            ['species_1[e]', 'species_2[e]', 'species_1[c]', 'species_2[c]', 'species_3[c]']
        adjustments = {species_id: (0, 0) for species_id in used_by_continuous_submodels}
        self.assertEqual(local_species_population.adjust_continuously(1, adjustments), None)
        not_in_a_reaction = ['H2O[e]', 'H2O[c]']
        used_by_discrete_submodels = ['species_4[c]', 'species_5[c]', 'species_6[c]']
        adjustments = {species_id: (0, 0) for species_id in used_by_discrete_submodels + not_in_a_reaction}
        with self.assertRaises(SpeciesPopulationError):
            local_species_population.adjust_continuously(2, adjustments)

    def test_initialize_components(self):
        self.multialgorithm_simulation.initialize_components()
        self.assertTrue(isinstance(self.multialgorithm_simulation.local_species_population,
                        LocalSpeciesPopulation))
        for dynamic_compartment in self.multialgorithm_simulation.dynamic_compartments.values():
            self.assertTrue(isinstance(dynamic_compartment.species_population, LocalSpeciesPopulation))

    def test_initialize_infrastructure(self):
        self.multialgorithm_simulation.initialize_components()
        self.multialgorithm_simulation.initialize_infrastructure()
        self.assertTrue(isinstance(self.multialgorithm_simulation.dynamic_model, DynamicModel))

        args = dict(dfba_time_step=1,
                    results_dir=self.results_dir,
                    checkpoint_period=10)
        multialg_sim = MultialgorithmSimulation(self.model, args)
        multialg_sim.initialize_components()
        multialg_sim.initialize_infrastructure()
        self.assertEqual(multialg_sim.checkpointing_sim_obj.checkpoint_dir, self.results_dir)
        self.assertTrue(multialg_sim.checkpointing_sim_obj.access_state_object is not None)
        self.assertTrue(isinstance(multialg_sim.checkpointing_sim_obj, MultialgorithmicCheckpointingSimObj))
        self.assertTrue(isinstance(multialg_sim.dynamic_model, DynamicModel))

    def test_build_simulation(self):
        args = dict(dfba_time_step=1,
                    results_dir=self.results_dir,
                    checkpoint_period=10)
        multialgorithm_simulation = MultialgorithmSimulation(self.model, args)
        simulation_engine, _ = multialgorithm_simulation.build_simulation()
        # 3 objects: 2 submodels, and the checkpointing obj:
        expected_sim_objs = set(['CHECKPOINTING_SIM_OBJ', 'submodel_1', 'submodel_2'])
        self.assertEqual(expected_sim_objs, set(list(simulation_engine.simulation_objects)))
        self.assertEqual(type(multialgorithm_simulation.checkpointing_sim_obj),
                         MultialgorithmicCheckpointingSimObj)
        self.assertEqual(multialgorithm_simulation.dynamic_model.get_num_submodels(), 2)
        self.assertTrue(callable(simulation_engine.stop_condition))

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
        self.multialgorithm_simulation.initialize_species_populations()
        self.multialgorithm_simulation.local_species_population = \
            self.multialgorithm_simulation.make_local_species_population(retain_history=False)
        self.assertIn('species_1[e]', str(self.multialgorithm_simulation))
        self.assertIn('model:', str(self.multialgorithm_simulation))


class TestMultialgorithmSimulationDynamically(unittest.TestCase):
    """
    Approach:
        Test dynamics:
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
        self.args = dict(results_dir=tempfile.mkdtemp(dir=self.tmp_dir),
                         checkpoint_period=1)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    # todo: move this into wc_sim/testing/utils.py
    # todo: test that compartment densities remain constant
    def check_simul_results(self, dynamic_model, results_dir, expected_initial_values=None,
                            expected_times=None, expected_species_trajectories=None,
                            expected_property_trajectories=None, delta=None):
        """ Evaluate whether a simulation predicted the expected results

        The expected trajectories must contain expected values at the times of the simulation's checkpoints.

        Args:
            dynamic_model (:obj:`DynamicModel`): the simulation's :obj:`DynamicModel`
            results_dir (:obj:`str`): simulation results directory
            expected_initial_values (:obj:`dict`, optional): expected initial values for each compartment
                in the test; indexed by compartment id, and attribute names for :obj:`DynamicCompartment`\ s
            expected_times (:obj:`iterator`, optional): expected time sequence for all trajectories
            expected_species_trajectories (:obj:`dict`, optional): expected trajectories of species
                copy numbers in the test; provides an iterator for each species, indexed by the species id
            expected_property_trajectories (:obj:`dict`, optional): expected trajectories of aggregate properties
                 for each compartment in the test; provides an iterator for each property; indexed by
                 compartment id. Properties must be keyed as structured by `AccessState.get_checkpoint_state()`
                 and `RunResults.convert_checkpoints()`
            delta (:obj:`bool`, optional): if set, then compare trajectory values approximately, reporting
                values that differ by more than `delta` as different
        """

        # test initial values
        if expected_initial_values:
            for compartment_id in expected_initial_values:
                dynamic_compartment = dynamic_model.dynamic_compartments[compartment_id]
                for initial_attribute, expected_value in expected_initial_values[compartment_id].items():
                    actual_value = getattr(dynamic_compartment, initial_attribute)
                    if delta:
                        self.assertAlmostEqual(actual_value, expected_value, delta=delta,
                                         msg=f"In model {dynamic_model.id}, {initial_attribute} is {actual_value}, "
                                            f"not within {delta} of {expected_value}")
                    else:
                        self.assertEqual(actual_value, expected_value,
                                         msg=f"In model {dynamic_model.id}, {initial_attribute} is {actual_value}, "
                                            f"not {expected_value}")

        # get results
        if expected_species_trajectories or expected_property_trajectories:
            run_results = RunResults(results_dir)
            populations_df = run_results.get('populations')
            aggregate_states_df = run_results.get('aggregate_states')

        # test expected_times
        if expected_times and not np.isnan(expected_times).all():
            np.testing.assert_array_equal(populations_df.index, expected_times,
                                          err_msg=f"In model {dynamic_model.id}, time sequence for populations "
                                          f"differs from expected time sequence:")
            np.testing.assert_array_equal(aggregate_states_df.index, expected_times,
                                          err_msg=f"In model {dynamic_model.id}, time sequence for aggregate states "
                                          f"differs from expected time sequence:")


        # test expected trajectories of species
        if expected_species_trajectories:
            for specie_id, expected_trajectory in expected_species_trajectories.items():
                if np.isnan(expected_trajectory).all():
                    continue
                actual_trajectory = populations_df[specie_id]
                if delta:
                    np.testing.assert_allclose(actual_trajectory, expected_trajectory, equal_nan=False,
                                               atol=delta,
                                               err_msg=f"In model {dynamic_model.id}, trajectory for {specie_id} "
                                                   f"not almost equal to expected trajectory:")
                else:
                    np.testing.assert_array_equal(actual_trajectory, expected_trajectory,
                                                  err_msg=f"In model {dynamic_model.id}, trajectory for {specie_id} "
                                                      f"differs from expected trajectory:")

        # test expected trajectories of properties of compartments
        if expected_property_trajectories:
            # get all properties
            properties = set()
            for property_array_dict in expected_property_trajectories.values():
                properties.update(property_array_dict.keys())
            for compartment_id in expected_property_trajectories:
                dynamic_compartment = dynamic_model.dynamic_compartments[compartment_id]
                for property in properties:
                    if compartment_id not in aggregate_states_df:
                        continue
                    actual_trajectory = aggregate_states_df[compartment_id][property]
                    expected_trajectory = expected_property_trajectories[compartment_id][property]
                    if np.isnan(expected_trajectory).all():
                        continue
                    if not delta:
                        delta = 1E-9
                    # todo: investigate possible numpy bug: without list(), this fails with
                    # "TypeError: ufunc 'isfinite' not supported for the input types, ..." but ndarray works elsewhere
                    # todo: when fixed, remove list()
                    np.testing.assert_allclose(list(actual_trajectory.to_numpy()), expected_trajectory, equal_nan=False,
                                               atol=delta,
                                               err_msg=f"In model {dynamic_model.id}, trajectory of {property} "
                                                  f"of compartment {compartment_id} "
                                                  f"not almost equal to expected trajectory:")

    # todo: move this into tests/testing/utils.py
    def test_check_simul_results(self):
        # test check_simul_results above
        init_volume = 1E-16
        init_density = 1000
        molecular_weight = 100.
        default_species_copy_number = 10_000
        init_accounted_mass = molecular_weight * default_species_copy_number / Avogadro
        init_accounted_density = init_accounted_mass / init_volume
        expected_initial_values_compt_1 = dict(
            init_volume=init_volume,
            init_accounted_mass=init_accounted_mass,
            init_mass= init_volume * init_density,
            init_density=init_density,
            init_accounted_density=init_accounted_density,
            accounted_fraction = init_accounted_density / init_density
        )
        expected_initial_values = {'compt_1': expected_initial_values_compt_1}
        model = MakeModel.make_test_model('1 species, 1 reaction',
                                        init_vols=[expected_initial_values_compt_1['init_volume']],
                                        init_vol_stds=[0],
                                        density=init_density,
                                        molecular_weight=molecular_weight,
                                        default_species_copy_number=default_species_copy_number,
                                        default_species_std=0,
                                        submodel_framework='WC:deterministic_simulation_algorithm')
        multialgorithm_simulation = MultialgorithmSimulation(model, self.args)
        _, dynamic_model = multialgorithm_simulation.build_simulation()
        self.check_simul_results(dynamic_model, None, expected_initial_values=expected_initial_values)

        # test dynamics
        simulation = Simulation(model)
        _, results_dir = simulation.run(end_time=2, **self.args)
        self.check_simul_results(dynamic_model, results_dir,
                                 expected_initial_values=expected_initial_values,
                                 expected_species_trajectories=\
                                     {'spec_type_0[compt_1]':[10000., 9999., 9998.]})
        with self.assertRaises(AssertionError):
            self.check_simul_results(dynamic_model, results_dir,
                                     expected_initial_values=expected_initial_values,
                                     expected_species_trajectories=\
                                         {'spec_type_0[compt_1]':[10000., 10000., 9998.]})
        with self.assertRaises(AssertionError):
            self.check_simul_results(dynamic_model, results_dir,
                                     expected_initial_values=expected_initial_values,
                                     expected_species_trajectories=\
                                         {'spec_type_0[compt_1]':[10000., 10000.]})
        self.check_simul_results(dynamic_model, results_dir,
                                 expected_initial_values=expected_initial_values,
                                 expected_species_trajectories=\
                                    {'spec_type_0[compt_1]':[10000., 9999., 9998.]},
                                    delta=1E-5)
        self.check_simul_results(dynamic_model, results_dir,
                                 expected_property_trajectories={'compt_1':
                                    {'mass':[1.000e-13, 1.000e-13, 9.999e-14]}})
        self.check_simul_results(dynamic_model, results_dir,
                                 expected_property_trajectories={'compt_1':
                                    {'mass':[1.000e-13, 1.000e-13, 9.999e-14]}},
                                    delta=1E-7)

    def specie_id_to_pop_attr(self, specie_id):
        # obtain an attribute name for the population of a species
        return f"pop_{specie_id.replace('[', '_').replace(']', '_')}"

    # todo: move this into wc_sim/testing/utils.py
    def define_trajectory_classes(self, model):
        # define the expected trajectory classes for model, SpeciesTrajectory & AggregateTrajectory
        # based on init_schema from obj_tables.utils
        # could put a list of species to ignore in Parameters, or, better yet, in a TestConfig worksheet

        # SpeciesTrajectory
        cls_specs = {}
        cls_name = 'SpeciesTrajectory'
        cls_specs[cls_name] = cls = {
            'name': cls_name,
            'attrs': {},
            'tab_format': obj_tables.TableFormat.row,
            'verbose_name': cls_name,
            'verbose_name_plural': 'SpeciesTrajectories',
            'desc': 'Trajectories of species populations',
        }
        # attributes: time, population for each species
        cls['attrs']['time'] = {
            'name': 'time',
            'type': obj_tables.FloatAttribute()
        }
        for specie in model.get_species():
            attr_name = self.specie_id_to_pop_attr(specie.id)
            cls['attrs'][attr_name] = {
                'name': attr_name,
                'type': obj_tables.FloatAttribute()
            }

        # AggregateTrajectory
        cls_name = 'AggregateTrajectory'
        cls_specs[cls_name] = cls = {
            'name': cls_name,
            'attrs': {},
            'tab_format': obj_tables.TableFormat.row,
            'verbose_name': cls_name,
            'verbose_name_plural': 'AggregateTrajectories',
            'desc': 'Trajectories of aggregate quantities',
        }
        # attributes: time, cell aggregates, aggregates for compartments
        cls['attrs']['time'] = {
            'name': 'time',
            'type': obj_tables.FloatAttribute()
        }
        for aggregate_value in DynamicModel.AGGREGATE_VALUES:
            attr_name = f'cell {aggregate_value}'
            attr_name.replace(' ', '_')
            cls['attrs'][attr_name] = {
                'name': attr_name,
                'type': obj_tables.FloatAttribute()
            }
        for compartment in model.get_compartments():
            # ignore compartments that aren't cellular
            if compartment.biological_type == onto['WC:cellular_compartment']:
                for aggregate_value in DynamicModel.AGGREGATE_VALUES:
                    attr_name = f'compartment {compartment.id} {aggregate_value}'
                    attr_name.replace(' ', '_')
                    cls['attrs'][attr_name] = {
                        'name': attr_name,
                        'type': obj_tables.FloatAttribute()
                    }

        # create the classes
        trajectory_classes = {}
        for cls_spec in cls_specs.values():
            meta_attrs = {
                'table_format': cls_spec['tab_format'],
                'description': cls_spec['desc'],
            }
            if cls_spec['verbose_name']:
                meta_attrs['verbose_name'] = cls_spec['verbose_name']
            if cls_spec['verbose_name_plural']:
                meta_attrs['verbose_name_plural'] = cls_spec['verbose_name_plural']

            attrs = {
                '__doc__': cls_spec['desc'],
                'Meta': type('Meta', (obj_tables.Model.Meta, ), meta_attrs),
            }
            for attr_spec in cls_spec['attrs'].values():
                attr = attr_spec['type']
                attrs[attr_spec['name']] = attr

            cls = type(cls_spec['name'], (obj_tables.Model, ), attrs)
            trajectory_classes[cls_spec['name']] = cls

        return(trajectory_classes)

    def verify_closed_form_model(self, model_filename):
        # alternatively, just load the SpeciesTrajectory & AggregateTrajectory worksheets into pandas
        # read model while ignoring missing models, with std dev = 0
        model = read_model_and_set_all_std_devs_to_0(model_filename)
        # simulate model
        # todo: move end_time & checkpoint_period to a separate simulation params worksheet
        end_time = model.parameters.get_one(id='end_time').value
        checkpoint_period = model.parameters.get_one(id='checkpoint_period').value
        args = dict(results_dir=self.results_dir,
                    checkpoint_period=checkpoint_period)
        simulation = Simulation(model)
        _, results_dir = simulation.run(end_time=end_time, **args)

        # test dynamics
        # read expected trajectories
        trajectories = self.define_trajectory_classes(model)
        SpeciesTrajectory = trajectories['SpeciesTrajectory']
        AggregateTrajectory = trajectories['AggregateTrajectory']
        trajectory_model_classes = list(trajectories.values())
        expected_trajectories = \
            obj_tables.io.Reader().run(model_filename, models=trajectory_model_classes,
                                       ignore_extra_models=True, ignore_attribute_order=True)

        # get species trajectories from model workbook
        expected_species_trajectories = {}
        species_ids = [specie.id for specie in model.get_species()]
        for specie_id in species_ids:
            expected_species_trajectories[specie_id] = []
        for specie_id in species_ids:
            for expected_species_trajectory in expected_trajectories[SpeciesTrajectory]:
                expected_pop = getattr(expected_species_trajectory, self.specie_id_to_pop_attr(specie_id))
                expected_species_trajectories[specie_id].append(expected_pop)
        expected_trajectory_times = []
        for expected_species_trajectory in expected_trajectories[SpeciesTrajectory]:
            expected_trajectory_times.append(expected_species_trajectory.time)

        # get aggregate trajectories from model workbook
        expected_aggregate_trajectories = {}
        for dyn_compartment_id in simulation.dynamic_model.dynamic_compartments:
            expected_aggregate_trajectories[dyn_compartment_id] = defaultdict(list)
        expected_trajectory_times = []
        for attr in AggregateTrajectory.Meta.local_attributes.values():
            attr_name = attr.name
            if attr_name != 'time':
                if attr_name.startswith('compartment'):
                    compartment_id = attr_name.split(' ')[1]
                    for expected_aggregate_trajectory in expected_trajectories[AggregateTrajectory]:
                        aggregate_prop = ' '.join(attr_name.split(' ')[2:])
                        expected_aggregate_trajectories[compartment_id][aggregate_prop].append(
                            getattr(expected_aggregate_trajectory, attr.name))
            else:
                for expected_aggregate_trajectory in expected_trajectories[AggregateTrajectory]:
                    expected_trajectory_times.append(expected_aggregate_trajectory.time)

        # compare expected & actual trajectories
        self.check_simul_results(simulation.dynamic_model, results_dir,
                                 expected_times=expected_trajectory_times,
                                 expected_species_trajectories=expected_species_trajectories,
                                 expected_property_trajectories=expected_aggregate_trajectories)
        # todo: plot expected & actual trajectories

    def test_closed_form_models(self):
        print()
        models_to_test = 'static one_reaction_linear one_rxn_exponential one_exchange_rxn_compt_growth'.split()
        for model_name in models_to_test:
            print(f'testing {model_name}')
            model_filename = os.path.join(os.path.dirname(__file__), 'fixtures', 'dynamic_tests', f'{model_name}.xlsx')
            self.verify_closed_form_model(model_filename)
        # todo: create and test other models
        models = 'static one_reaction_linear one_rxn_exponential one_exchange_rxn_compt_growth two_compts_exponential_2'.split()
        '''
        models = 'template static one_reaction_linear one_rxn_exponential'.split()
        for model_name in models:
            model_filename = os.path.join(os.path.dirname(__file__), 'fixtures', 'dynamic_tests', f'{model_name}.xlsx')
            model = Reader().run(model_filename, ignore_extra_models=True)[Model][0]
        '''

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
        multialgorithm_simulation = MultialgorithmSimulation(model, self.args)
        _, dynamic_model = multialgorithm_simulation.build_simulation()
        self.check_simul_results(dynamic_model, None, expected_initial_values=expected_initial_values)

        # test dynamics
        simulation = Simulation(model)
        _, results_dir = simulation.run(end_time=20, **self.args)

    def test_one_reaction_linear_species_pop_change(self):
        pass

    def test_two_submodels_linear_species_pop_changes(self):
        pass

    def test_two_submodels_exponential_species_pop_changes(self):
        pass


class TestRunSSASimulation(unittest.TestCase):

    def setUp(self):
        self.results_dir = tempfile.mkdtemp()
        self.args = dict(dfba_time_step=1,
                         results_dir=self.results_dir,
                         checkpoint_period=10)
        self.out_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.results_dir)
        shutil.rmtree(self.out_dir)

    def make_model_and_simulation(self, model_type, num_submodels, species_copy_numbers=None,
                                  species_stds=None, init_vols=None):
        # make simple model
        if init_vols is not None:
            if not isinstance(init_vols, list):
                init_vols = [init_vols]*num_submodels
        model = MakeModel.make_test_model(model_type, num_submodels=num_submodels,
                                          species_copy_numbers=species_copy_numbers,
                                          species_stds=species_stds,
                                          init_vols=init_vols)
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

    def perform_ssa_test_run(self, model_type, run_time, initial_species_copy_numbers, initial_species_stds,
                             expected_mean_copy_numbers, delta, num_submodels=1, invariants=None,
                             iterations=3, init_vols=None):
        """ Test SSA by comparing expected and actual simulation copy numbers

        Args:
            model_type (:obj:`str`): model type description
            run_time (:obj:`float`): duration of the simulation run
            initial_species_copy_numbers (:obj:`dict`): initial specie counts, with IDs as keys and counts as values
            expected_mean_copy_numbers (:obj:`dict`): expected final mean specie counts, in same format as
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
            model, multialgorithm_simulation, simulation_engine = self.make_model_and_simulation(
                model_type,
                num_submodels=num_submodels,
                species_copy_numbers=initial_species_copy_numbers,
                species_stds=initial_species_stds,
                init_vols=init_vols)
            local_species_pop = multialgorithm_simulation.local_species_population
            simulation_engine.initialize()
            simulation_engine.simulate(run_time)
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
        specie = 'spec_type_0[compt_1]'
        # tests checkpoint history in which the last checkpoint time < run time
        self.perform_ssa_test_run('1 species, 1 reaction',
                                  run_time=999,
                                  initial_species_copy_numbers={specie: 3000},
                                  initial_species_stds={specie: 0},
                                  expected_mean_copy_numbers={specie: 2000},
                                  delta=50)
        # species counts, and cell mass and volume steadily decline
        prev_ckpt = None
        for time in MultialgorithmCheckpoint.list_checkpoints(self.results_dir):
            ckpt = MultialgorithmCheckpoint.get_checkpoint(self.results_dir, time=time)
            if prev_ckpt:
                prev_species_pops, prev_observables, prev_functions, prev_aggregate_state = \
                    RunResults.get_state_components(prev_ckpt.state)
                species_pops, observables, functions, aggregate_state = RunResults.get_state_components(ckpt.state)
                self.assertTrue(species_pops[specie] < prev_species_pops[specie])
                self.assertTrue(aggregate_state['cell mass'] < prev_aggregate_state['cell mass'])
            prev_ckpt = ckpt

        self.perform_ssa_test_run('2 species, 1 reaction',
            run_time=1000,
            initial_species_copy_numbers={'spec_type_0[compt_1]': 3000, 'spec_type_1[compt_1]': 0},
            initial_species_stds={'spec_type_0[compt_1]': 0, 'spec_type_1[compt_1]': 0},
            expected_mean_copy_numbers={'spec_type_0[compt_1]': 2000,  'spec_type_1[compt_1]': 1000},
            delta=50)

    def test_runtime_errors(self):
        init_spec_type_0_pop = 2000
        # this model consumes all the reactants, driving propensities to 0:
        with self.assertRaisesRegex(MultialgorithmError,
                                    "simulation with 1 SSA submodel and total propensities = 0 cannot progress"):
            self.perform_ssa_test_run('2 species, 1 reaction, with rates given by reactant population',
                                      run_time=5000,
                                      initial_species_copy_numbers={
                                          'spec_type_0[compt_1]': init_spec_type_0_pop,
                                          'spec_type_1[compt_1]': 0},
                                      initial_species_stds={
                                          'spec_type_0[compt_1]': 0,
                                          'spec_type_1[compt_1]': 0},
                                      expected_mean_copy_numbers={},
                                      delta=0,
                                      init_vols=1E-22)

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
            locals = {'simulation_engine': simulation_engine,
                      'end_sim_time': end_sim_time}
            cProfile.runctx('num_events = simulation_engine.simulate(end_sim_time)', {}, locals, filename=out_file)
            profile = pstats.Stats(out_file)
            print("Profile for {} simulation objects:".format(num_ssa_submodels))
            profile.strip_dirs().sort_stats('cumulative').print_stats(15)

            num_ssa_submodels *= 4

        print('Performance summary')
        print("\n".join(unprofiled_perf))
        self.restore_logging()


    # TODO(Arthur): test multiple ssa submodels, in shared or different compartments
    # TODO(Arthur): test have identify_enabled_reactions() return a disabled reaction & ssa submodel with reactions that cannot run
    # TODO(Arthur): use invariants to test saving aggregate values from DynamicModel in checkpoints
    # TODO(Arthur): delete unused parts of CheckpointLogger
    # TODO(Arthur): catch MultialgorithmErrors from get_species_counts, and elsewhere
