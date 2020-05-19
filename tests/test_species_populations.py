""" Test species_populations.py

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2017-02-04
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

from numpy import all
from scipy.constants import Avogadro
from scipy.stats import binom
import copy
import numpy as np
import os
import re
import string
import sys
import unittest
import unittest

from de_sim.errors import SimulatorError
from de_sim.simulation_config import SimulationConfig
from de_sim.simulation_engine import SimulationEngine
from de_sim.simulation_message import SimulationMessage
from de_sim.simulation_object import SimulationObject
from de_sim.testing.mock_simulation_object import MockSimulationObject
from wc_lang.io import Reader
from wc_sim import distributed_properties
from wc_sim import message_types
from wc_sim.config import core as config_core_multialgorithm
from wc_sim.dynamic_components import DynamicModel
from wc_sim.multialgorithm_errors import (DynamicSpeciesPopulationError, DynamicNegativePopulationError,
                                          SpeciesPopulationError)
from wc_sim.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.sim_config import WCSimulationConfig
from wc_sim.species_populations import (LOCAL_POP_STORE, DynamicSpeciesState, SpeciesPopSimObject,
                                        SpeciesPopulationCache, LocalSpeciesPopulation, TempPopulationsLSP,
                                        MakeTestLSP, AccessSpeciesPopulations)
from wc_sim.testing.utils import read_model_for_test
from wc_utils.util.rand import RandomStateManager
import wc_lang
import wc_sim.species_populations

config_multialgorithm = config_core_multialgorithm.get_config()['wc_sim']['multialgorithm']


def store_i(i):
    return "store_{}".format(i)

def species_l(l):
    return "species_{}".format(l)

remote_pop_stores = {store_i(i): None for i in range(1, 4)}
species_ids = [species_l(l) for l in list(string.ascii_lowercase)[0:5]]


class TestAccessSpeciesPopulations(unittest.TestCase):

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures',
                                  'test_model_for_access_species_populations.xlsx')

    # MODEL_FILENAME_STEADY_STATE contains a model with no net population change every 2 sec.
    MODEL_FILENAME_STEADY_STATE = os.path.join(os.path.dirname(__file__), 'fixtures',
                                               'test_model_for_access_species_populations_steady_state.xlsx')

    def setUp(self):
        self.an_ASP = AccessSpeciesPopulations(None, remote_pop_stores)
        self.simulator = SimulationEngine()

    def test_add_species_locations(self):

        self.an_ASP.add_species_locations(store_i(1), species_ids[:2])
        map = dict.fromkeys(species_ids[:2], store_i(1))
        self.assertEqual(self.an_ASP.species_locations, map)

        self.an_ASP.add_species_locations(store_i(2), species_ids[2:])
        map.update(dict(zip(species_ids[2:], [store_i(2)]*3)))
        self.assertEqual(self.an_ASP.species_locations, map)

        locs = self.an_ASP.locate_species(species_ids[1:4])
        self.assertEqual(locs[store_i(1)], {'species_b'})
        self.assertEqual(locs[store_i(2)], {'species_c', 'species_d'})

        self.an_ASP.del_species_locations([species_l('b')])
        del map[species_l('b')]
        self.assertEqual(self.an_ASP.species_locations, map)
        self.an_ASP.del_species_locations(species_ids, force=True)
        self.assertEqual(self.an_ASP.species_locations, {})

    def test_add_species_locations(self):
        with self.assertRaisesRegex(SpeciesPopulationError, "'no_such_store' not a known population store"):
            self.an_ASP.add_species_locations('no_such_store', species_ids[:2])

        self.an_ASP.add_species_locations(store_i(1), species_ids[:2])
        with self.assertRaisesRegex(SpeciesPopulationError,
                                    re.escape("species ['species_a', 'species_b'] already have "
                                              "assigned locations.")):
            self.an_ASP.add_species_locations(store_i(1), species_ids[:2])

        with self.assertRaisesRegex(SpeciesPopulationError,
                                    re.escape("species ['species_c', 'species_d'] are not in the "
                                              "location map")):
            self.an_ASP.del_species_locations([species_l('d'), species_l('c')])

        with self.assertRaisesRegex(SpeciesPopulationError,
                                    re.escape("species ['species_c', 'species_d'] are not in the "
                                              "location map")):
            self.an_ASP.locate_species([species_l('d'), species_l('c')])

    def test_other_exceptions(self):
        with self.assertRaisesRegex(SpeciesPopulationError,
                                    f"{LOCAL_POP_STORE} not a valid remote_pop_store name"):
            AccessSpeciesPopulations(None, {'a': None, LOCAL_POP_STORE: None})

        with self.assertRaisesRegex(SpeciesPopulationError,
                                    "read_one: species'no_such_species' not in the location map."):
            self.an_ASP.read_one(0, 'no_such_species')

    @unittest.skip("use when SpeciesPopulationCache is being tested")
    def test_population_changes(self):
        """ Test population changes that occur without using event messages."""
        self.set_up_simulation(self.MODEL_FILENAME)
        theASP = self.submodels['dfba_submodel'].access_species_population
        init_val = 100
        self.assertEqual(theASP.read_one(0, 'species_1[c]'), init_val)
        self.assertEqual(theASP.read(0, set(['species_1[c]'])), {'species_1[c]': init_val})

        with self.assertRaisesRegex(SpeciesPopulationError,
                                    re.escape("read: species ['species_2[c]'] not in cache.")):
            theASP.read(0, set(['species_2[c]']))

        adjustment = -10
        self.assertEqual(theASP.adjust_discretely(0, {'species_1[c]': adjustment}),
                         ['LOCAL_POP_STORE'])
        self.assertEqual(theASP.read_one(0, 'species_1[c]'), init_val+adjustment)

        with self.assertRaisesRegex(SpeciesPopulationError,
                                    "read_one: species'species_none' not in the location map."):
            theASP.read_one(0, 'species_none')

        self.assertEqual(sorted(theASP.adjust_discretely(0,
                         {'species_1[c]': adjustment, 'species_2[c]': adjustment})),
                         sorted(['shared_store_1', 'LOCAL_POP_STORE']))
        self.assertEqual(theASP.read_one(0, 'species_1[c]'), init_val + 2*adjustment)

        self.assertEqual(theASP.adjust_continuously(1, {'species_1[c]': (adjustment, 0)}),
                         ['LOCAL_POP_STORE'])
        self.assertEqual(theASP.read_one(1, 'species_1[c]'), init_val + 3*adjustment)

        population_slope = 1
        time = 2
        delay = 3
        self.assertEqual(theASP.adjust_continuously(time, {'species_1[c]': (adjustment, population_slope)}),
                         ['LOCAL_POP_STORE'])
        self.assertEqual(theASP.read_one(time+delay, 'species_1[c]'),
                         init_val + 4*adjustment + delay*population_slope)

        with self.assertRaisesRegex(SpeciesPopulationError,
                                    "prefetch: 0 provided, but delay must be non-negative"):
            theASP.prefetch(0, ['species_1[c]', 'species_2[c]'])

        self.assertEqual(theASP.prefetch(1, ['species_1[c]', 'species_2[c]']), ['shared_store_1'])

    """
    todo: replace this code with calls to MultialgorithmSimulation().initialize()
    def initialize_simulation(self, model_file):
        self.set_up_simulation(model_file)
        delay_to_first_event = 1.0/len(self.submodels)
        for name,submodel in self.submodels.items():

            # prefetch into caches
            submodel.access_species_population.prefetch(delay_to_first_event,
                submodel.get_species_ids())

            # send initial event messages
            msg_body = message_types.ExecuteSsaReaction(0)
            submodel.send_event(delay_to_first_event, submodel, message_types.ExecuteSsaReaction,
                msg_body)

            delay_to_first_event += 1/len(self.submodels)
    """

    def verify_simulation(self, expected_final_pops, sim_end):
        """ Verify the final simulation populations."""
        for species_id in self.shared_species:
            pop = self.shared_pop_sim_obj['shared_store_1'].read_one(sim_end, species_id)
            self.assertEqual(expected_final_pops[species_id], pop)

        for submodel in self.submodels.values():
            for species_id in self.private_species[submodel.name]:
                pop = submodel.access_species_population.read_one(sim_end, species_id)
                self.assertEqual(expected_final_pops[species_id], pop)

    @unittest.skip("use when SpeciesPopSimObject is being tested")
    def test_simulation(self):
        """ Test a short simulation."""

        self.initialize_simulation(self.MODEL_FILENAME)

        # run the simulation
        sim_end = 3
        self.simulator.simulate(sim_end)

        # test final populations
        # Expected changes, based on the reactions executed
        expected_changes = """
        species c   e
        species_1   -2  0
        species_2   -2  0
        species_3   3   -2
        species_4   0   -1
        species_5   0   1"""

        expected_final_pops = copy.deepcopy(self.init_populations)
        for row in expected_changes.split('\n')[2:]:
            (species, c, e) = row.strip().split()
            for com in 'c e'.split():
                id = wc_lang.Species._gen_id(species, com)
                expected_final_pops[id] += float(eval(com))

        self.verify_simulation(expected_final_pops, sim_end)

    @unittest.skip("use when SpeciesPopSimObject is being tested")
    def test_stable_simulation(self):
        """ Test a steady state simulation.

        MODEL_FILENAME_STEADY_STATE contains a model with no net population change every 2 sec.
        """
        self.initialize_simulation(self.MODEL_FILENAME_STEADY_STATE)

        # run the simulation
        sim_end = 100
        self.simulator.simulate(sim_end)
        expected_final_pops = self.init_populations
        self.verify_simulation(expected_final_pops, sim_end)

# TODO(Arthur): test multiple SpeciesPopSimObjects
# TODO(Arthur): test adjust_continuously of remote_pop_stores


class TestLocalSpeciesPopulation(unittest.TestCase):

    def setUp(self):
        RandomStateManager.initialize(seed=123)

        self.species_nums = species_nums = range(1, 5)
        self.species_type_ids = species_type_ids = list(map(lambda x: "species_{}".format(x), species_nums))
        self.compartment_ids = compartment_ids = ['c1', 'c2']
        self.species_ids = species_ids = []
        for species_type_id in species_type_ids[:2]:
            for compartment_id in compartment_ids[:2]:
                species_ids.append(wc_lang.Species._gen_id(species_type_id, compartment_id))
        self.init_populations = dict(zip(species_ids, species_nums))
        self.init_pop_slope = 1
        self.init_pop_slopes = init_pop_slopes = dict(zip(species_ids, [self.init_pop_slope]*len(species_ids)))
        self.molecular_weights = dict(zip(species_ids, species_nums))

        self.local_species_pop = LocalSpeciesPopulation('test',
                                                        self.init_populations,
                                                        self.molecular_weights,
                                                        initial_population_slopes=init_pop_slopes,
                                                        random_state=RandomStateManager.instance())
        self.local_species_pop_no_history = \
            LocalSpeciesPopulation('test_no_history',
                                   self.init_populations,
                                   self.molecular_weights,
                                   initial_population_slopes=init_pop_slopes,
                                   random_state=RandomStateManager.instance(),
                                   retain_history=False)

        self.local_species_pop_no_init_pop_change = \
            LocalSpeciesPopulation('test',
                                   self.init_populations,
                                   self.molecular_weights,
                                   random_state=RandomStateManager.instance())

        molecular_weights_w_nans = copy.deepcopy(self.molecular_weights)
        species_w_nan_mw = 'species_w_nan_mw[c1]'
        molecular_weights_w_nans[species_w_nan_mw] = float('nan')
        init_populations = copy.deepcopy(self.init_populations)
        init_populations[species_w_nan_mw] = 0.
        self.local_species_pop_w_nan_mws = \
            LocalSpeciesPopulation('test',
                                   init_populations,
                                   molecular_weights_w_nans,
                                   initial_population_slopes=init_pop_slopes,
                                   random_state=RandomStateManager.instance())
        self.population_slope = 1
        self.local_species_pop.adjust_continuously(0, {id: self.population_slope for id in species_ids})

        self.local_species_pop_no_init_pop_slope = \
            LocalSpeciesPopulation('test',
                                   self.init_populations,
                                   self.molecular_weights)

        self.MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures',
                                           'test_model_for_access_species_populations.xlsx')

    def test_init(self):
        self.assertEqual(self.local_species_pop_no_init_pop_slope._all_species(), set(self.species_ids))
        an_LSP = LocalSpeciesPopulation('test', {}, {}, retain_history=False)
        s1_id = 's1[c]'
        mw = 1.5
        an_LSP.init_cell_state_species(s1_id, 2, mw)
        self.assertEqual(an_LSP.read(0, {s1_id}), {s1_id: 2})
        self.assertEqual(an_LSP._molecular_weights[s1_id], mw)

        # test initial_population_slope == 0
        an_LSP_2 = LocalSpeciesPopulation('test', {}, {}, random_state=RandomStateManager.instance(),
                                          retain_history=False)
        init_pop = 3
        an_LSP_2.init_cell_state_species(s1_id, init_pop, mw, initial_population_slope=0)
        time = 1
        slope = 2
        an_LSP_2.adjust_continuously(time, {s1_id: slope})
        time_delta = 1
        self.assertEqual(an_LSP_2.read(time + time_delta, {s1_id}), {s1_id: init_pop + time_delta * slope})

        with self.assertRaisesRegex(SpeciesPopulationError,
                                    re.escape(f"species_id '{s1_id}' already stored by this LocalSpeciesPopulation")):
            an_LSP.init_cell_state_species(s1_id, 2, mw)

        with self.assertRaisesRegex(SpeciesPopulationError, "history not recorded"):
            an_LSP.report_history()

        with self.assertRaisesRegex(SpeciesPopulationError, "history not recorded"):
            an_LSP.history_debug()

        with self.assertRaisesRegex(SpeciesPopulationError,
                                    "Cannot init LocalSpeciesPopulation .* species are missing weights"):
            LocalSpeciesPopulation('test',
                                   {s1_id: 2, 's2': 1}, {}, random_state=RandomStateManager.instance())

    def test_optional_species_argument(self):
        self.assertEqual(self.local_species_pop_no_init_pop_slope.read(0), self.init_populations)
        self.assertEqual(self.local_species_pop_no_init_pop_slope.read(2), self.init_populations)
        self.assertEqual(self.local_species_pop_no_init_pop_slope._check_species(0, species=None), None)
        t = 3
        self.local_species_pop_no_init_pop_slope._update_access_times(t, species=None)
        for species_id in self.local_species_pop_no_init_pop_slope._all_species():
            self.assertEqual(self.local_species_pop_no_init_pop_slope.last_access_time[species_id], t)

    def test_read_one(self):
        test_species = 'species_2[c2]'
        self.assertEqual(self.local_species_pop_no_init_pop_change.read_one(1, test_species),
                         self.init_populations[test_species])
        wc_sim.species_populations.RUN_TIME_ERROR_CHECKING = True
        with self.assertRaisesRegex(DynamicSpeciesPopulationError,
                                    "request for population of unknown species: 'unknown_species_id'"):
            self.local_species_pop_no_init_pop_change.read_one(2, 'unknown_species_id')
        with self.assertRaisesRegex(DynamicSpeciesPopulationError, r"is an earlier access of species"):
            self.local_species_pop_no_init_pop_change.read_one(0, test_species)
        wc_sim.species_populations.RUN_TIME_ERROR_CHECKING = False
        with self.assertRaises(KeyError):
            self.local_species_pop_no_init_pop_change.read_one(2, 'unknown_species_id')

    def reusable_assertions(self, the_local_species_pop, population_slope):
        # test both discrete and hybrid species

        with self.assertRaisesRegex(DynamicSpeciesPopulationError, "must be a set"):
            the_local_species_pop._check_species(0, 2)

        with self.assertRaisesRegex(DynamicSpeciesPopulationError,
                                    re.escape("request for population of unknown species:")):
            the_local_species_pop._check_species(0, {'x'})

        # populations have not changed
        self.assertEqual(the_local_species_pop.read(0, set(self.species_ids)), self.init_populations)
        first_species = self.species_ids[0]
        the_local_species_pop.adjust_discretely(0, {first_species: 3})
        self.assertEqual(the_local_species_pop.read(0, {first_species}), {first_species: 4})

        if population_slope:
            # counts: 1 initialization + 3 discrete adjustment + 2*population_slope:
            self.assertEqual(the_local_species_pop.read(2, {first_species}),
                             {first_species: 4+2*population_slope})
            the_local_species_pop.adjust_continuously(2, {first_species:0})
            # counts: 1 initialization + 3 discrete adjustment + 2 population_slope = 6:
            self.assertEqual(the_local_species_pop.read(2, {first_species}), {first_species: 6})
            for species_id in self.species_ids:
                self.assertIn(species_id, str(the_local_species_pop))

    def test_discrete_and_hybrid(self):
        wc_sim.species_populations.RUN_TIME_ERROR_CHECKING = True
        for (local_species_pop, population_slope) in [(self.local_species_pop, self.population_slope),
                                                      (self.local_species_pop_no_init_pop_slope, None)]:
            self.reusable_assertions(local_species_pop, population_slope)

    def test_get_continuous_species(self):
        self.assertEqual(self.local_species_pop.get_continuous_species(), set(self.species_ids))

        # test LSP with no continuous species
        local_species_pop = LocalSpeciesPopulation('test',
                                                   self.init_populations,
                                                   self.molecular_weights,
                                                   random_state=RandomStateManager.instance())
        self.assertEqual(local_species_pop.get_continuous_species(), set())

    def test_adjustment_exceptions(self):
        time = 1.0
        with self.assertRaisesRegex(DynamicSpeciesPopulationError, "adjust_discretely error"):
            self.local_species_pop.adjust_discretely(time, {id: -10 for id in self.species_ids})

        self.local_species_pop.adjust_continuously(time, {id: -10 for id in self.species_ids})
        with self.assertRaisesRegex(DynamicSpeciesPopulationError, "adjust_continuously error"):
            self.local_species_pop.adjust_continuously(time + 1, {id: 0 for id in self.species_ids})

        self.local_species_pop.adjust_continuously(time, {self.species_ids[0]: -10})
        with self.assertRaises(DynamicSpeciesPopulationError) as context:
            self.local_species_pop.adjust_continuously(time + 2, {self.species_ids[0]: 0})
        self.assertIn(f"{time + 2}: adjust_continuously error(s):", str(context.exception))
        self.assertIn("negative population predicted", str(context.exception))

    def test_temp_lsp_populations(self):
        s_0 = self.species_ids[0]
        time = 1
        pop_s_0 = self.local_species_pop.read_one(time, s_0)
        temp_pop_s_0 = 18
        self.local_species_pop.set_temp_populations({s_0: temp_pop_s_0})
        self.assertEqual(self.local_species_pop.read_one(time, s_0), temp_pop_s_0)
        self.local_species_pop.clear_temp_populations({s_0})
        self.assertEqual(self.local_species_pop.read_one(time, s_0), pop_s_0)

        wc_sim.species_populations.RUN_TIME_ERROR_CHECKING = True
        with self.assertRaises(DynamicSpeciesPopulationError):
            self.local_species_pop.set_temp_populations({'not a species id': temp_pop_s_0})

        with self.assertRaisesRegex(SpeciesPopulationError, 'cannot use negative population'):
            self.local_species_pop.set_temp_populations({s_0: -4})

        with self.assertRaises(DynamicSpeciesPopulationError):
            self.local_species_pop.clear_temp_populations(['not a species id'])

        wc_sim.species_populations.RUN_TIME_ERROR_CHECKING = False
        with self.assertRaises(KeyError):
            self.local_species_pop.set_temp_populations({'not a species id': temp_pop_s_0})

    def test_concentrations_api(self):
        self.assertFalse(self.local_species_pop.concentrations_api())
        self.local_species_pop.concentrations_api_on()
        self.assertTrue(self.local_species_pop.concentrations_api())
        self.local_species_pop.concentrations_api_off()
        self.assertFalse(self.local_species_pop.concentrations_api())

    def make_dynamic_model(self):

        # read model while ignoring missing models
        model = read_model_for_test(self.MODEL_FILENAME)

        # create dynamic model
        de_simulation_config = SimulationConfig(time_max=10)
        wc_sim_config = WCSimulationConfig(de_simulation_config)
        self.multialgorithm_simulation = MultialgorithmSimulation(model, wc_sim_config)
        self.multialgorithm_simulation.initialize_components()
        self.local_species_population = self.multialgorithm_simulation.local_species_population
        dynamic_model = DynamicModel(model, self.local_species_population,
                                          self.multialgorithm_simulation.temp_dynamic_compartments)

        self.compt_accounted_volumes = {}
        for dynamic_compartment in dynamic_model.dynamic_compartments.values():
            self.compt_accounted_volumes[dynamic_compartment.id] = dynamic_compartment.accounted_volume()

        return dynamic_model

    def test_accounted_volumes_for_species(self):
        dynamic_model = self.make_dynamic_model()
        species_ids = ['specie_3[c]', 'specie_2[e]', 'specie_3[e]']
        time = 2
        volumes = self.local_species_population._accounted_volumes_for_species(time, species_ids, dynamic_model)
        self.assertEqual(self.compt_accounted_volumes, volumes)

        # test one volume
        species_ids = ['specie_3[c]']
        volumes = self.local_species_population._accounted_volumes_for_species(time, species_ids, dynamic_model)
        self.assertEqual(['c'], list(volumes))
        self.assertEqual(self.compt_accounted_volumes['c'], volumes['c'])

        with self.assertRaises(DynamicSpeciesPopulationError):
            self.local_species_population._accounted_volumes_for_species(time, [], dynamic_model)

    def test_populations_to_concentrations(self):
        dynamic_model = self.make_dynamic_model()
        species_ids = ['specie_3[c]']
        time = 1
        volumes = self.local_species_population._accounted_volumes_for_species(time, species_ids,
                                                                               dynamic_model)

        populations = np.empty(len(species_ids))
        self.local_species_population.read_into_array(0, species_ids, populations)
        concentrations = np.empty(len(species_ids))
        self.local_species_population.populations_to_concentrations(time, species_ids, populations,
            dynamic_model, concentrations)
        expected_concentration = populations[0] / (Avogadro * volumes['c'])
        self.assertEqual(concentrations[0], expected_concentration)

    def test_populations_to_concentrations_and_concentrations_to_populations(self):
        dynamic_model = self.make_dynamic_model()
        species_ids = ['specie_3[c]', 'specie_2[e]', 'specie_3[e]']

        # test round-trip
        populations = np.empty(len(species_ids))
        self.local_species_population.read_into_array(0, species_ids, populations)
        concentrations = np.empty(len(species_ids))
        time = 1
        self.local_species_population.populations_to_concentrations(time, species_ids, populations,
            dynamic_model, concentrations)
        populations_round_trip = np.empty(len(species_ids))
        self.local_species_population.concentrations_to_populations(time, species_ids, concentrations,
            dynamic_model, populations_round_trip)
        self.assertTrue(np.allclose(populations, populations_round_trip))

        # test using volumes
        volumes = self.local_species_population._accounted_volumes_for_species(time, species_ids, dynamic_model)
        self.local_species_population.populations_to_concentrations(time, species_ids, populations,
            dynamic_model, concentrations, volumes=volumes)
        populations_round_trip = np.empty(len(species_ids))
        self.local_species_population.concentrations_to_populations(time, species_ids, concentrations,
            dynamic_model, populations_round_trip, volumes=volumes)
        self.assertTrue(np.allclose(populations, populations_round_trip))

    def test_read_into_array(self):
        populations = np.empty(len(self.species_ids))
        self.local_species_pop.read_into_array(0, self.species_ids, populations)
        self.assertTrue(np.array_equiv(populations, np.fromiter(self.species_nums, np.float64)))

    def test_history(self):
        an_LSP_wo_recording_history = LocalSpeciesPopulation('test',
                                                             self.init_populations,
                                                             self.init_populations,
                                                             random_state=RandomStateManager.instance(),
                                                             retain_history=False)

        with self.assertRaisesRegex(SpeciesPopulationError, 'history not recorded'):
            an_LSP_wo_recording_history.report_history()

        with self.assertRaisesRegex(SpeciesPopulationError, 'history not recorded'):
            an_LSP_wo_recording_history.history_debug()

        an_LSP_recording_history = LocalSpeciesPopulation('test',
                                                          self.init_populations,
                                                          self.init_populations,
                                                          random_state=RandomStateManager.instance(),
                                                          retain_history=True)
        self.assertTrue(an_LSP_recording_history._recording_history())
        next_time = 1
        first_species = self.species_ids[0]
        an_LSP_recording_history.read(next_time, {first_species})
        an_LSP_recording_history._record_history()
        with self.assertRaisesRegex(DynamicSpeciesPopulationError,
                                    re.escape(f"time of previous _record_history() ({next_time}) not "
                                              f"less than current time")):
            an_LSP_recording_history._record_history()

        history = an_LSP_recording_history.report_history()
        self.assertEqual(history['time'], [0, next_time])
        first_species_history = [1.0, 1.0]
        self.assertEqual(history['population'][first_species], first_species_history)
        self.assertIn(
            '\t'.join(map(lambda x: str(x), [first_species, 2] + first_species_history)),
            an_LSP_recording_history.history_debug())

        # test numpy array history
        with self.assertRaisesRegex(SpeciesPopulationError,
                                    'species_type_ids and compartment_ids must be provided'):
            an_LSP_recording_history.report_history(numpy_format=True)

        species_type_ids = self.species_type_ids
        compartment_ids = self.compartment_ids
        time_hist, species_counts_hist = \
            an_LSP_recording_history.report_history(numpy_format=True,
                                                    species_type_ids=species_type_ids,
                                                    compartment_ids=compartment_ids)
        self.assertTrue((time_hist == np.array([0, next_time])).all())
        for time_idx in [0, 1]:
            self.assertEqual(species_counts_hist[0, 0, time_idx], first_species_history[time_idx])

    def test_mass(self):
        self.assertEqual(self.local_species_pop.compartmental_mass('no_such_compartment'), 0)
        # 'species_1[c1]' is not in compartment 'c2'
        self.assertEqual(self.local_species_pop.compartmental_mass('c2', species_ids=['species_1[c1]']), 0)

        total_mass_c1 = 0
        for species_id in self.species_ids:
            if '[c1]' in species_id:
                total_mass_c1 += self.init_populations[species_id] * self.molecular_weights[species_id]
        total_mass_c1 = total_mass_c1 / Avogadro
        self.assertAlmostEqual(self.local_species_pop.compartmental_mass('c1'),
                               total_mass_c1, places=30)
        self.assertAlmostEqual(self.local_species_pop.compartmental_mass('c1', time=0),
                               total_mass_c1, places=30)
        self.assertAlmostEqual(self.local_species_pop_w_nan_mws.compartmental_mass('c1', time=0),
                               total_mass_c1, places=30)

        mass_of_species_1_in_c1 = \
            self.init_populations['species_1[c1]'] * self.molecular_weights['species_1[c1]'] / Avogadro
        self.assertAlmostEqual(self.local_species_pop.compartmental_mass('c1', species_ids=['species_1[c1]']),
                               mass_of_species_1_in_c1, places=30)

    def test_invalid_weights(self):
        bad_molecular_weights = ['x', float('nan'), -2, 0]
        good_molecular_weights = [3, 1E5, 1E-5, 2.34]
        num_mws = len(bad_molecular_weights)+len(good_molecular_weights)
        species_ids = [str(i)+'[c]' for i in range(num_mws)]
        molecular_weights = dict(zip(species_ids, bad_molecular_weights+good_molecular_weights))
        init_populations = dict(zip(species_ids, [1]*num_mws))
        local_species_pop = LocalSpeciesPopulation('test_invalid_weights',
                                                   init_populations,
                                                   molecular_weights)
        ids_w_bad_mws = species_ids[:len(bad_molecular_weights)]
        self.assertEqual(local_species_pop.invalid_weights(), set(ids_w_bad_mws))
        ids_w_bad_or_no_mw = ['x'] + ids_w_bad_mws
        self.assertEqual(local_species_pop.invalid_weights(species_ids=ids_w_bad_or_no_mw), set(ids_w_bad_or_no_mw))

    def test_make_test_lsp(self):
        make_test_lsp = MakeTestLSP()
        self.assertEqual(make_test_lsp.num_species, MakeTestLSP.DEFAULT_NUM_SPECIES)
        self.assertEqual(make_test_lsp.all_pops, MakeTestLSP.DEFAULT_ALL_POPS)
        self.assertEqual(make_test_lsp.all_mol_weights, MakeTestLSP.DEFAULT_ALL_MOL_WEIGHTS)
        kwargs = dict(
            num_species=7,
            all_pops=3E4,
            all_mol_weights=1000
        )
        make_test_lsp = MakeTestLSP(**kwargs)
        self.assertEqual(make_test_lsp.num_species, kwargs['num_species'])
        self.assertEqual(make_test_lsp.all_pops, kwargs['all_pops'])
        self.assertEqual(make_test_lsp.all_mol_weights, kwargs['all_mol_weights'])
        self.assertEqual(make_test_lsp.local_species_pop.read_one(0, 'species_1[comp_id]'), kwargs['all_pops'])
        name = 'foo'
        make_test_lsp_3 = MakeTestLSP(name=name, initial_population=make_test_lsp.initial_population)
        self.assertEqual(make_test_lsp_3.initial_population, make_test_lsp.initial_population)
        make_test_lsp_4 = MakeTestLSP(initial_population=make_test_lsp.initial_population,
                                      molecular_weights=make_test_lsp.molecular_weights)
        self.assertEqual(make_test_lsp_4.initial_population, make_test_lsp.initial_population)
        self.assertEqual(make_test_lsp_4.molecular_weights, make_test_lsp.molecular_weights)


class TestTempPopulationsLSP(unittest.TestCase):

    def setUp(self):
        self.pop = 10
        kwargs = dict(num_species=4, all_pops=self.pop, all_mol_weights=0)
        make_test_lsp = MakeTestLSP(**kwargs)
        self.test_lsp = make_test_lsp.local_species_pop
        self.species_ids = make_test_lsp.species_ids

    def test(self):
        num_species = 2
        species_ids = self.species_ids[:num_species]
        temp_pop = 123
        temp_pops = dict(zip(species_ids, [temp_pop]*num_species))
        with TempPopulationsLSP(self.test_lsp, temp_pops):
            for i in range(num_species):
                self.assertEqual(self.test_lsp.read_one(1, species_ids[i]), temp_pop)

        for i in range(num_species):
            self.assertEqual(self.test_lsp.read_one(1, species_ids[i]), self.pop)


class TestSpeciesPopulationCache(unittest.TestCase):

    def setUp(self):
        kwargs = dict(num_species=4, all_pops=0, all_mol_weights=0)
        make_test_lsp = MakeTestLSP(**kwargs)
        self.species_ids = make_test_lsp.species_ids
        local_species_population = make_test_lsp.local_species_pop

        remote_pop_stores = {store_i(i): None for i in range(1, 4)}
        self.an_ASP = AccessSpeciesPopulations(local_species_population, remote_pop_stores)
        self.an_ASP.add_species_locations(store_i(1), self.species_ids)
        self.an_ASP.add_species_locations(LOCAL_POP_STORE, ["species_0"])
        self.species_population_cache = self.an_ASP.species_population_cache

    def test_species_population_cache(self):
        populations = [x*10 for x in range(len(self.species_ids))]
        population_dict = dict(zip(self.species_ids, populations))
        self.species_population_cache.cache_population(1, population_dict)
        s = self.species_ids[0]
        self.assertEqual(self.species_population_cache.read_one(1, s), population_dict[s])
        self.assertEqual(self.species_population_cache.read(1, self.species_ids),
                         population_dict)

    def test_species_population_cache_exceptions(self):
        with self.assertRaisesRegex(SpeciesPopulationError,
                                    re.escape("some species are stored in the AccessSpeciesPopulations's "
                                              "local store: ['species_0'].")):
            self.species_population_cache.cache_population(1, {"species_0": 3})

        self.species_population_cache.cache_population(0, {"species_1[comp_id]": 3})
        with self.assertRaisesRegex(SpeciesPopulationError,
                                    re.escape("cache_population: caching an earlier population: species_id:"
                                              " species_1[comp_id]; current time: -1 <= previous time 0.")):
            self.species_population_cache.cache_population(-1, {"species_1[comp_id]": 3})

        with self.assertRaisesRegex(SpeciesPopulationError,
                                    "SpeciesPopulationCache.read_one: species'species_none' not in cache."):
            self.species_population_cache.read_one(1, 'species_none')

        with self.assertRaisesRegex(SpeciesPopulationError,
                                    re.escape("cache age of 1 too big for read at time 1 of species"
                                              "'species_1[comp_id]'")):
            self.species_population_cache.read_one(1, 'species_1[comp_id]')

        with self.assertRaisesRegex(SpeciesPopulationError,
                                    re.escape("SpeciesPopulationCache.read: species ['species_none'] "
                                              "not in cache.")):
            self.species_population_cache.read(0, ['species_none'])

        with self.assertRaisesRegex(SpeciesPopulationError,
                                    re.escape(".read: species ['species_1[comp_id]'] not reading recently "
                                              "cached value(s)")):
            self.species_population_cache.read(1, ['species_1[comp_id]'])


class TestDynamicSpeciesState(unittest.TestCase):

    def setUp(self):
        self.random_state = RandomStateManager.instance()

    def test_dynamic_species_state(self):

        # dynamic species modeled only by discrete submodel(s)
        pop = 10
        s1 = DynamicSpeciesState('s1[c]', self.random_state, pop)
        self.assertEqual(s1.compartment_id, 'c')
        self.assertEqual(s1.get_population(0), pop)
        time = 0
        adjustment = 1
        pop += adjustment
        self.assertEqual(s1.discrete_adjustment(time, adjustment), pop)
        time += 1
        self.assertEqual(s1.get_population(time), pop)
        adjustment = -1
        pop += adjustment
        self.assertEqual(s1.discrete_adjustment(time, adjustment), pop)
        time += 1
        self.assertEqual(s1.get_population(time), pop)

        # dynamic species modeled by both continuous and discrete
        pop = 2
        s2 = DynamicSpeciesState('s2[c]', self.random_state, pop, modeled_continuously=True)
        adjustment = 3
        pop += adjustment
        # no population slope yet
        time = 4
        self.assertEqual(s2.discrete_adjustment(time, adjustment), pop)
        # set population slope
        time += 2
        slope = 0.5
        self.assertEqual(s2.continuous_adjustment(time, slope), pop)
        # 1 sec later the pop has risen by the slope of 0.5; get_population() rounds by default
        time += 1
        self.assertIn(s2.get_population(time), {pop, pop+1})
        exact_pop = pop + 0.5
        # if round=False get_population return non-integer population
        self.assertEqual(s2.get_population(time, round=False), exact_pop)
        slope = 1.0
        time += 1
        exact_pop = pop + 1
        self.assertEqual(s2.continuous_adjustment(time, slope), exact_pop)

    def test_text_output(self):
        pop = 10
        s1 = DynamicSpeciesState('species_1[c]', self.random_state, pop)
        self.assertEqual("species_name: species_1[c]; last_population: {}".format(pop), str(s1))
        self.assertRegex(s1.row(), 'species_1\[c\]\t{}.*'.format(pop))

        s2 = DynamicSpeciesState('species_2[c]', self.random_state, pop, modeled_continuously=True)
        self.assertEqual("species_name: species_2[c]; last_population: {}; continuous_time: None; "
                         "population_slope: None".format(pop), str(s2))
        self.assertRegex(s2.row(), '^species_2\[c\]\t{}\..*$'.format(pop))
        time = 3
        slope = 2
        s2.continuous_adjustment(time, slope)
        self.assertRegex(s2.row(), '^species_2\[c\]\t{}\..*\t{}\..*\t{}\..*$'.format(pop, time, slope))

        self.assertRegex(DynamicSpeciesState.heading(), 'species_name\t.*')

    def test_species_with_interpolation_on_or_off(self):
        pop = 10
        s0 = DynamicSpeciesState('s0[c]', self.random_state, pop, modeled_continuously=True)
        time = 0
        slope = 1
        s0.continuous_adjustment(time, slope)
        time = 1
        interpolated_pop = pop + time * slope
        self.assertEqual(s0.get_population(time), interpolated_pop)
        self.assertEqual(s0.get_population(time, interpolate=True), interpolated_pop)
        non_interpolated_pop = pop
        self.assertEqual(s0.get_population(time, interpolate=False), non_interpolated_pop)

        # set the config interpolate variable to False
        from wc_sim.species_populations import config_multialgorithm
        existing_interpolate = config_multialgorithm['interpolate']
        config_multialgorithm['interpolate'] = False

        pop = 1
        s1 = DynamicSpeciesState('s1[c]', self.random_state, pop, modeled_continuously=True)
        time = 0
        slope = 1
        s1.continuous_adjustment(time, slope)
        time = 1
        interpolated_pop = pop + time * slope
        non_interpolated_pop = pop
        self.assertEqual(s1.get_population(time), non_interpolated_pop)
        self.assertEqual(s1.get_population(time, interpolate=False), non_interpolated_pop)
        self.assertEqual(s1.get_population(time, interpolate=True), interpolated_pop)

        # change the config interpolate variable back because all imports may already have been cached
        config_multialgorithm['interpolate'] = existing_interpolate

    def test_validation(self):
        ### dynamic species modeled only by a discrete submodel ###
        ds_discrete = DynamicSpeciesState('ds_discrete[c]', self.random_state, 0)
        self.assertTrue(ds_discrete.last_adjustment_time == ds_discrete.last_read_time == -float('inf'))
        time = 1
        ds_discrete.get_population(time)
        self.assertEqual(ds_discrete.last_read_time, time)
        time = 3
        ds_discrete.discrete_adjustment(time, 0)
        self.assertEqual(ds_discrete.last_adjustment_time, time)

        # test exceptions
        with self.assertRaisesRegex(DynamicSpeciesPopulationError,
                                    "get_population\(\): read_time is earlier than latest "
                                    "prior adjustment: "):
            ds_discrete.get_population(time-1)

        time = 5
        ds_discrete.get_population(time)
        with self.assertRaisesRegex(DynamicSpeciesPopulationError,
                                    "discrete_adjustment\(\): adjustment_time is earlier than "
                                    "latest prior read: "):
            ds_discrete.discrete_adjustment(time-1, 0)

        time = 6
        ds_discrete.discrete_adjustment(time, 0)
        with self.assertRaisesRegex(DynamicSpeciesPopulationError,
                                    "discrete_adjustment\(\): adjustment_time is earlier than "
                                    "latest prior adjustment: "):
            ds_discrete.discrete_adjustment(time-1, 0)

        ### dynamic species modeled by both continuous and discrete ###
        ds_hybrid = DynamicSpeciesState('ds_hybrid[c]', self.random_state, 0, modeled_continuously=True)
        time = 0
        ds_hybrid.continuous_adjustment(time, 0)
        self.assertEqual(ds_hybrid.last_adjustment_time, time)

        # test exceptions
        time = 2
        ds_hybrid.get_population(time)
        with self.assertRaisesRegex(DynamicSpeciesPopulationError,
                                    "continuous_adjustment\(\): adjustment_time is earlier than "
                                    "latest prior read: "):
            ds_hybrid.continuous_adjustment(time-1, 0)

        time = 4
        ds_hybrid.continuous_adjustment(time, 0)
        with self.assertRaisesRegex(DynamicSpeciesPopulationError,
                                    "continuous_adjustment\(\): adjustment_time is earlier than "
                                    "latest prior adjustment: "):
            ds_hybrid.continuous_adjustment(time-1, 0)

        # test failure to set modeled_continuously
        ds_discrete_2 = DynamicSpeciesState('ds_discrete_2[c]', self.random_state, 0)
        with self.assertRaisesRegex(DynamicSpeciesPopulationError,
                                    "DynamicSpeciesState for .* needs self.modeled_continuously==True"):
            ds_discrete_2.continuous_adjustment(time, 0)

    def test_raise_NegativePopulationError(self):
        # dynamic species modeled by just discrete submodels
        pop = 2
        s1 = DynamicSpeciesState('s1[c]', self.random_state, pop)

        time = 0
        adjustment = -3
        with self.assertRaises(DynamicNegativePopulationError) as context:
            s1.discrete_adjustment(time, adjustment)
        self.assertEqual(context.exception, DynamicNegativePopulationError(time, 'discrete_adjustment',
                                                                           's1[c]', pop, adjustment))

        # dynamic species modeled by continuous and discrete submodels
        pop = 3
        s2 = DynamicSpeciesState('s2[c]', self.random_state, pop, modeled_continuously=True)
        # set population slope
        time = 0
        slope = -1
        s2.continuous_adjustment(time, slope)
        time_advance = 4
        time += time_advance
        predicted_pop = pop + time * slope
        self.assertEqual(-1, predicted_pop)
        with self.assertRaises(DynamicNegativePopulationError) as context:
            new_slope = 0
            s2.continuous_adjustment(time, new_slope)
        self.assertEqual(context.exception, DynamicNegativePopulationError(time, 'continuous_adjustment', 's2[c]',
                                                                           pop, time_advance * slope,
                                                                           time_advance))

        with self.assertRaises(DynamicNegativePopulationError) as context:
            s2.get_population(time)
        self.assertEqual(context.exception, DynamicNegativePopulationError(time, 'get_population', 's2[c]',
                                                                           pop, time_advance * slope,
                                                                           time_advance))

        MINIMUM_ALLOWED_POPULATION = config_multialgorithm['minimum_allowed_population']
        pop = 3
        s3 = DynamicSpeciesState('s3[c]', self.random_state, pop, modeled_continuously=True)
        # set population slope
        time = 0
        slope = -1
        s3.continuous_adjustment(time, slope)
        # advance to time when population == MINIMUM_ALLOWED_POPULATION
        time_advance = (MINIMUM_ALLOWED_POPULATION - pop) / slope
        self.assertAlmostEqual(s3.get_population(time_advance, round=False), MINIMUM_ALLOWED_POPULATION)
        # advance to time when population < MINIMUM_ALLOWED_POPULATION
        with self.assertRaises(DynamicNegativePopulationError):
            s3.get_population(time_advance + 1e-3)

    def test_assertions(self):
        # raise AssertionErrors
        with self.assertRaisesRegex(AssertionError,
                                    r'DynamicSpeciesState .*: population should be >= 0'):
            DynamicSpeciesState('s1[c]', self.random_state, -10)

        with self.assertRaisesRegex(AssertionError,
                                    "DynamicSpeciesState '.*': discrete population .* a non-negative "
                                    "integer, but .* isn't"):
            DynamicSpeciesState('species[c]', self.random_state, 1.5)

        ds = DynamicSpeciesState('species[c]', self.random_state, 1)
        with self.assertRaisesRegex(AssertionError,
                                    r"DynamicSpeciesState '.*': population_change must be an integer, "
                                    r"but .* isn't"):
            ds.discrete_adjustment(2, .5)

        s1 = DynamicSpeciesState('s1[c]', self.random_state, 0, modeled_continuously=True)
        with self.assertRaisesRegex(AssertionError, r"population_slope .* must be an int or float"):
            s1.continuous_adjustment(1, 'not a number')

    def test_species_stochastic_rounding(self):
        s1 = DynamicSpeciesState('s1[c]', self.random_state, 10.5, modeled_continuously=True)
        samples = 1000
        for i in range(samples):
            pop = s1.get_population(0)
            self.assertTrue(pop in [10, 11])

        mean = np.mean([s1.get_population(0, round=True) for i in range(samples)])
        min = 10 + binom.ppf(0.01, n=samples, p=0.5) / samples
        max = 10 + binom.ppf(0.99, n=samples, p=0.5) / samples
        self.assertTrue(min <= mean <= max)

    def test_history(self):
        pop = 10
        ds = DynamicSpeciesState('s[c]', self.random_state, pop, modeled_continuously=True,
                                 record_history=True)
        slope = -2
        ds.continuous_adjustment(1, slope)
        discrete_adjustment = 3
        ds.discrete_adjustment(2, discrete_adjustment)
        HistoryRecord = DynamicSpeciesState.HistoryRecord
        Operation = DynamicSpeciesState.Operation
        expected_history = [HistoryRecord(0, Operation['initialize'], pop),
                            HistoryRecord(1, Operation['continuous_adjustment'], slope),
                            HistoryRecord(2, Operation['discrete_adjustment'], discrete_adjustment)]
        self.assertEqual(ds.get_history(), expected_history)

        ds = DynamicSpeciesState('s[c]', self.random_state, 0)
        with self.assertRaisesRegex(SpeciesPopulationError, 'history not recorded'):
            ds.get_history()

    def test_temp_populations(self):
        ds = DynamicSpeciesState('s[c]', None, 0)
        self.assertEqual(ds.get_temp_population_value(), None)
        population = 3
        ds.set_temp_population_value(population)
        self.assertEqual(ds.get_temp_population_value(), population)
        ds.clear_temp_population_value()
        self.assertEqual(ds.get_temp_population_value(), None)


""" Run a simulation with another simulation object to test SpeciesPopSimObject.

A SpeciesPopSimObject manages the population of one species, 'x'. A MockSimulationTestingObject sends
initialization events to SpeciesPopSimObject and compares the 'x's correct population with
its simulated population.
"""
class MockSimulationTestingObject(MockSimulationObject):

    def send_initial_events(self): pass

    def get_state(self):
        return 'object state to be provided'

    def send_debugging_events(self, species_pop_sim_obj, update_time, update_message, update_msg_body,
                              get_pop_time, get_pop_msg_body):
        self.send_event(update_time, species_pop_sim_obj, update_msg_body)
        self.send_event(get_pop_time, species_pop_sim_obj, get_pop_msg_body)

    def handle_GivePopulation_event(self, event):
        """ Perform a unit test on the population of self.species_id."""

        # event.message is a GivePopulation instance
        the_population = event.message.population
        species_id = self.kwargs['species_id']
        expected_value = self.kwargs['expected_value']
        self.test_case.assertEqual(the_population[species_id], expected_value,
                                   msg="At event_time {} for species'{}': the correct population "
                                   "is {} but the actual population is {}.".format(
            event.event_time, species_id, expected_value, the_population[species_id]))

    def handle_GiveProperty_event(self, event):
        """ Perform a unit test on the mass of a SpeciesPopSimObject"""
        property_name = event.message.property_name
        self.test_case.assertEqual(property_name, distributed_properties.MASS)
        self.test_case.assertEqual(event.message.value, self.kwargs['expected_value'])

    # register the event handler for each type of message received
    event_handlers = [(message_types.GivePopulation, handle_GivePopulation_event),
                      (message_types.GiveProperty, handle_GiveProperty_event)]

    # register the message types sent
    messages_sent = [message_types.GetPopulation,
                     message_types.AdjustPopulationByDiscreteSubmodel,
                     message_types.AdjustPopulationByContinuousSubmodel,
                     message_types.GetCurrentProperty]


class InitMsg1(SimulationMessage):
    "Blank docstring"


class TestSpeciesPopSimObject(unittest.TestCase):

    def setUp(self):
        self.simulator = SimulationEngine()
        self.species_ids = 's1[c] s2[c] s3[c]'.split()
        self.initial_population = dict(zip(self.species_ids, range(3)))
        self.molecular_weight = dict(zip(self.species_ids, [10.]*3))
        self.test_species_pop_sim_obj = SpeciesPopSimObject('test_name', self.initial_population,
                                                            self.molecular_weight,
                                                            random_state=RandomStateManager.instance())
        self.simulator.add_object(self.test_species_pop_sim_obj)

    def test_init(self):
        for s in self.initial_population.keys():
            self.assertEqual(self.test_species_pop_sim_obj.read_one(0, s), self.initial_population[s])

    def test_invalid_event_types(self):

        with self.assertRaisesRegex(SimulatorError,
                                    re.escape("'wc_sim.species_populations.SpeciesPopSimObject' "
                                              "simulation objects not registered to send")):
            self.test_species_pop_sim_obj.send_event(1.0, self.test_species_pop_sim_obj, InitMsg1())

        with self.assertRaisesRegex(SimulatorError,
                                    re.escape("'wc_sim.species_populations.SpeciesPopSimObject' "
                                              "simulation objects not registered to receive")):
            self.test_species_pop_sim_obj.send_event(1.0, self.test_species_pop_sim_obj,
                                                     message_types.GivePopulation(7))
