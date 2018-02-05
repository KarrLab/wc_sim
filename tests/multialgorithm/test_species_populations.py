""" Test species_populations.py.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-02-04
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

import numpy as np
import os, unittest, copy
import re
import six
import string
import sys
import unittest
from builtins import super

from scipy.constants import Avogadro
from scipy.stats import binom

from wc_lang.io import Reader
import wc_lang
from wc_sim.core.errors import SimulatorError
from wc_sim.core.simulation_engine import SimulationEngine
from wc_sim.core.simulation_object import EventQueue, SimulationObject, SimulationObjectInterface
from wc_sim.multialgorithm import message_types
from wc_sim.multialgorithm.species_populations import AccessSpeciesPopulations
from wc_sim.multialgorithm.species_populations import (LOCAL_POP_STORE, Specie, SpeciesPopSimObject,
    SpeciesPopulationCache, LocalSpeciesPopulation)
from wc_sim.multialgorithm.multialgorithm_errors import NegativePopulationError, SpeciesPopulationError
from wc_sim.multialgorithm.submodels.skeleton_submodel import SkeletonSubmodel
from wc_sim.multialgorithm import distributed_properties
from wc_utils.util.rand import RandomStateManager

def store_i(i):
    return "store_{}".format(i)

def specie_l(l):
    return "specie_{}".format(l)

remote_pop_stores = {store_i(i):None for i in range(1, 4)}
species_ids = [specie_l(l) for l in list(string.ascii_lowercase)[0:5]]

class TestAccessSpeciesPopulations(unittest.TestCase):

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures',
        'test_model_for_access_species_populations.xlsx')

    # MODEL_FILENAME_STEADY_STATE contains a model with no net population change every 2 sec.
    MODEL_FILENAME_STEADY_STATE = os.path.join(os.path.dirname(__file__), 'fixtures',
        'test_model_for_access_species_populations_steady_state.xlsx')

    def setUp(self):
        self.an_ASP = AccessSpeciesPopulations(None, remote_pop_stores)
        self.simulator = SimulationEngine()
        self.simulator.register_object_types([SpeciesPopSimObject, SkeletonSubmodel])

    """
    todo: replace this code with calls to MultialgorithmSimulation().initialize()
    def set_up_simulation(self, model_file):
        '''Set up a simulation from a test model.

        Create two SkeletonSubmodels, a LocalSpeciesPopulation for each, and
        a SpeciesPopSimObject that they share.
        '''

        # make a model
        self.model = Reader().run(model_file)

        # make SpeciesPopSimObjects
        self.private_species = ModelUtilities.find_private_species(self.model, return_ids=True)
        self.shared_species = ModelUtilities.find_shared_species(self.model, return_ids=True)

        self.init_populations={}
        for species in self.model.get_species():
            sc_id = species.serialize()
            self.init_populations[sc_id] = \
                int(species.concentration.value * species.compartment.initial_volume * Avogadro)

        self.shared_pop_sim_obj = {}
        SHARED_STORE_ID = 'shared_store_1'
        self.shared_pop_sim_obj[SHARED_STORE_ID] = SpeciesPopSimObject(SHARED_STORE_ID,
            {specie_id:self.init_populations[specie_id] for specie_id in self.shared_species})
        self.simulator.add_object(self.shared_pop_sim_obj[SHARED_STORE_ID])

        # make submodels and their parts
        self.submodels={}
        for submodel in self.model.get_submodels():

            # make LocalSpeciesPopulations
            local_species_population = LocalSpeciesPopulation(
                submodel.id.replace('_', '_lsp_'),
                {specie_id:self.init_populations[specie_id] for specie_id in
                    self.private_species[submodel.id]},
                initial_fluxes={specie_id:0 for specie_id in self.private_species[submodel.id]})

            # make AccessSpeciesPopulations objects
            # TODO(Arthur): stop giving all SpeciesPopSimObjects to each AccessSpeciesPopulations
            access_species_population = AccessSpeciesPopulations(local_species_population, self.shared_pop_sim_obj)

            # make SkeletonSubmodels
            behavior = {'INTER_REACTION_TIME':1}
            self.submodels[submodel.id] = SkeletonSubmodel(behavior, self.model, submodel.id,
                access_species_population, submodel.reactions, submodel.get_species(), submodel.parameters)
            self.simulator.add_object(self.submodels[submodel.id])
            # connect AccessSpeciesPopulations object to its affiliated SkeletonSubmodels
            access_species_population.set_submodel(self.submodels[submodel.id])

            # make access_species_population.species_locations
            access_species_population.add_species_locations(LOCAL_POP_STORE,
                self.private_species[submodel.id])
            access_species_population.add_species_locations('shared_store_1', self.shared_species)
    """

    def test_add_species_locations(self):

        self.an_ASP.add_species_locations(store_i(1), species_ids[:2])
        map = dict.fromkeys(species_ids[:2], store_i(1))
        self.assertEqual(self.an_ASP.species_locations, map)

        self.an_ASP.add_species_locations(store_i(2), species_ids[2:])
        map.update(dict(zip(species_ids[2:], [store_i(2)]*3)))
        self.assertEqual(self.an_ASP.species_locations, map)

        locs = self.an_ASP.locate_species(species_ids[1:4])
        self.assertEqual(locs[store_i(1)], {'specie_b'})
        self.assertEqual(locs[store_i(2)], {'specie_c', 'specie_d'})

        self.an_ASP.del_species_locations([specie_l('b')])
        del map[specie_l('b')]
        self.assertEqual(self.an_ASP.species_locations, map)
        self.an_ASP.del_species_locations(species_ids, force=True)
        self.assertEqual(self.an_ASP.species_locations, {})

    def test_add_species_locations(self):
        with self.assertRaises(SpeciesPopulationError) as cm:
            self.an_ASP.add_species_locations('no_such_store', species_ids[:2])
        self.assertIn("'no_such_store' not a known population store", str(cm.exception))

        self.an_ASP.add_species_locations(store_i(1), species_ids[:2])
        with self.assertRaises(SpeciesPopulationError) as cm:
            self.an_ASP.add_species_locations(store_i(1), species_ids[:2])
        self.assertIn("species ['specie_a', 'specie_b'] already have assigned locations.",
            str(cm.exception))

        with self.assertRaises(SpeciesPopulationError) as cm:
            self.an_ASP.del_species_locations([specie_l('d'), specie_l('c')])
        self.assertIn("species ['specie_c', 'specie_d'] are not in the location map",
            str(cm.exception))

        with self.assertRaises(SpeciesPopulationError) as cm:
            self.an_ASP.locate_species([specie_l('d'), specie_l('c')])
        self.assertIn("species ['specie_c', 'specie_d'] are not in the location map",
            str(cm.exception))

    def test_other_exceptions(self):
        with self.assertRaises(SpeciesPopulationError) as cm:
            AccessSpeciesPopulations(None, {'a':None, LOCAL_POP_STORE:None})
        self.assertIn("{} not a valid remote_pop_store name".format(LOCAL_POP_STORE),
            str(cm.exception))
        with self.assertRaises(SpeciesPopulationError) as cm:
            self.an_ASP.read_one(0, 'no_such_specie')
        self.assertEqual(str(cm.exception), "read_one: specie 'no_such_specie' not in the location map.")


    @unittest.skip("skip until MultialgorithmSimulation().initialize() is ready")
    def test_population_changes(self):
        """ Test population changes that occur without using event messages."""
        self.set_up_simulation(self.MODEL_FILENAME)
        theASP = self.submodels['dfba_submodel'].access_species_population
        init_val=100
        self.assertEqual(theASP.read_one(0, 'specie_1[c]'), init_val)
        self.assertEqual(theASP.read(0, set(['specie_1[c]'])), {'specie_1[c]': init_val})

        with self.assertRaises(SpeciesPopulationError) as cm:
            theASP.read(0, set(['specie_2[c]']))
        self.assertIn("read: species ['specie_2[c]'] not in cache.", str(cm.exception))

        adjustment=-10
        self.assertEqual(theASP.adjust_discretely(0, {'specie_1[c]':adjustment}),
            ['LOCAL_POP_STORE'])
        self.assertEqual(theASP.read_one(0, 'specie_1[c]'), init_val+adjustment)

        with self.assertRaises(SpeciesPopulationError) as cm:
            theASP.read_one(0, 'specie_none')
        self.assertIn("read_one: specie 'specie_none' not in the location map.", str(cm.exception))

        self.assertEqual(sorted(theASP.adjust_discretely(0,
            {'specie_1[c]': adjustment, 'specie_2[c]': adjustment})),
                sorted(['shared_store_1', 'LOCAL_POP_STORE']))
        self.assertEqual(theASP.read_one(0, 'specie_1[c]'), init_val + 2*adjustment)

        self.assertEqual(theASP.adjust_continuously(1, {'specie_1[c]':(adjustment, 0)}),
            ['LOCAL_POP_STORE'])
        self.assertEqual(theASP.read_one(1, 'specie_1[c]'), init_val + 3*adjustment)

        flux=1
        time=2
        delay=3
        self.assertEqual(theASP.adjust_continuously(time, {'specie_1[c]':(adjustment, flux)}),
            ['LOCAL_POP_STORE'])
        self.assertEqual(theASP.read_one(time+delay, 'specie_1[c]'),
            init_val + 4*adjustment + delay*flux)

        with self.assertRaises(SpeciesPopulationError) as cm:
            theASP.prefetch(0, ['specie_1[c]', 'specie_2[c]'])
        self.assertIn("prefetch: 0 provided, but delay must be non-negative", str(cm.exception))

        self.assertEqual(theASP.prefetch(1, ['specie_1[c]', 'specie_2[c]']), ['shared_store_1'])

    """
    todo: replace this code with calls to MultialgorithmSimulation().initialize()
    def initialize_simulation(self, model_file):
        self.set_up_simulation(model_file)
        delay_to_first_event = 1.0/len(self.submodels)
        for name,submodel in six.iteritems(self.submodels):

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
        for specie_id in self.shared_species:
            pop = self.shared_pop_sim_obj['shared_store_1'].read_one(sim_end, specie_id)
            self.assertEqual(expected_final_pops[specie_id], pop)

        for submodel in self.submodels.values():
            for specie_id in self.private_species[submodel.name]:
                pop = submodel.access_species_population.read_one(sim_end, specie_id)
                self.assertEqual(expected_final_pops[specie_id], pop)

    @unittest.skip("skip until MultialgorithmSimulation().initialize() is ready")
    def test_simulation(self):
        """ Test a short simulation."""

        self.initialize_simulation(self.MODEL_FILENAME)

        # run the simulation
        sim_end=3
        self.simulator.simulate(sim_end)

        # test final populations
        # Expected changes, based on the reactions executed
        expected_changes="""
        specie	c	e
        specie_1	-2	0
        specie_2	-2	0
        specie_3	3	-2
        specie_4	0	-1
        specie_5	0	1"""

        expected_final_pops = copy.deepcopy(self.init_populations)
        for row in expected_changes.split('\n')[2:]:
            (specie, c, e) = row.strip().split()
            for com in 'c e'.split():
                id = wc_lang.core.Species.gen_id(specie, com)
                expected_final_pops[id] += float(eval(com))

        self.verify_simulation(expected_final_pops, sim_end)

    @unittest.skip("skip until MODEL_FILENAME_STEADY_STATE is migrated")
    def test_stable_simulation(self):
        """ Test a steady state simulation.

        MODEL_FILENAME_STEADY_STATE contains a model with no net population change every 2 sec.
        """
        self.initialize_simulation(self.MODEL_FILENAME_STEADY_STATE)

        # run the simulation
        sim_end=100
        self.simulator.simulate(sim_end)
        expected_final_pops = self.init_populations
        self.verify_simulation(expected_final_pops, sim_end)

# TODO(Arthur): test multiple SpeciesPopSimObjects
# TODO(Arthur): test adjust_continuously of remote_pop_stores
# TODO(Arthur): evaluate coverage


class TestLocalSpeciesPopulation(unittest.TestCase):

    def setUp(self):
        RandomStateManager.initialize(seed=123)

        species_nums = range(1, 5)
        species = list(map(lambda x: "specie_{}".format(x), species_nums))
        self.species = species
        self.init_populations = dict(zip(species, species_nums))
        self.flux = 1
        init_fluxes = dict(zip(species, [self.flux]*len(species)))
        self.init_fluxes = init_fluxes
        self.molecular_weights = dict(zip(species, species_nums))
        self.local_species_pop = LocalSpeciesPopulation('test', self.init_populations,
            self.molecular_weights, initial_fluxes=init_fluxes)
        self.local_species_pop_no_init_flux = LocalSpeciesPopulation(
            'test', self.init_populations, self.molecular_weights)

    def reusable_assertions(self, the_local_species_pop, flux):
        # test both discrete and hybrid species

        with self.assertRaises(SpeciesPopulationError) as context:
            the_local_species_pop._check_species(0, 2)
        self.assertIn("must be a set", str(context.exception))

        with self.assertRaises(SpeciesPopulationError) as context:
            the_local_species_pop._check_species(0, {'x'})
        self.assertIn("Error: request for population of unknown specie(s):", str(context.exception))

        self.assertEqual(the_local_species_pop.read(0, set(self.species)), self.init_populations)
        first_specie = self.species[0]
        the_local_species_pop.adjust_discretely(0, { first_specie: 3 })
        self.assertEqual(the_local_species_pop.read(0, {first_specie}),  {first_specie: 4})

        if flux:
            # counts: 1 initialization + 3 discrete adjustment + 2*flux:
            self.assertEqual(the_local_species_pop.read(2, {first_specie}),  {first_specie: 4+2*flux})
            the_local_species_pop.adjust_continuously(2, {first_specie:(9, 0) })
            # counts: 1 initialization + 3 discrete adjustment + 9 continuous adjustment + 0 flux = 13:
            self.assertEqual(the_local_species_pop.read(2, {first_specie}),  {first_specie: 13})

    def test_read_one(self):
        self.assertEqual(self.local_species_pop.read_one(1,'specie_3'), 4)
        with self.assertRaises(SpeciesPopulationError) as context:
            self.local_species_pop.read_one(2, 's1')
        self.assertIn("request for population of unknown specie(s): 's1'", str(context.exception))
        with self.assertRaises(SpeciesPopulationError) as context:
            self.local_species_pop.read_one(0, 'specie_3')
        self.assertIn("earlier access of specie(s): ['specie_3']", str(context.exception))

    def test_discrete_and_hyrid(self):

        for (local_species_pop, flux) in [(self.local_species_pop,self.flux),
            (self.local_species_pop_no_init_flux,None)]:
            self.reusable_assertions(local_species_pop, flux)

    def test_init(self):
        an_LSP = LocalSpeciesPopulation('test', {}, {}, retain_history=False)
        an_LSP.init_cell_state_specie('s1', 2)
        self.assertEqual(an_LSP.read(0, {'s1'}),  {'s1': 2})

        with self.assertRaises(SpeciesPopulationError) as context:
            an_LSP.init_cell_state_specie('s1', 2)
        self.assertIn("Error: specie_id 's1' already stored by this LocalSpeciesPopulation",
            str(context.exception))
        with self.assertRaises(SpeciesPopulationError) as context:
            an_LSP.report_history()
        self.assertIn("Error: history not recorded", str(context.exception))
        with self.assertRaises(SpeciesPopulationError) as context:
            an_LSP.history_debug()
        self.assertIn("Error: history not recorded", str(context.exception))

    def test_history(self):
        """ Test population history."""
        an_LSP_recording_history = LocalSpeciesPopulation('test',
            self.init_populations, self.init_populations, retain_history=False)
        with self.assertRaises(SpeciesPopulationError) as context:
            an_LSP_recording_history.report_history()
        self.assertIn("Error: history not recorded", str(context.exception))
        with self.assertRaises(SpeciesPopulationError) as context:
            an_LSP_recording_history.history_debug()
        self.assertIn("Error: history not recorded", str(context.exception))

        an_LSP_recording_history = LocalSpeciesPopulation('test',
            self.init_populations, self.init_populations, retain_history=True)
        self.assertTrue(an_LSP_recording_history._recording_history())
        next_time = 1
        first_specie = self.species[0]
        an_LSP_recording_history.read(next_time, {first_specie})
        an_LSP_recording_history._record_history()
        with self.assertRaises(SpeciesPopulationError) as context:
            an_LSP_recording_history._record_history()
        self.assertIn("time of previous _record_history() (1) not less than current time",
            str(context.exception))

        history = an_LSP_recording_history.report_history()
        self.assertEqual(history['time'],  [0,next_time])
        first_specie_history = [1.0,1.0]
        self.assertEqual(history['population'][first_specie], first_specie_history)

        self.assertIn(
            '\t'.join(map(lambda x:str(x), [ first_specie, 2, ] + first_specie_history)),
            an_LSP_recording_history.history_debug())

    def test_mass(self):
        """ Test mass """
        total_mass = sum([self.init_populations[specie_id]*self.molecular_weights[specie_id]/Avogadro
            for specie_id in self.species])
        self.assertAlmostEqual(self.local_species_pop.mass(), total_mass, places=37)

        all_but_1st_species = self.species[1:]
        mass_of_all_but_1st_species = sum([self.init_populations[specie_id]*self.molecular_weights[specie_id]/Avogadro
            for specie_id in all_but_1st_species])
        self.assertAlmostEqual(self.local_species_pop.mass(species_ids=all_but_1st_species),
            mass_of_all_but_1st_species, places=37)

        removed_specie = self.species[0]
        del self.local_species_pop._molecular_weights[removed_specie]
        with self.assertRaises(SpeciesPopulationError) as context:
            self.local_species_pop.mass()
        self.assertIn("molecular weight not available for '{}'".format(removed_specie),
            str(context.exception))

    """
    todo: test the distributed property MASS
    def test_mass(self):
        self.mass = sum([self.initial_population[specie_id]*self.molecular_weight[specie_id]/Avogadro
            for specie_id in self.species_ids])
        mock_obj = MockSimulationObject('mock_name', self, None, self.mass)
        self.simulator.add_object(mock_obj)
        mock_obj.send_event(1, self.test_species_pop_sim_obj, message_types.GetCurrentProperty,
            message_types.GetCurrentProperty(distributed_properties.MASS))
        self.simulator.initialize()
        self.simulator.simulate(2)
    """

class TestSpeciesPopulationCache(unittest.TestCase):

    def setUp(self):
        species_nums = range(1, 5)
        self.species_ids = list(map(lambda x: "specie_{}".format(x), species_nums))
        self.init_populations = dict(zip(self.species_ids, [0]*len(self.species_ids)))
        self.molecular_weights = self.init_populations
        local_species_population = LocalSpeciesPopulation('name', self.init_populations,
            self.molecular_weights)

        remote_pop_stores = {store_i(i):None for i in range(1, 4)}
        self.an_ASP = AccessSpeciesPopulations(local_species_population, remote_pop_stores)
        self.an_ASP.add_species_locations(store_i(1), self.species_ids)
        self.an_ASP.add_species_locations(LOCAL_POP_STORE, ["specie_0"])
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
        with self.assertRaises(SpeciesPopulationError) as context:
            self.species_population_cache.cache_population(1, {"specie_0": 3})
        self.assertIn("some species are stored in the AccessSpeciesPopulations's local store: "
            "['specie_0'].", str(context.exception))

        self.species_population_cache.cache_population(0, {"specie_1": 3})
        with self.assertRaises(SpeciesPopulationError) as context:
            self.species_population_cache.cache_population(-1, {"specie_1": 3})
        self.assertIn("cache_population: caching an earlier population: specie_id: specie_1; "
            "current time: -1 <= previous time 0.", str(context.exception))

        with self.assertRaises(SpeciesPopulationError) as context:
            self.species_population_cache.read_one(1, 'specie_none')
        self.assertIn("SpeciesPopulationCache.read_one: specie 'specie_none' not in cache.",
            str(context.exception))

        with self.assertRaises(SpeciesPopulationError) as context:
            self.species_population_cache.read_one(1, 'specie_1')
        self.assertIn("cache age of 1 too big for read at time 1 of specie 'specie_1'",
            str(context.exception))

        with self.assertRaises(SpeciesPopulationError) as context:
            self.species_population_cache.read(0, ['specie_none'])
        self.assertIn("SpeciesPopulationCache.read: species ['specie_none'] not in cache.",
            str(context.exception))

        with self.assertRaises(SpeciesPopulationError) as context:
            self.species_population_cache.read(1, ['specie_1'])
        self.assertIn(".read: species ['specie_1'] not reading recently cached value(s)",
            str(context.exception))


class TestSpecie(unittest.TestCase):

    def setUp(self):
        RandomStateManager.initialize()

    def test_Specie(self):
        s1 = Specie('specie', 10)

        self.assertEqual(s1.get_population(), 10)
        self.assertEqual(s1.discrete_adjustment(1, 0), 11)
        self.assertEqual(s1.get_population(), 11)
        self.assertEqual(s1.discrete_adjustment(-1, 0), 10)
        self.assertEqual(s1.get_population(), 10)

        s1 = Specie('specie_3', 2, 1)
        self.assertEqual(s1.discrete_adjustment(3, 4), 9)

        s1 = Specie('specie', 10, initial_flux=0)
        self.assertEqual("specie_name: specie; last_population: 10; continuous_time: 0; "
            "continuous_flux: 0", str(s1))

        if six.PY3:
            six.assertRegex(self, s1.row(), 'specie\t10\..*\t0\..*\t0\..*')
            s2 = Specie('specie2', 10, initial_flux=2.1)
            six.assertRegex(self, s2.row(), 'specie2\t10\..*\t0\..*\t2\.1.*')
        else:
            six.assertRegex(self, s1.row(), 'specie\t10\..*\t0\..*\t0\..*')
            s2 = Specie('specie2', 10, initial_flux=2.1)
            six.assertRegex(self, s2.row(), 'specie2\t10\..*\t0\..*\t2\.1.*')

        with self.assertRaises(ValueError) as context:
            s1.continuous_adjustment(2, -23, 1)
        self.assertIn('continuous_adjustment(): time <= self.continuous_time', str(context.exception))

        self.assertEqual(s1.continuous_adjustment(2, 4, 1), 12)
        self.assertEqual(s1.get_population(4.0), 12)
        self.assertEqual(s1.get_population(6.0), 14)

        # ensure that continuous_adjustment() returns an integral population
        adjusted_pop = s1.continuous_adjustment(0.5, 5, 0)
        self.assertEqual(int(adjusted_pop), adjusted_pop)

        with self.assertRaises(ValueError) as context:
            s1.continuous_adjustment(2, 3, 1)
        self.assertIn('continuous_adjustment(): time <= self.continuous_time', str(context.exception))

        with self.assertRaises(ValueError) as context:
            s1.get_population()
        self.assertIn('get_population(): time needed because continuous adjustment received at time',
            str(context.exception))

        with self.assertRaises(ValueError) as context:
            s1.get_population(3)
        self.assertIn('get_population(): time < self.continuous_time', str(context.exception))

        s1 = Specie('specie', 10)
        with self.assertRaises(ValueError) as context:
            s1.continuous_adjustment(2, 2, 1)
        self.assertIn('initial flux was not provided', str(context.exception))

        # raise asserts
        with self.assertRaises(AssertionError) as context:
            s1 = Specie('specie', -10)
        self.assertIn('__init__(): population should be >= 0', str(context.exception))

    def test_NegativePopulationError(self):
        s='specie_3'
        args = ('m', s, 2, -4.0)
        n1 = NegativePopulationError(*args)
        self.assertEqual(n1.specie, s)
        self.assertEqual(n1, NegativePopulationError(*args))
        n1.last_population += 1
        self.assertNotEqual(n1, NegativePopulationError(*args))
        self.assertTrue(n1.__ne__(NegativePopulationError(*args)))
        self.assertFalse(n1 == 3)

        p = "m(): negative population predicted for 'specie_3', with decline from 3 to -1"
        self.assertEqual(str(n1), p)
        n1.delta_time=2
        self.assertEqual(str(n1), p + " over 2 time units")
        n1.delta_time=1
        self.assertEqual(str(n1), p + " over 1 time unit")

        d = { n1:1 }
        self.assertTrue(n1 in d)

    def test_raise_NegativePopulationError(self):
        s1 = Specie('specie_3', 2, -2.0)

        with self.assertRaises(NegativePopulationError) as context:
            s1.discrete_adjustment(-3, 0)
        self.assertEqual(context.exception, NegativePopulationError('discrete_adjustment', 'specie_3', 2, -3))

        with self.assertRaises(NegativePopulationError) as context:
            s1.discrete_adjustment(0, 3)
        self.assertEqual(context.exception, NegativePopulationError('get_population', 'specie_3', 2, -6, 3))

        with self.assertRaises(NegativePopulationError) as context:
            s1.continuous_adjustment(-3, 1, 0)
        self.assertEqual(context.exception, NegativePopulationError('continuous_adjustment', 'specie_3', 2, -3.0, 1))

        with self.assertRaises(NegativePopulationError) as context:
            s1.get_population(2)
        self.assertEqual(context.exception, NegativePopulationError('get_population', 'specie_3', 2, -4.0, 2))

        s1 = Specie('specie_3', 3)
        self.assertEqual(s1.get_population(1), 3)

        with self.assertRaises(NegativePopulationError) as context:
            s1.discrete_adjustment(-4, 1)
        self.assertEqual(context.exception, NegativePopulationError('discrete_adjustment', 'specie_3', 3, -4))

    def test_Specie_stochastic_rounding(self):
        s1 = Specie('specie', 10.5)

        samples = 1000
        for i in range(samples):
            pop = s1.get_population()
            self.assertTrue(pop in [10, 11])

        mean = np.mean([s1.get_population() for i in range(samples) ])
        min = 10 + binom.ppf(0.01, n=samples, p=0.5) / samples
        max = 10 + binom.ppf(0.99, n=samples, p=0.5) / samples
        self.assertTrue(min <= mean <= max)

        s1 = Specie('specie', 10.5, initial_flux=0)
        s1.continuous_adjustment(0, 1, 0.25)
        for i in range(samples):
            self.assertEqual(s1.get_population(3), 11.0)

""" Run a simulation with another simulation object to test SpeciesPopSimObject.

A SpeciesPopSimObject manages the population of one specie, 'x'. A MockSimulationObject sends
initialization events to SpeciesPopSimObject and compares the 'x's correct population with
its simulated population.
"""

# todo: MockSimulationObjects are handy for testing other objects; generalize and place in separate module
class MockSimulationObject(SimulationObject, SimulationObjectInterface):

    def __init__(self, name, test_case, specie_id, expected_value):
        """ Init a MockSimulationObject that can unittest a specie's population.

        Args:
            test_case (:obj:`unittest.TestCase`): reference to the TestCase that launches the simulation
        """
        (self.test_case, self.specie_id, self.expected_value) = (test_case, specie_id, expected_value)
        super().__init__(name)

    def send_initial_events(self): pass

    def send_debugging_events(self, species_pop_sim_obj, update_time, update_message, update_msg_body,
        get_pop_time, get_pop_msg_body):
        self.send_event(update_time, species_pop_sim_obj, update_message, event_body=update_msg_body)
        self.send_event(get_pop_time, species_pop_sim_obj, message_types.GetPopulation,
            event_body=get_pop_msg_body)

    def handle_GivePopulation_event(self, event):
        """ Perform a unit test on the population of self.specie_id."""

        # populations is a GivePopulation_body instance
        populations = event.event_body
        self.test_case.assertEqual(populations[self.specie_id], self.expected_value,
            msg="At event_time {} for specie '{}': the correct population "
                "is {} but the actual population is {}.".format(
                event.event_time, self.specie_id, self.expected_value, populations[self.specie_id]))

    def handle_GiveProperty_event(self, event):
        """ Perform a unit test on the mass of a SpeciesPopSimObject"""
        property_name = event.event_body.property_name
        self.test_case.assertEqual(property_name, distributed_properties.MASS)
        self.test_case.assertEqual(event.event_body.value, self.expected_value)

    @classmethod
    def register_subclass_handlers(this_class):
        SimulationObject.register_handlers(this_class, [
            (message_types.GivePopulation, this_class.handle_GivePopulation_event),
            (message_types.GiveProperty, this_class.handle_GiveProperty_event)])

    @classmethod
    def register_subclass_sent_messages(this_class):
        SimulationObject.register_sent_messages(this_class,
            [message_types.GetPopulation,
            message_types.AdjustPopulationByDiscreteSubmodel,
            message_types.AdjustPopulationByContinuousSubmodel,
            message_types.GetCurrentProperty])

class TestSpeciesPopSimObjectWithAnotherSimObject(unittest.TestCase):

    def try_update_species_pop_sim_obj(self, specie_id, init_pop, mol_weight, init_flux, update_message,
        msg_body, update_time, get_pop_time, expected_value):
        """ Run a simulation that tests an update of a SpeciesPopSimObject by a update_msg_type message.

        initialize simulation:
            create SpeciesPopSimObject object
            create MockSimulationObject with reference to this TestCase and expected population value
            Mock obj sends update_message for time=update_time
            Mock obj sends GetPopulation for time=get_pop_time
        run simulation:
            SpeciesPopSimObject obj processes both messages
            SpeciesPopSimObject obj sends GivePopulation
            Mock obj receives GivePopulation and checks value
        """
        self.simulator = SimulationEngine()
        self.simulator.register_object_types([MockSimulationObject, SpeciesPopSimObject])

        if get_pop_time<=update_time:
            raise SpeciesPopulationError('get_pop_time<=update_time')
        species_pop_sim_obj = SpeciesPopSimObject('test_name',
            {specie_id:init_pop}, {specie_id:mol_weight}, initial_fluxes={specie_id:init_flux})
        mock_obj = MockSimulationObject('mock_name', self, specie_id, expected_value)
        self.simulator.load_objects([species_pop_sim_obj, mock_obj])
        mock_obj.send_debugging_events(species_pop_sim_obj, update_time, update_message, msg_body,
            get_pop_time, message_types.GetPopulation({specie_id}))
        self.simulator.initialize()

        self.assertEqual(self.simulator.simulate(get_pop_time+1), 3)

    def test_message_types(self):
        """ Test both discrete and continuous updates, with a range of population & flux values"""
        s_id = 's'
        update_adjustment = +5
        get_pop_time = 4
        for s_init_pop in range(3, 7, 2):
            for s_init_flux in range(-1, 2):
                for update_time in range(1, 4):

                    self.try_update_species_pop_sim_obj(s_id, s_init_pop, 0, s_init_flux,
                        message_types.AdjustPopulationByDiscreteSubmodel,
                        message_types.AdjustPopulationByDiscreteSubmodel({s_id:update_adjustment}),
                        update_time, get_pop_time,
                        s_init_pop + update_adjustment + get_pop_time*s_init_flux)

        """
        Test AdjustPopulationByContinuousSubmodel.

        Note that the expected_value does not include a term for update_time*s_init_flux. This is
        deliberately ignored by `wc_sim.multialgorithm.species_populations.Specie()` because it is
        assumed that an adjustment by a continuous submodel will incorporate the flux predicted by
        the previous iteration of that submodel.
        """
        for s_init_pop in range(3, 8, 2):
            for s_init_flux in range(-1, 2):
                for update_time in range(1, 4):
                    for updated_flux in range(-1, 2):
                        self.try_update_species_pop_sim_obj(s_id, s_init_pop, 0, s_init_flux,
                            message_types.AdjustPopulationByContinuousSubmodel,
                            message_types.AdjustPopulationByContinuousSubmodel({s_id:
                                message_types.ContinuousChange(update_adjustment, updated_flux)}),
                            update_time, get_pop_time,
                            s_init_pop + update_adjustment +
                                (get_pop_time-update_time)*updated_flux)



class InitMsg1(object): pass

class TestSpeciesPopSimObject(unittest.TestCase):

    def setUp(self):
        self.simulator = SimulationEngine()
        self.simulator.register_object_types([MockSimulationObject, SpeciesPopSimObject])
        RandomStateManager.initialize()
        self.species_ids = 's1 s2 s3'.split()
        self.initial_population = dict(zip(self.species_ids, range(3)))
        self.molecular_weight = dict(zip(self.species_ids, [10]*3))
        self.test_species_pop_sim_obj = SpeciesPopSimObject('test_name', self.initial_population,
            self.molecular_weight)
        self.simulator.add_object(self.test_species_pop_sim_obj)

    def test_init(self):
        for s in self.initial_population.keys():
            self.assertEqual(self.test_species_pop_sim_obj.read_one(0,s), self.initial_population[s])

    def test_invalid_event_types(self):

        with self.assertRaises(SimulatorError) as context:
            self.test_species_pop_sim_obj.send_event(1.0, self.test_species_pop_sim_obj, InitMsg1)
        self.assertIn("'wc_sim.multialgorithm.species_populations.SpeciesPopSimObject' simulation "
            "objects not registered to send", str(context.exception))

        with self.assertRaises(SimulatorError) as context:
            self.test_species_pop_sim_obj.send_event(1.0, self.test_species_pop_sim_obj,
                message_types.GivePopulation)
        self.assertIn("'wc_sim.multialgorithm.species_populations.SpeciesPopSimObject' simulation "
            "objects not registered to receive", str(context.exception))

