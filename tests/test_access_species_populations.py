'''Test access_species_populations.py.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-12-04
:Copyright: 2016, Karr Lab
:License: MIT
'''

import os, unittest, copy
import string
from scipy.constants import Avogadro
from six import iteritems

from wc_lang.io import Excel
from wc_sim.core.simulation_engine import SimulationEngine
from wc_sim.multialgorithm import message_types
from wc_sim.multialgorithm.executable_model import ExecutableModel
from wc_sim.multialgorithm.access_species_populations import AccessSpeciesPopulations as ASP
from wc_sim.multialgorithm.access_species_populations import LOCAL_POP_STORE
from wc_sim.multialgorithm.local_species_population import LocalSpeciesPopulation
from wc_sim.multialgorithm.species_population_cache import SpeciesPopulationCache
from wc_sim.multialgorithm.species_pop_sim_object import SpeciesPopSimObject
from wc_sim.multialgorithm.submodels.skeleton_submodel import SkeletonSubmodel
from wc_sim.multialgorithm.multialgorithm_errors import SpeciesPopulationError

def store_i(i):
    return "store_{}".format(i)

def specie_l(l):
    return "specie_{}".format(l)

remote_pop_stores = {store_i(i):None for i in range(1, 4)}
species_ids = [specie_l(l) for l in list(string.ascii_lowercase)[0:5]]

class TestAccessSpeciesPopulations(unittest.TestCase):

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures',
        'test_model_for_access_species_populations.xlsx')

    MODEL_FILENAME_STEADY_STATE = os.path.join(os.path.dirname(__file__), 'fixtures',
        'test_model_for_access_species_populations_2.xlsx')

    def setUp(self):
        self.an_ASP = ASP(None, remote_pop_stores)
        SimulationEngine.reset()

    def set_up_simulation(self, model_file):
        '''Set up a simulation from a test model.

        Create two SkeletonSubmodels, a LocalSpeciesPopulation for each, and
        a SpeciesPopSimObject that they share.
        '''

        # make a model
        self.model = Excel.read(model_file)
        # make SpeciesPopSimObjects
        self.private_species = ExecutableModel.find_private_species(self.model)
        self.shared_species = ExecutableModel.find_shared_species(self.model)

        self.init_populations={}
        for s in self.model.species:
            for c in s.concentrations:
                sc_id = '{}[{}]'.format(s.id, c.compartment.id)
                self.init_populations[sc_id]=int(c.value * c.compartment.initial_volume * Avogadro)

        self.shared_pop_sim_obj = {}
        self.shared_pop_sim_obj['shared_store_1'] = SpeciesPopSimObject('shared_store_1',
            {specie_id:self.init_populations[specie_id] for specie_id in self.shared_species})
        print(self.shared_pop_sim_obj['shared_store_1'])

        # make submodels and their parts
        self.submodels={}
        for submodel in self.model.submodels:

            # make LocalSpeciesPopulations
            local_species_population = LocalSpeciesPopulation(self.model,
                submodel.id.replace('_', '_lsp_'),
                {specie_id:self.init_populations[specie_id] for specie_id in
                    self.private_species[submodel.id]},
                initial_fluxes={specie_id:0 for specie_id in self.private_species[submodel.id]})

            print(local_species_population)

            # make AccessSpeciesPopulations objects
            # TODO(Arthur): stop giving all SpeciesPopSimObjects to each AccessSpeciesPopulations
            access_species_population = ASP(local_species_population, self.shared_pop_sim_obj)

            # make SkeletonSubmodels
            behavior = {'INTER_REACTION_TIME':1}
            self.submodels[submodel.id] = SkeletonSubmodel(behavior, self.model, submodel.id,
                access_species_population, submodel.reactions, submodel.species, submodel.parameters)
            # connect AccessSpeciesPopulations object to its affiliated SkeletonSubmodels
            access_species_population.set_submodel(self.submodels[submodel.id])

            species_population_cache = SpeciesPopulationCache(access_species_population)
            access_species_population.set_species_population_cache(species_population_cache)

            # make access_species_population.species_locations
            access_species_population.add_species_locations(LOCAL_POP_STORE,
                self.private_species[submodel.id])
            access_species_population.add_species_locations('shared_store_1', self.shared_species)

    def test_species_locations(self):

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

    def test_species_locations_exceptions(self):
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
            ASP(None, {'a':None, LOCAL_POP_STORE:None})
        self.assertIn("{} not a valid remote_pop_store name".format(LOCAL_POP_STORE),
            str(cm.exception))

    def test_population_changes(self):
        '''Test population changes that occur without using event messages.'''
        self.set_up_simulation(self.MODEL_FILENAME)
        theASP = self.submodels['submodel_1'].access_species_population
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

    def initialize_simulation(self, model_file):
        self.set_up_simulation(model_file)
        delay_to_first_event = 1.0/len(self.submodels)
        for name,submodel in iteritems(self.submodels):

            # prefetch into caches
            submodel.access_species_population.prefetch(delay_to_first_event,
                submodel.get_species_ids())

            # send initial event messages
            msg_body = message_types.ExecuteSsaReaction.Body(0)
            submodel.send_event(delay_to_first_event, submodel, message_types.ExecuteSsaReaction,
                msg_body)

            delay_to_first_event += 1/len(self.submodels)

    def verify_simulation(self, expected_final_pops, sim_end):
        '''Verify the final simulatin populations.'''
        for specie_id in self.shared_species:
            pop = self.shared_pop_sim_obj['shared_store_1'].read_one(sim_end, specie_id)
            self.assertEqual(expected_final_pops[specie_id], pop)

        for submodel in self.submodels.values():
            for specie_id in self.private_species[submodel.name]:
                pop = submodel.access_species_population.read_one(sim_end, specie_id)
                self.assertEqual(expected_final_pops[specie_id], pop)

    def test_simulation(self):
        '''Test a short simulation.'''

        self.initialize_simulation(self.MODEL_FILENAME)

        # run the simulation
        sim_end=3
        SimulationEngine.simulate(sim_end)

        # test final populations
        # Expected changes, based on the reactions executed
        expected_changes='''
        specie	c	e
        specie_1	-2	0
        specie_2	-2	0
        specie_3	3	-2
        specie_4	0	-1
        specie_5	0	1'''

        expected_final_pops = copy.deepcopy(self.init_populations)
        for row in expected_changes.split('\n')[2:]:
            (specie, c, e) = row.strip().split()
            for com in 'c e'.split():
                id = '{}[{}]'.format(specie, com)
                expected_final_pops[id] += float(eval(com))

        self.verify_simulation(expected_final_pops, sim_end)

    def test_stable_simulation(self):
        '''Test a steady state simulation.

        MODEL_FILENAME_STEADY_STATE contains a model with no net population change every 2 sec.
        '''
        self.initialize_simulation(self.MODEL_FILENAME_STEADY_STATE)

        # run the simulation
        sim_end=100
        SimulationEngine.simulate(sim_end)
        expected_final_pops = self.init_populations
        self.verify_simulation(expected_final_pops, sim_end)

    # TODO(Arthur): test multiple SpeciesPopSimObjects
    # TODO(Arthur): test adjust_continuously of remote_pop_stores
    # TODO(Arthur): remove evaluate coverage
    # TODO(Arthur): take care of the convert print() to log message todos
    '''
    TODO(Arthur): create a factory object that assembles a submodel with its
    AccessSpeciesPopulations, LocalSpeciesPopulation, set of SpeciesPopSimObjects and
    SpeciesPopulationCache; this will ease setting up the connections between them.
    Steps:
        1. Determine specie placements (partition)
        2. Create the SpeciesPopSimObjects
        3. For each submodel
            a. Create its LocalSpeciesPopulation
            b. Create its AccessSpeciesPopulations
            c. Create its SpeciesPopulationCache
            d. Create the submodel
            e. Connect everything
            f. Send initial messages to the submodel
    '''
