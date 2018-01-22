'''
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2017-03-26
:Copyright: 2016-2018, Karr Lab
:License: MIT
'''

import unittest
import os, sys
import six
from scipy.constants import Avogadro

from wc_lang.io import Reader
from wc_lang.core import Reaction, SpeciesType, Species
from wc_sim.core.simulation_object import SimulationObject, SimulationObjectInterface
from wc_sim.core.simulation_engine import SimulationEngine
from wc_sim.multialgorithm.submodels.submodel import Submodel
from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.multialgorithm.species_populations import (LOCAL_POP_STORE, LocalSpeciesPopulation,
    AccessSpeciesPopulations)
from wc_sim.multialgorithm import message_types, distributed_properties
from wc_sim.multialgorithm.utils import get_species_and_compartment_from_name

# todo: MockSimulationObjects are handy for testing other objects; generalize and place in separate module
class MockSimulationObject(SimulationObject, SimulationObjectInterface):

    def __init__(self, name, test_case, expected_value):
        '''Init a MockSimulationObject that can unittest a `SimulationObject`s behavior

        Args:
            test_case (:obj:`unittest.TestCase`): reference to the TestCase that launches the simulation
        '''
        (self.test_case, self.expected_value) = (test_case, expected_value)
        super(MockSimulationObject, self).__init__(name)

    def send_initial_events(self): pass

    def handle_GiveProperty_event(self, event):
        '''Perform a unit test on the molecular weight of a SpeciesPopSimObject'''
        property_name = event.event_body.property_name
        self.test_case.assertEqual(property_name, distributed_properties.MASS)
        self.test_case.assertEqual(event.event_body.value, self.expected_value)

    @classmethod
    def register_subclass_handlers(this_class):
        SimulationObject.register_handlers(this_class, [
            (message_types.GiveProperty, this_class.handle_GiveProperty_event)])

    @classmethod
    def register_subclass_sent_messages(this_class):
        SimulationObject.register_sent_messages(this_class, [message_types.GetCurrentProperty])


# todo: test submodels with shared species
class TestSubmodel(unittest.TestCase):

    MODEL_FILENAME = os.path.join(os.path.dirname(__file__), 'fixtures',
        'test_submodel_no_shared_species.xlsx')

    def setUp(self):
        SpeciesType.objects.reset()
        self.model = Reader().run(self.MODEL_FILENAME)
        self.multialgorithm_simulation = MultialgorithmSimulation(self.model, None)
        self.multialgorithm_simulation.initialize()

        self.submodels = {}
        for lang_submodel in self.model.get_submodels():
            access_species_pop = self.multialgorithm_simulation.create_access_species_pop(lang_submodel)
            self.submodels[lang_submodel.id] = Submodel(None,
                    lang_submodel.name,
                    access_species_pop,
                    list(lang_submodel.reactions),
                    lang_submodel.get_species(),
                    lang_submodel.compartment,
                    None)

    def test_get_species_ids(self):
        for id,submodel in self.submodels.items():
            self.assertEqual(set(self.multialgorithm_simulation.private_species[id]),
                set(submodel.get_species_ids()))

    def test_get_specie_concentrations(self):
        actual_concentrations = {}
        for conc in self.model.get_concentrations():
            actual_concentrations[
                Species.gen_id(conc.species.species_type, conc.species.compartment)] = conc.value
        for submodel in self.submodels.values():
            for k,v in submodel.get_specie_concentrations().items():
                if k in actual_concentrations:
                    self.assertAlmostEqual(actual_concentrations[k], v)

    def test_mass(self):
        for submodel in self.submodels.values():
            mass = 0
            for specie_id in submodel.get_species_ids():
                (species_type_id, _) = get_species_and_compartment_from_name(specie_id)
                mass += (self.multialgorithm_simulation.init_populations[specie_id]*
                        SpeciesType.objects.get_one(id=species_type_id).molecular_weight)/Avogadro
            self.assertAlmostEqual(submodel.mass(), mass, places=30)

    def test_enabled_reaction(self):
        actually_enabled = ['reaction_1', 'reaction_3', 'reaction_4']
        for submodel in self.submodels.values():
            for rxn in submodel.reactions:
                self.assertEqual(rxn.id in actually_enabled, submodel.enabled_reaction(rxn))

    def test_execute_reaction(self):
        # test one reaction 'by hand'
        adjustments = {'reaction_3': {'specie_2[c]':-1, 'specie_4[c]':-2, 'specie_5[c]':1}}
        for submodel in self.submodels.values():
            for reaction in submodel.reactions:
                if reaction.id in adjustments:
                    before = submodel.get_specie_counts()
                    submodel.execute_reaction(reaction)
                    after = submodel.get_specie_counts()
                    for specie_id in adjustments[reaction.id].keys():
                        self.assertEqual(adjustments[reaction.id][specie_id],
                            after[specie_id]-before[specie_id])


# todo: test submodels with running simulator
# todo: eliminate redundant code in test_submodel.py; search for distributed_properties
'''
class TestSubmodelSimulating(unittest.TestCase):

    def test_mass(self):
        for submodel in self.submodels.values():
            mass = sum([self.initial_population[specie_id]*self.molecular_weights[specie_id]/Avogadro
                for specie_id in self.species_ids])
            mock_obj = MockSimulationObject('', self, mass)
            self.simulator.add_object(mock_obj)
            mock_obj.send_event(1, submodel, message_types.GetCurrentProperty,
                message_types.GetCurrentProperty(distributed_properties.MASS))
            self.simulator.initialize()
            simulator = self.multialgorithm_simulation.build_simulation()
            simulator.simulate(2)
'''
