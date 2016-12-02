import unittest

from wc_utils.util.rand import RandomStateManager
from wc_sim.multialgorithm.specie import Specie
from wc_sim.multialgorithm.species_pop_sim_object import SpeciesPopSimObject
from wc_sim.core.simulation_engine import SimulationEngine, MessageTypesRegistry
from wc_sim.multialgorithm import message_types
from tests.universal_sender_receiver_simulation_object import UniversalSenderReceiverSimulationObject

class TestSpeciesPopSimObject(unittest.TestCase):

    def setUp(self):
        SimulationEngine.reset()
        RandomStateManager.initialize()
        self.initial_population = dict( zip( 's1 s2 s3'.split(), range(3) ) )
        self.test_species_pop_sim_obj = SpeciesPopSimObject('test_name', self.initial_population) 

    def test_init(self):
        for s in self.initial_population.keys():
            self.assertEqual(self.test_species_pop_sim_obj.read_one(0,s), self.initial_population[s])

    def test_invalid_event_types(self):
    
        with self.assertRaises(ValueError) as context:
            self.test_species_pop_sim_obj.send_event(1.0, self.test_species_pop_sim_obj, 'InitMsg1')
        self.assertIn( "'SpeciesPopSimObject' simulation objects not registered to send "
            "'InitMsg1' messages", str(context.exception) )

        with self.assertRaises(ValueError) as context:
            self.test_species_pop_sim_obj.send_event(1.0, self.test_species_pop_sim_obj,
                message_types.GivePopulation)
        self.assertIn( "'SpeciesPopSimObject' simulation objects not registered to receive "
            "'GivePopulation' messages", str(context.exception) )

    def test_unknown_msg(self):
    
        # check the assert statement at the end of SpeciesPopSimObject.handle_event()
        MessageTypesRegistry.set_receiver_priorities('SpeciesPopSimObject',
            [message_types.GivePopulation, message_types.AdjustPopulationByContinuousModel])
        self.test_species_pop_sim_obj.send_event(1.0, self.test_species_pop_sim_obj,
            message_types.GivePopulation)
        with self.assertRaises(AssertionError) as context:
            SimulationEngine.simulate( 5.0 )
        self.assertIn( "Shouldn't get here - GivePopulation should be covered in the if "
            "statement above", str(context.exception) )

