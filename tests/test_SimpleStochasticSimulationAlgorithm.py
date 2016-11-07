import unittest

from wc_sim.core.simulation_object import (EventQueue, SimulationObject)
from wc_sim.core.simulation_engine import (SimulationEngine, MessageTypesRegistry)
from wc_sim.multialgorithm.submodels.ssa import SsaSubmodel
from wc_sim.multialgorithm.message_types import GivePopulation, ExecuteSsaReaction
from tests.universal_sender_receiver_simulation_object import UniversalSenderReceiverSimulationObject

class TestSsaSubmodel(unittest.TestCase):

    def setUp(self):
        SimulationEngine.reset()

    @unittest.skip("skip, as not a test, and this code doesn't run")
    def test_ssa_submodel(self):

        # TODO(Arthur): use assert for testing; stop printing
        ssa1 = SsaSubmodel( 'name1' )
        ssa2 = SsaSubmodel( 'name2', debug=True )
        
        usr1 = UniversalSenderReceiverSimulationObject( 'usr1' )
        usr1.send_event( 1, ssa2, GivePopulation, 
            event_body=GivePopulation.Body( { 'x':1, 'y':2,  } ) )
        usr1.send_event( 2, ssa2, ExecuteSsaReaction )
        SimulationEngine.simulate( 5.0 )
