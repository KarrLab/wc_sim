import unittest

from Sequential_WC_Simulator.core.simulation_object import (EventQueue, SimulationObject)
from Sequential_WC_Simulator.core.simulation_engine import (SimulationEngine, MessageTypesRegistry)
from Sequential_WC_Simulator.multialgorithm.submodels.simple_SSA_submodel import simple_SSA_submodel
from Sequential_WC_Simulator.multialgorithm.message_types import *
from tests.universal_sender_receiver_simulation_object import UniversalSenderReceiverSimulationObject

class Testsimple_SSA_submodel(unittest.TestCase):

    def setUp(self):
        SimulationEngine.reset()

    @unittest.skip("skip, as not a test, and this code doesn't run")
    def test_simple_SSA_submodel(self):

        # TODO(Arthur): use assert for testing; stop printing
        ssa1 = simple_SSA_submodel( 'name1' )
        ssa2 = simple_SSA_submodel( 'name2', debug=True )
        
        usr1 = UniversalSenderReceiverSimulationObject( 'usr1' )
        usr1.send_event( 1, ssa2, GivePopulation, 
            event_body=GivePopulation.body( { 'x':1, 'y':2,  } ) )
        usr1.send_event( 2, ssa2, ExecuteSSAReaction )
        SimulationEngine.simulate( 5.0 )
