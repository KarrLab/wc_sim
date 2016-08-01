#!/usr/bin/env python

import unittest

from Sequential_WC_Simulator.core.SimulationObject import (EventQueue, SimulationObject)
from Sequential_WC_Simulator.core.SimulationEngine import (SimulationEngine, MessageTypesRegistry)
from Sequential_WC_Simulator.multialgorithm.submodels.simple_SSA_submodel import simple_SSA_submodel
from Sequential_WC_Simulator.multialgorithm.MessageTypes import *
from tests.UniversalSenderReceiverSimulationObject import UniversalSenderReceiverSimulationObject

class Testsimple_SSA_submodel(unittest.TestCase):

    def setUp(self):
        SimulationEngine.reset()

    @unittest.skip("skip, as not a test, and this code doesn't run")
    def test_simple_SSA_submodel(self):

        # TODO(Arthur): use assert for testing; stop printing
        ssa1 = simple_SSA_submodel( 'name1' )
        ssa2 = simple_SSA_submodel( 'name2', random_seed=123, debug=True, write_plot_output=True )
        
        usr1 = UniversalSenderReceiverSimulationObject( 'usr1' )
        usr1.send_event( 1, ssa2, GIVE_POPULATION, 
            event_body=GIVE_POPULATION.body( { 'x':1, 'y':2,  } ) )
        usr1.send_event( 2, ssa2, EXECUTE_SSA_REACTION )
        SimulationEngine.simulate( 5.0 )

if __name__ == '__main__':
    try:
        unittest.main()
    except KeyboardInterrupt:
        pass
