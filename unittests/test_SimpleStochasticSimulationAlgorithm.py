#!/usr/bin/env python

import unittest

from Sequential_WC_Simulator.core.SimulationObject import (EventQueue, SimulationObject)
from Sequential_WC_Simulator.core.SimulationEngine import (SimulationEngine, MessageTypesRegistry)
from Sequential_WC_Simulator.multialgorithm.SimpleStochasticSimulationAlgorithm import SimpleStochasticSimulationAlgorithm
from Sequential_WC_Simulator.unittests.UniversalSenderReceiverSimulationObject import UniversalSenderReceiverSimulationObject
from Sequential_WC_Simulator.multialgorithm.MessageTypes import (MessageTypes, 
    ADJUST_POPULATION_BY_DISCRETE_MODEL_body, 
    Continuous_change, ADJUST_POPULATION_BY_CONTINUOUS_MODEL_body, 
    GET_POPULATION_body, GIVE_POPULATION_body,
    EXECUTE_SSA_REACTION_body )

class TestSimpleStochasticSimulationAlgorithm(unittest.TestCase):

    def setUp(self):
        SimulationEngine.reset()

    @unittest.skip("skip, as not a test")
    def test_SimpleStochasticSimulationAlgorithm(self):

        # TODO(Arthur): use assert for testing; stop printing
        print 'test_SimpleStochasticSimulationAlgorithm.py:'
        print '# TODO(Arthur): EXPAND into tests'
        ssa1 = SimpleStochasticSimulationAlgorithm( 'name1' )
        ssa2 = SimpleStochasticSimulationAlgorithm( 'name2', random_seed=123, debug=True, write_plot_output=True )
        
        usr1 = UniversalSenderReceiverSimulationObject( 'usr1' )
        usr1.send_event( 1, ssa2, MessageTypes.GIVE_POPULATION, 
            event_body=GIVE_POPULATION_body( { 'x':1, 'y':2,  } ) )
        usr1.send_event( 2, ssa2, MessageTypes.EXECUTE_SSA_REACTION, 
            event_body=EXECUTE_SSA_REACTION_body( 'a reaction, internals TBD'  ) )
        SimulationEngine.simulate( 5.0 )

if __name__ == '__main__':
    try:
        unittest.main()
    except KeyboardInterrupt:
        pass
