#!/usr/bin/env python

import unittest
import sys

from SequentialSimulator.core.SimulationObject import (EventQueue, SimulationObject)
from SequentialSimulator.core.SimulationEngine import SimulationEngine
from SequentialSimulator.multialgorithm.MessageTypes import MessageTypes
from SequentialSimulator.multialgorithm.CellState import (Specie, CellState)

class TestCellState(unittest.TestCase):

    id = 0
    @staticmethod
    def get_name():
        TestCellState.id += 1
        return "CellState_{:d}".format( TestCellState.id )

    def setUp(self):
        SimulationEngine.reset()

    def test_specie(self):
        # covers all executable statements in Specie()
        s1 = Specie( 10 )
        s1.continuous_update( 2, 3, 1 )
        
        self.assertEqual( s1.get_population( ), 10 )
        s1.discrete_update( 1 )
        self.assertEqual( s1.get_population( ), 11 )
        s1.discrete_update( -1 )
        self.assertEqual( s1.get_population( ), 10 )

        with self.assertRaises(AssertionError) as context:
            s1.continuous_update( 2, -23, 1 )
        self.assertIn( 'negative time:', context.exception.message )

        s1.continuous_update( 2, 4, 1 )
        self.assertEqual( s1.get_population( 4.0 ), 12 )
        self.assertEqual( s1.get_population( 6.0 ), 14 )

        with self.assertRaises(ValueError) as context:
            s1.continuous_update( 2, 3, 1 )
        self.assertIn( 'continuous_update(): time <= self.continuous_time', context.exception.message )
        
        with self.assertRaises(ValueError) as context:
            s1.get_population( )
        self.assertIn( 'get_population(): time needed because continuous updates received at time',
            context.exception.message )
        
        with self.assertRaises(ValueError) as context:
            s1.get_population( 3 )
        self.assertIn( 'get_population(): time < self.continuous_time', context.exception.message )
        
        with self.assertRaises(ValueError) as context:
            s1.discrete_update( -20 )
        self.assertIn( 'discrete_update(): negative population from self.last_population + population_change',
            context.exception.message )

        with self.assertRaises(ValueError) as context:
            s1.continuous_update( -22, 5, 2 )
        self.assertIn( 'continuous_update(): negative population from self.last_population + population_change',
            context.exception.message )

        # raise asserts
        with self.assertRaises(AssertionError) as context:
            s1 = Specie( -10 )
        self.assertIn( '__init__(): population should be >= 0', context.exception.message )
        with self.assertRaises(AssertionError) as context:
            s1.continuous_update( 2, 5, -22 )
        self.assertIn( 'negative flux:', context.exception.message )

    
    def test_invalid_event_types(self):
    
        # CellState( name, initial_population, type, debug=False ) 
        pop = dict( zip( 's1 s2 s3'.split(), range(3) ) )
        cs1 = CellState( 'name', pop, CellState.DISCRETE_TYPE, debug=False ) 
        # initial events
        cs1.send_event( 1.0, cs1, 'init_msg1' )
        cs1.send_event( 1.0, cs1, 'init_msg2' )
        with self.assertRaises(ValueError) as context:
            SimulationEngine.simulate( 5.0 )
        self.assertIn( 'init_msg1, init_msg2', context.exception.message )
        self.assertIn( 'invalid event event_type(s)', context.exception.message )

    @unittest.skip("demonstrating skipping")
    def test_init_options(self):
        pop2 = dict( zip( 's1 s2 s3'.split(), map( lambda x: x*7, range(3,6) ) ) )
        for type in [CellState.DISCRETE_TYPE, CellState.CONTINUOUS_TYPE ]:
            for debug in [True, False]:
                for plot_output in [True, False]:
                    SimulationEngine.reset()
                    name = TestCellState.get_name()
                    print "Creating CellState( {}, --population--, {}, debug={}, write_plot_output={} ) ".format(
                        name, type, debug, plot_output )
                    cs1 = CellState( name, pop2, type, 
                        debug=debug, write_plot_output=plot_output ) 
                    for msg_type in [MessageTypes.CHANGE_POPULATION, MessageTypes.GET_POPULATION]:  # , 'junk'
                        # initial events
                        cs1.send_event( 1.0, cs1, msg_type )
                        cs1.send_event( 1.0, cs1, msg_type )
                    SimulationEngine.simulate( 5.0 )
    

if __name__ == '__main__':
    try:
        unittest.main()
    except KeyboardInterrupt:
        pass

