#!/usr/bin/env python

import unittest
import sys

from SequentialSimulator.core.SimulationObject import (EventQueue, SimulationObject)
from SequentialSimulator.core.SimulationEngine import SimulationEngine
from SequentialSimulator.multialgorithm.CellState import CellState


class TestCellState(unittest.TestCase):

    def setUp(self):
        SimulationEngine.reset()

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

if __name__ == '__main__':
    try:
        unittest.main()
    except KeyboardInterrupt:
        pass

