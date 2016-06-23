#!/usr/bin/env python

import unittest
import sys
import re

from SequentialSimulator.core.SimulationObject import (EventQueue, SimulationObject)
from SequentialSimulator.core.SimulationEngine import SimulationEngine
from SequentialSimulator.multialgorithm.MessageTypes import MessageTypes
from SequentialSimulator.multialgorithm.CellState import (Specie, CellState)


class TestSpecie(unittest.TestCase):

    def test_Specie(self):
        # covers all executable statements in Specie(), including exceptions
        s1 = Specie( 10 )
        
        self.assertEqual( s1.get_population( ), 10 )
        s1.discrete_adjustment( 1 )
        self.assertEqual( s1.get_population( ), 11 )
        s1.discrete_adjustment( -1 )
        self.assertEqual( s1.get_population( ), 10 )

        with self.assertRaises(AssertionError) as context:
            s1.continuous_adjustment( 2, -23, 1 )
        self.assertIn( 'negative time:', context.exception.message )

        s1.continuous_adjustment( 2, 4, 1 )
        self.assertEqual( s1.get_population( 4.0 ), 12 )
        self.assertEqual( s1.get_population( 6.0 ), 14 )

        with self.assertRaises(ValueError) as context:
            s1.continuous_adjustment( 2, 3, 1 )
        self.assertIn( 'continuous_adjustment(): time <= self.continuous_time', context.exception.message )
        
        with self.assertRaises(ValueError) as context:
            s1.get_population( )
        self.assertIn( 'get_population(): time needed because continuous adjustment received at time',
            context.exception.message )
        
        with self.assertRaises(ValueError) as context:
            s1.get_population( 3 )
        self.assertIn( 'get_population(): time < self.continuous_time', context.exception.message )
        
        with self.assertRaises(ValueError) as context:
            s1.discrete_adjustment( -20 )
        self.assertIn( 'discrete_adjustment(): negative population from self.last_population + population_change',
            context.exception.message )

        with self.assertRaises(ValueError) as context:
            s1.continuous_adjustment( -22, 5, 2 )
        self.assertIn( 'continuous_adjustment(): negative population from self.last_population + population_change',
            context.exception.message )

        # raise asserts
        with self.assertRaises(AssertionError) as context:
            s1 = Specie( -10 )
        self.assertIn( '__init__(): population should be >= 0', context.exception.message )
        with self.assertRaises(AssertionError) as context:
            s1.continuous_adjustment( 2, 5, -22 )
        self.assertIn( 'negative flux:', context.exception.message )
    
if __name__ == '__main__':
    try:
        unittest.main()
    except KeyboardInterrupt:
        pass

