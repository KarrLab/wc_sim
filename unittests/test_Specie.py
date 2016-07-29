#!/usr/bin/env python

import unittest
import sys
import re

from Sequential_WC_Simulator.core.SimulationObject import (EventQueue, SimulationObject)
from Sequential_WC_Simulator.core.SimulationEngine import (SimulationEngine, MessageTypesRegistry)
from Sequential_WC_Simulator.multialgorithm.MessageTypes import MessageTypes
from Sequential_WC_Simulator.multialgorithm.specie import Specie


class TestSpecie(unittest.TestCase):

    # these tests cover all executable statements in Specie(), including exceptions, and 
    # all branches
    def test_Specie(self):
        s1 = Specie( 'specie', 10 )
        
        self.assertEqual( s1.get_population( ), 10 )
        s1.discrete_adjustment( 1 )
        self.assertEqual( s1.get_population( ), 11 )
        s1.discrete_adjustment( -1 )
        self.assertEqual( s1.get_population( ), 10 )

        s1 = Specie( 'specie', 10, initial_flux=0 )
        self.assertEqual( 'specie_name:specie; last_population:10; continuous_time:0; continuous_flux:0', str(s1) )
        
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
        self.assertIn( 'discrete_adjustment(): negative population: last_population + population_change = ',
            context.exception.message )

        with self.assertRaises(ValueError) as context:
            s1.continuous_adjustment( -22, 5, 2 )
        self.assertIn( 'continuous_adjustment(): negative population: last_population + population_change = ',
            context.exception.message )

        s1 = Specie( 'specie', 10 )
        with self.assertRaises(ValueError) as context:
            s1.continuous_adjustment( 2, 2, 1 )
        self.assertIn( 'initial flux was not provided', context.exception.message )

        # raise asserts
        with self.assertRaises(AssertionError) as context:
            s1 = Specie( 'specie', -10 )
        self.assertIn( '__init__(): population should be >= 0', context.exception.message )
    
    def test_Specie_stochastic_rounding(self):
        s1 = Specie( 'specie', 10.5 )
        
        samples = 1000
        for i in range(samples):
            pop = s1.get_population( )
            self.assertTrue( pop == 10 or pop == 11 )
        
        mean = sum(s1.get_population( ) for i in range(samples) ) / float(samples)
        # TODO(Arthur): make sure P[ 10.4 <= mean <= 10.6 ] is high enough 
        self.assertTrue( 10.4 <= mean <= 10.6 )
        
        s1 = Specie( 'specie', 10.5, initial_flux=0 )
        s1.continuous_adjustment( 0, 1, 0.25 )
        for i in range(samples):
            self.assertEqual( s1.get_population( 3 ), 11.0 )
            
        s2 = Specie( 'specie', 10.5, random_seed=123 )
        pops=[]
        for i in range(10):
            pops.append( s2.get_population( ) )
        expected_pops = [11.0, 11.0, 11.0, 11.0, 10.0, 11.0, 10.0, 11.0, 10.0, 11.0]
        self.assertEqual( pops, expected_pops )

        
if __name__ == '__main__':
    try:
        unittest.main()
    except KeyboardInterrupt:
        pass

