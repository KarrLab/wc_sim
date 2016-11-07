import numpy as np
import re
import sys
import unittest

from scipy.stats import binom
from wc_sim.core.simulation_object import EventQueue, SimulationObject
from wc_sim.core.simulation_engine import SimulationEngine, MessageTypesRegistry
from wc_sim.multialgorithm.specie import Specie
from wc_utils.util.rand import RandomStateManager


class TestSpecie(unittest.TestCase):

    def setUp(self):
        RandomStateManager.initialize( )

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
        self.assertIn( 'negative time:', str(context.exception) )

        s1.continuous_adjustment( 2, 4, 1 )
        self.assertEqual( s1.get_population( 4.0 ), 12 )
        self.assertEqual( s1.get_population( 6.0 ), 14 )

        with self.assertRaises(ValueError) as context:
            s1.continuous_adjustment( 2, 3, 1 )
        self.assertIn( 'continuous_adjustment(): time <= self.continuous_time', str(context.exception) )
        
        with self.assertRaises(ValueError) as context:
            s1.get_population( )
        self.assertIn( 'get_population(): time needed because continuous adjustment received at time',
            str(context.exception) )
        
        with self.assertRaises(ValueError) as context:
            s1.get_population( 3 )
        self.assertIn( 'get_population(): time < self.continuous_time', str(context.exception) )
        
        with self.assertRaises(ValueError) as context:
            s1.discrete_adjustment( -20 )
        self.assertIn( 'discrete_adjustment(): negative population: last_population + population_change = ',
            str(context.exception) )

        with self.assertRaises(ValueError) as context:
            s1.continuous_adjustment( -22, 5, 2 )
        self.assertIn( 'continuous_adjustment(): negative population: last_population + population_change = ',
            str(context.exception) )

        s1 = Specie( 'specie', 10 )
        with self.assertRaises(ValueError) as context:
            s1.continuous_adjustment( 2, 2, 1 )
        self.assertIn( 'initial flux was not provided', str(context.exception) )

        # raise asserts
        with self.assertRaises(AssertionError) as context:
            s1 = Specie( 'specie', -10 )
        self.assertIn( '__init__(): population should be >= 0', str(context.exception) )
    
    def test_Specie_stochastic_rounding(self):
        s1 = Specie( 'specie', 10.5 )
        
        samples = 1000
        for i in range(samples):
            pop = s1.get_population( )
            self.assertTrue( pop in [10, 11] )
        
        mean = np.mean([s1.get_population( ) for i in range(samples) ])
        min = 10 + binom.ppf(0.01, n=samples, p=0.5) / samples
        max = 10 + binom.ppf(0.99, n=samples, p=0.5) / samples
        self.assertTrue( min <= mean <= max )
        
        s1 = Specie( 'specie', 10.5, initial_flux=0 )
        s1.continuous_adjustment( 0, 1, 0.25 )
        for i in range(samples):
            self.assertEqual( s1.get_population( 3 ), 11.0 )
