''' Test specie.py.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-10-01
:Copyright: 2016, Karr Lab
:License: MIT
'''
import numpy as np
import unittest
import sys
import re
from scipy.stats import binom

from wc_sim.core.simulation_object import EventQueue, SimulationObject
from wc_sim.core.simulation_engine import SimulationEngine, MessageTypesRegistry
from wc_sim.multialgorithm.specie import Specie, NegativePopulationError
from wc_utils.util.rand import RandomStateManager

class TestSpecie(unittest.TestCase):

    def setUp(self):
        RandomStateManager.initialize( )

    def test_Specie(self):
        s1 = Specie( 'specie', 10 )

        self.assertEqual( s1.get_population( ), 10 )
        self.assertEqual( s1.discrete_adjustment( 1, 0 ), 11 )
        self.assertEqual( s1.get_population( ), 11 )
        self.assertEqual( s1.discrete_adjustment( -1, 0 ), 10 )
        self.assertEqual( s1.get_population( ), 10 )

        s1 = Specie( 'specie_3', 2, 1 )
        self.assertEqual( s1.discrete_adjustment( 3, 4 ), 9 )

        s1 = Specie( 'specie', 10, initial_flux=0 )
        self.assertEqual( 'specie_name:specie; last_population:10; continuous_time:0; continuous_flux:0', str(s1) )

        with self.assertRaises(ValueError) as context:
            s1.continuous_adjustment( 2, -23, 1 )
        self.assertIn( 'continuous_adjustment(): time <= self.continuous_time', str(context.exception) )

        self.assertEqual( s1.continuous_adjustment( 2, 4, 1 ), 12 )
        self.assertEqual( s1.get_population( 4.0 ), 12 )
        self.assertEqual( s1.get_population( 6.0 ), 14 )

        # ensure that continuous_adjustment() returns an integral population
        adjusted_pop = s1.continuous_adjustment( 0.5, 5, 0 )
        self.assertEqual( int(adjusted_pop), adjusted_pop )

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

        s1 = Specie( 'specie', 10 )
        with self.assertRaises(ValueError) as context:
            s1.continuous_adjustment( 2, 2, 1 )
        self.assertIn( 'initial flux was not provided', str(context.exception) )

        # raise asserts
        with self.assertRaises(AssertionError) as context:
            s1 = Specie( 'specie', -10 )
        self.assertIn( '__init__(): population should be >= 0', str(context.exception) )

    def test_NegativePopulationError(self):
        s='specie_3'
        args = ('m', s, 2, -4.0)
        n1 = NegativePopulationError( *args )
        self.assertEqual( n1.specie, s )
        self.assertEqual( n1, NegativePopulationError( *args ) )
        n1.last_population += 1
        self.assertNotEqual( n1, NegativePopulationError( *args ) )
        self.assertTrue( n1.__ne__( NegativePopulationError( *args ) ) )
        self.assertFalse( n1 == 3 )

        p = "m(): negative population predicted for 'specie_3', with decline from 3 to -1"
        self.assertEqual( str(n1), p )
        n1.delta_time=2
        self.assertEqual( str(n1), p + " over 2 time units" )
        n1.delta_time=1
        self.assertEqual( str(n1), p + " over 1 time unit" )

        d = { n1:1 }
        self.assertTrue( n1 in d )

    def test_raise_NegativePopulationError(self):
        s1 = Specie( 'specie_3', 2, -2.0 )

        with self.assertRaises(NegativePopulationError) as context:
            s1.discrete_adjustment( -3, 0 )
        self.assertEqual( context.exception, NegativePopulationError('discrete_adjustment', 'specie_3', 2, -3) )

        with self.assertRaises(NegativePopulationError) as context:
            s1.discrete_adjustment( 0, 3 )
        self.assertEqual( context.exception, NegativePopulationError('get_population', 'specie_3', 2, -6, 3) )

        with self.assertRaises(NegativePopulationError) as context:
            s1.continuous_adjustment( -3, 1, 0 )
        self.assertEqual( context.exception, NegativePopulationError('continuous_adjustment', 'specie_3', 2, -3.0, 1) )

        with self.assertRaises(NegativePopulationError) as context:
            s1.get_population( 2 )
        self.assertEqual( context.exception, NegativePopulationError('get_population', 'specie_3', 2, -4.0, 2) )

        s1 = Specie( 'specie_3', 3 )
        self.assertEqual( s1.get_population( 1 ), 3 )

        with self.assertRaises(NegativePopulationError) as context:
            s1.discrete_adjustment( -4, 1 )
        self.assertEqual( context.exception, NegativePopulationError('discrete_adjustment', 'specie_3', 3, -4) )

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
