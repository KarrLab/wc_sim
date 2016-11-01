''' Test specie.py.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-10-01
:Copyright: 2016, Karr Lab
:License: MIT
'''
import unittest
import sys
import re

from wc_utils.util.RandomUtilities import ReproducibleRandom

from Sequential_WC_Simulator.core.simulation_object import EventQueue, SimulationObject
from Sequential_WC_Simulator.core.simulation_engine import SimulationEngine, MessageTypesRegistry
from Sequential_WC_Simulator.multialgorithm.specie import Specie
from Sequential_WC_Simulator.multialgorithm.specie import NegativePopulationError

class TestSpecie(unittest.TestCase):

    def setUp(self):
        ReproducibleRandom.init( )

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

        p = "m(): negative population for 'specie_3', with decline from 3 to -1"
        self.assertEqual( str(n1), p )
        n1.delta_time=2
        self.assertEqual( str(n1), p + " over 2 time units" )

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
            self.assertTrue( pop == 10 or pop == 11 )

        mean = sum(s1.get_population( ) for i in range(samples) ) / float(samples)
        # TODO(Arthur): make sure P[ 10.4 <= mean <= 10.6 ] is high enough
        self.assertTrue( 10.4 <= mean <= 10.6 )

        s1 = Specie( 'specie', 10.5, initial_flux=0 )
        s1.continuous_adjustment( 0, 1, 0.25 )
        for i in range(samples):
            self.assertEqual( s1.get_population( 3 ), 11.0 )

        ReproducibleRandom.init( seed=123 )
        s2 = Specie( 'specie', 10.5 )
        pops=[]
        for i in range(10):
            pops.append( s2.get_population( ) )
        expected_pops = [10, 11, 10, 10, 11, 10, 11, 10, 11, 11]
        self.assertEqual( pops, expected_pops )
