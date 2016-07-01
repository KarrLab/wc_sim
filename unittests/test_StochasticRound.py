#!/usr/bin/env python

import unittest
from random import Random

from SequentialSimulator.multialgorithm.CellState import StochasticRound

class TestStochasticRound(unittest.TestCase):

    @staticmethod
    def get_sequence_of_rounds( samples, value, seed=None ):
        aStochasticRound = StochasticRound( seed=seed )
        return [ aStochasticRound.Round( value ) for j in range(samples) ]

    def test_seed( self ):
        seed = 123
        samples = 100
        value = 3.5
        initial_results = TestStochasticRound.get_sequence_of_rounds( samples, value, seed=seed )
        for i in range(samples):
            test_results = TestStochasticRound.get_sequence_of_rounds( samples, value, seed=seed )
            self.assertEquals( initial_results, test_results )

        for i in range(samples):
            test_results = TestStochasticRound.get_sequence_of_rounds( samples, value )
            # P[ this test failing | Random is truly random ] = 2**-100 = (2**-10)**10 ~= (10**-3)**10 = 10**-30
            self.assertNotEquals( initial_results, test_results )
        
    def test_mean( self ):
        # the mean of a set of values should converge towards the mean of stochastic rounds of the same set
        myRandom = Random()
        samples = 10000000
        lower, upper = (0,10)
        values = [myRandom.uniform( lower, upper ) for i in range(samples)]
        mean_values = sum(values)/float(samples)
        aStochasticRound = StochasticRound( )
        mean_stochastic_rounds_values = \
            sum( map( lambda x: aStochasticRound.Round( x ), values) )/float(samples)
        '''
        print "samples: {:7d} mean_values: {:8.5f} mean_stochastic_rounds: {:8.5f}".format( 
            samples, mean_values, mean_stochastic_rounds_values )
        '''
        # TODO(Arthur): determine an analytic relationship between samples and places
        self.assertAlmostEqual( mean_values, mean_stochastic_rounds_values, places=3 )
        
if __name__ == '__main__':
    try:
        unittest.main()
    except KeyboardInterrupt:
        pass

        

