#!/usr/bin/env python

import unittest

from Sequential_WC_Simulator.core.utilities import ExponentialMovingAverage

class TestExponentialMovingAverage(unittest.TestCase):

    def test_ExponentialMovingAverage(self):
        # test sequences of averages that have simple analyic solutions
        x = 2**10
        ema = ExponentialMovingAverage( x, center_of_mass=1,  )
        self.assertEqual( x, ema.get_value() )
        for i in range(10):
            self.assertEqual( x, ema.add_value(x) )
        ema = ExponentialMovingAverage( x, alpha=0.75,  )
        for i in range( 10 ):
            self.assertEqual( x, ema.add_value(x) )
        with self.assertRaises(ValueError) as context:
            ExponentialMovingAverage( x )
        self.assertIn( 'alpha or center_of_mass must be provided', context.exception.message )
        with self.assertRaises(ValueError) as context:
            ExponentialMovingAverage( x, alpha=2 )
        self.assertIn( 'alpha should satisfy 0<alpha<1: but alpha=2', context.exception.message )
        
if __name__ == '__main__':
    try:
        unittest.main()
    except KeyboardInterrupt:
        pass

