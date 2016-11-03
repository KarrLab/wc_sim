import unittest

from wc_utils.util.stat_utils import ExponentialMovingAverage

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
        self.assertIn( 'alpha or center_of_mass must be provided', str(context.exception) )
        with self.assertRaises(ValueError) as context:
            ExponentialMovingAverage( x, alpha=2 )
        self.assertIn( 'alpha should satisfy 0<alpha<1: but alpha=2', str(context.exception) )
