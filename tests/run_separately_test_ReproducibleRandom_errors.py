import unittest
import re

from Sequential_WC_Simulator.core.utilities import ReproducibleRandom as RepRand

class TestReproducibleRandomErrors(unittest.TestCase):

    def test_ReproducibleRandom_errors(self):
        with self.assertRaises(ValueError) as context:
            RepRand.check_that_init_was_called()
        self.assertIn( "Error: ReproducibleRandom: ReproducibleRandom.init() must", 
            str(context.exception) )
        
        with self.assertRaises(ValueError) as context:
            RepRand.get_numpy_random_state()
        self.assertIn( "Error: ReproducibleRandom: ReproducibleRandom.init() must", 
            str(context.exception) )