#!/usr/bin/env python

import unittest

from Sequential_WC_Simulator.multialgorithm.MessageTypes import *

class TestMessageTypes(unittest.TestCase):
    
    def test_type_check(self):
    
        change, flux = 7, 9
        t = Continuous_change( change, flux )
        self.assertEqual( t.change, change )
        self.assertEqual( t.flux, flux )

        for (change, flux) in ( (None, 3), ('x', 7), ):
            with self.assertRaises(ValueError) as context:
                Continuous_change( change, flux )
            self.assertRegexpMatches( context.exception.message, 
                "Continuous_change.type_check\(\): .* is '.*' which is not an int or float" )

    def test_one_message_type(self):
    
        self.assertEqual( ADJUST_POPULATION_BY_CONTINUOUS_MODEL.__name__,
            'ADJUST_POPULATION_BY_CONTINUOUS_MODEL' )
        specie_id = 'x'
        adjustment = 7
        body = ADJUST_POPULATION_BY_CONTINUOUS_MODEL.body( { specie_id:adjustment } )
        self.assertEqual( body.population_change[specie_id], adjustment )

if __name__ == '__main__':
    try:
        unittest.main()
    except KeyboardInterrupt:
        pass
