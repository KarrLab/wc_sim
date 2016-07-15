#!/usr/bin/env python

import unittest

from Sequential_WC_Simulator.multialgorithm.MessageTypes import (MessageTypes, ADJUST_POPULATION_BY_DISCRETE_MODEL_body, 
    Continuous_change, ADJUST_POPULATION_BY_CONTINUOUS_MODEL_body, GET_POPULATION_body, GIVE_POPULATION_body)

# TODO(Arthur): converg to unittests
'''
d={
'x':Continuous_change(4,.5) ,
'y':Continuous_change(3,1) 
}
t=ADJUST_POPULATION_BY_CONTINUOUS_MODEL_body( d )

v={ 'x':5, 'y':222}
t=GIVE_POPULATION_body( v )
print t
etc.
'''

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

if __name__ == '__main__':
    try:
        unittest.main()
    except KeyboardInterrupt:
        pass
