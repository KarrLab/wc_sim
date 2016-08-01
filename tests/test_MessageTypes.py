import unittest

from Sequential_WC_Simulator.multialgorithm.MessageTypes import *
from Sequential_WC_Simulator.core.utilities import compare_name_with_class

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
    
        self.assertTrue( compare_name_with_class( 'ADJUST_POPULATION_BY_CONTINUOUS_MODEL',
            ADJUST_POPULATION_BY_CONTINUOUS_MODEL ) )
        specie_id = 'x'
        adjustment = 7
        body = ADJUST_POPULATION_BY_CONTINUOUS_MODEL.body( { specie_id:adjustment } )
        self.assertEqual( body.population_change[specie_id], adjustment )
