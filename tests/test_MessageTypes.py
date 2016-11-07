import unittest

from wc_utils.util.misc import isclass_by_name

from wc_sim.multialgorithm.message_types import *

class TestMessageTypes(unittest.TestCase):
    
    def test_type_check(self):
    
        change, flux = 7, 9
        t = Continuous_change( change, flux )
        self.assertEqual( t.change, change )
        self.assertEqual( t.flux, flux )

        for (change, flux) in ( (None, 3), ('x', 7), ):
            with self.assertRaises(ValueError) as context:
                Continuous_change( change, flux )
            self.assertRegexpMatches( str(context.exception), 
                "Continuous_change.type_check\(\): .* is '.*' which is not an int or float" )

    def test_one_message_type(self):
    
        self.assertTrue( isclass_by_name( 'AdjustPopulationByContinuousModel',
            AdjustPopulationByContinuousModel ) )
        specie_id = 'x'
        adjustment = 7
        body = AdjustPopulationByContinuousModel.body( { specie_id:adjustment } )
        self.assertEqual( body.population_change[specie_id], adjustment )
