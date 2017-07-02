import unittest

from wc_utils.util.misc import isclass_by_name
from wc_sim.multialgorithm.message_types import AdjustPopulationByContinuousSubmodel, ContinuousChange

class TestMessageTypes(unittest.TestCase):
    
    def test_type_check(self):
        change, flux = 7, 9
        t = ContinuousChange(change, flux)
        self.assertEqual(t.change, change)
        self.assertEqual(t.flux, flux)

        for (change, flux) in ((None, 3), ('x', 7),):
            with self.assertRaises(ValueError) as context:
                ContinuousChange(change, flux)
            self.assertRegexpMatches(str(context.exception), 
                "ContinuousChange.type_check\(\): .* is '.*' which is not an int or float")

    @unittest.skip("")
    def test_one_message_type(self):
        self.assertTrue(
            isclass_by_name('wc_sim.multialgorithm.message_types.AdjustPopulationByContinuousSubmodel',
            AdjustPopulationByContinuousSubmodel))
        specie_id = 'x'
        adjustment = 7
        body = AdjustPopulationByContinuousSubmodel.Body({ specie_id:adjustment })
        self.assertEqual(body.population_change[specie_id], adjustment)
