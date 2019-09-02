import unittest
import re
from wc_sim.utils import get_species_and_compartment_from_name


class TestUtilities(unittest.TestCase):

    def test_utils(self):
        for test, correct in zip(['spc]',  'sp[c]', 'sp[[c]', 'sp[cxx ', 'sp[c]x]x'],
                                 [None,  ('sp', 'c'), None, None, None, ]):
            if correct is None:
                with self.assertRaises(ValueError) as context:
                    get_species_and_compartment_from_name(test)
                self.assertIn("species_id must have the form species_id[compartment_id], but is",
                              str(context.exception))

            else:
                (species, compartment) = get_species_and_compartment_from_name(test)
                self.assertEqual(correct, (species, compartment))
