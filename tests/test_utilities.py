import unittest
import re
from Sequential_WC_Simulator.multialgorithm.utilities import get_species_and_compartment_from_name

class TestUtilities(unittest.TestCase):

    def test_utilities(self):
        for test,correct in zip([ 'spc]',  'sp[c]' , 'sp[[c]' , 'sp[cxx ' , 'sp[c]x]x' ],
            [ None,  ('sp','c'), None , None , None , ]):
            if correct is None:
                with self.assertRaisesRegex(ValueError, 
                    re.escape("species_compartment_name must have the form "
                        "species_id[compartment_id], but is")+".*$"):
                    get_species_and_compartment_from_name(test)
                
            else:
                (species, compartment) = get_species_and_compartment_from_name( test )
                self.assertEqual( correct, (species, compartment) )
                