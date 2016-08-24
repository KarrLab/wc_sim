#!/usr/bin/env python

import unittest
from Sequential_WC_Simulator.multialgorithm.utilities import get_species_and_compartment_from_name
import pprint 
pp = pprint.PrettyPrinter(indent=4)

class TestUtilities(unittest.TestCase):

    @unittest.skip("not working in python 3.5")
    def test_utilities(self):
        for test,correct in zip([ 'spc]',  'sp[c]' , 'sp[[c]' , 'sp[cxx ' , 'sp[c]x]x' ],
            [ None,  ('sp','c'), None , None , None , ]):
            print(test,correct)
            if correct is None:
                with self.assertRaisesRegex(ValueError, 
                    'species_compartment_name must have the form species_id[compartment_id], but is.*$'):
                    get_species_and_compartment_from_name(test)
                '''
                pp.pprint(context.exception)
                print(dir(context.exception))
                print(dir(context.exception.args))
                self.assertIn( "species_compartment_name must have the form species_id[compartment_id], but is", 
                    context.exception )
                '''
                
            else:
                (species, compartment) = get_species_and_compartment_from_name( test )
                self.assertEqual( correct, (species, compartment) )
                
if __name__ == '__main__':
    try:
        unittest.main()
    except KeyboardInterrupt:
        pass

