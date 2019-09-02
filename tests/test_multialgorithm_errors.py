"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-01-26
:Copyright: 2018, Karr Lab
:License: MIT
"""

import unittest, os

from wc_sim.multialgorithm_errors import (Error,
    MultialgorithmError, SpeciesPopulationError, NegativePopulationError)


class TestMultialgorithmErrors(unittest.TestCase):

    def test_errors(self):
        msg = 'test msg'
        with self.assertRaises(Error) as context:
            raise Error(msg)
        self.assertEqual(msg, str(context.exception))

        with self.assertRaises(MultialgorithmError) as context:
            raise MultialgorithmError(msg)
        self.assertEqual(msg, str(context.exception))

        with self.assertRaises(SpeciesPopulationError) as context:
            raise SpeciesPopulationError(msg)
        self.assertEqual(msg, str(context.exception))

    def test_negative_population_error(self):
        npe1 = NegativePopulationError('method_name', 'species_name', 1, 2, 3)
        npe2 = NegativePopulationError('method_name', 'species_name', 1, 2, 3)
        npe3 = NegativePopulationError('method_name_3', 'species_name', 1, 2, 3)
        self.assertTrue(npe1 == npe1)
        self.assertTrue(npe1 == npe2)
        self.assertTrue(npe1 != npe3)

        with self.assertRaises(NegativePopulationError) as context:
            raise NegativePopulationError('method_name', 'species_name', 1, 3, 1)
        self.assertRegex(str(context.exception),
            "negative population predicted for .* with decline from .* to .* over .* time unit")
