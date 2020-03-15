"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-01-26
:Copyright: 2018, Karr Lab
:License: MIT
"""

import os
import unittest

from wc_sim.multialgorithm_errors import (DynamicFrozenSimulationError, DynamicMultialgorithmError,
                                          DynamicNegativePopulationError, DynamicSpeciesPopulationError,
                                          Error, MultialgorithmError, SpeciesPopulationError)


class TestMultialgorithmErrors(unittest.TestCase):

    def test_errors(self):
        msg = 'test msg'

        with self.assertRaisesRegexp(Error, msg):
            raise Error(msg)

        with self.assertRaisesRegexp(MultialgorithmError, msg):
            raise MultialgorithmError(msg)

        with self.assertRaisesRegexp(SpeciesPopulationError, msg):
            raise SpeciesPopulationError(msg)

        def expected(time, msg):
            return f"{time}: {msg}"

        time = 3.5
        with self.assertRaisesRegexp(DynamicFrozenSimulationError, expected(time, msg)):
            raise DynamicFrozenSimulationError(time, msg)

        with self.assertRaisesRegexp(DynamicMultialgorithmError, expected(time, msg)):
            raise DynamicMultialgorithmError(time, msg)

        with self.assertRaisesRegexp(DynamicSpeciesPopulationError, expected(time, msg)):
            raise DynamicSpeciesPopulationError(time, msg)

class TestNegativePopulationError(unittest.TestCase):

    def test(self):
        time = 10.0
        npe1 = DynamicNegativePopulationError(time, 'method_name', 'species_name', 1, 2, 3)
        npe2 = DynamicNegativePopulationError(time, 'method_name', 'species_name', 1, 2, 3)
        npe3 = DynamicNegativePopulationError(time, 'method_name_3', 'species_name', 1, 2, 3)
        npe4 = DynamicNegativePopulationError(time, 'method_name', 'species_name', 1, 5, 3)
        self.assertEqual(npe1, npe1)
        self.assertEqual(hash(npe1), hash(npe1))
        self.assertEqual(npe1, npe2)
        self.assertEqual(hash(npe1), hash(npe2))
        self.assertNotEqual(npe1, npe3)
        self.assertNotEqual(hash(npe1), hash(npe3))
        self.assertNotEqual(npe1, 'string')

        expected = "negative population predicted for .* with decline from .* to .* over .* time unit"
        with self.assertRaisesRegexp(DynamicNegativePopulationError, expected):
            raise DynamicNegativePopulationError(time, 'method_name', 'species_name', 1, 3, 1)

    def test___str__(self):
        time = 12.125
        npe = str(DynamicNegativePopulationError(time, 'method_name', 'species_name', 1, -2))
        self.assertIn('decline from 1', npe)
        npe = str(DynamicNegativePopulationError(time, 'method_name', 'species_name', 1, -2, 5))
        self.assertIn('over 5 time units', npe)
        npe = str(DynamicNegativePopulationError(time, 'method_name', 'species_name', 1, -2, 1))
        self.assertIn('over 1 time unit', npe)
