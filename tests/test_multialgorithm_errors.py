"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-01-26
:Copyright: 2018, Karr Lab
:License: MIT
"""

import os
import unittest

from wc_sim.multialgorithm_errors import (Error, MultialgorithmError, SpeciesPopulationError,
                                          NegativePopulationError, FrozenSimulationError)


class TestMultialgorithmErrors(unittest.TestCase):

    def test_errors(self):
        msg = 'test msg'

        with self.assertRaisesRegexp(Error, msg):
            raise Error(msg)

        with self.assertRaisesRegexp(MultialgorithmError, msg):
            raise MultialgorithmError(msg)

        with self.assertRaisesRegexp(SpeciesPopulationError, msg):
            raise SpeciesPopulationError(msg)

        with self.assertRaisesRegexp(FrozenSimulationError, msg):
            raise FrozenSimulationError(msg)


class TestNegativePopulationError(unittest.TestCase):

    def test(self):
        npe1 = NegativePopulationError('method_name', 'species_name', 1, 2, 3)
        npe2 = NegativePopulationError('method_name', 'species_name', 1, 2, 3)
        npe3 = NegativePopulationError('method_name_3', 'species_name', 1, 2, 3)
        npe4 = NegativePopulationError('method_name', 'species_name', 1, 5, 3)
        self.assertEqual(npe1, npe1)
        self.assertEqual(hash(npe1), hash(npe1))
        self.assertEqual(npe1, npe2)
        self.assertEqual(hash(npe1), hash(npe2))
        self.assertNotEqual(npe1, npe3)
        self.assertNotEqual(hash(npe1), hash(npe3))
        self.assertNotEqual(npe1, 'string')

        expected = "negative population predicted for .* with decline from .* to .* over .* time unit"
        with self.assertRaisesRegexp(NegativePopulationError, expected):
            raise NegativePopulationError('method_name', 'species_name', 1, 3, 1)

    def test___str__(self):
        npe = str(NegativePopulationError('method_name', 'species_name', 1, -2))
        self.assertIn('decline from 1', npe)
        npe = str(NegativePopulationError('method_name', 'species_name', 1, -2, 5))
        self.assertIn('over 5 time units', npe)
        npe = str(NegativePopulationError('method_name', 'species_name', 1, -2, 1))
        self.assertIn('over 1 time unit', npe)
