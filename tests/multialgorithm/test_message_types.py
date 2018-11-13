"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2017-03-26
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

import unittest

from wc_sim.multialgorithm import message_types
from wc_sim.multialgorithm.message_types import ContinuousChange

def make_message(msg_type, args):
    return msg_type(*args)

class Test(unittest.TestCase):

    def test_message_types(self):
        for message_type in message_types.ALL_MESSAGE_TYPES:
            n_args = len(message_type.__slots__)
            a_tuple = tuple(range(n_args))
            self.assertTrue(type(make_message(message_type, a_tuple)) is message_type)

    def test_continuous_change(self):
        change = 1
        flux = 2
        try:
            continuous_change = ContinuousChange(change, flux)
            self.assertEqual(continuous_change.change, change)
            self.assertEqual(continuous_change.flux, flux)
        except:
            self.fail("Exception not expected")

        with self.assertRaises(ValueError) as context:
            ContinuousChange(change, 'test')
        self.assertIn('which cannot be cast to a float', str(context.exception))

        with self.assertRaises(ValueError) as context:
            ContinuousChange(None, flux)
        self.assertIn('which cannot be cast to a float', str(context.exception))
