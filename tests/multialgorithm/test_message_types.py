'''
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2017-03-26
:Copyright: 2016-2018, Karr Lab
:License: MIT
'''

import unittest

import wc_sim.multialgorithm.message_types as message_types

def make_message(msg_type, args):
    return msg_type(*args)

class Test(unittest.TestCase):

    def test_message_types(self):
        for message_type in message_types.ALL_MESSAGE_TYPES:
            n_args = len(message_type.__slots__)
            a_tuple = tuple(range(n_args))
            self.assertTrue(type(make_message(message_type, a_tuple)) is message_type)
