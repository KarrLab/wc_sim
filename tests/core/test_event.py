'''
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-01-22
:Copyright: 2018, Karr Lab
:License: MIT
'''

import unittest
import six
from wc_sim.core.event import Event


class example(object):

    def __init__(self, name):
        self.name = name


class TestEvent(unittest.TestCase):

    def test_event_inequalities(self):
        e1 = Event(0, 1, object(), object(), 'event type', object())
        e2 = Event(0, 2, object(), object(), 'event type', object())
        self.assertTrue(e1 < e2)
        self.assertFalse(e1 > e2)
        self.assertTrue(e1 <= e1)
        self.assertTrue(e1 <= e2)
        self.assertTrue(e2 > e1)
        self.assertFalse(e2 < e1)
        self.assertTrue(e1 >= e1)
        self.assertTrue(e2 >= e1)

    def test_event_strings(self):
        e1 = Event(0, 1, example('sender'), example('receiver'), 'event type', object())
        self.assertIn('sender', str(e1))
        self.assertIn('Event time', Event.header())
