"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-02-06
:Copyright: 2018, Karr Lab
:License: MIT
"""
import unittest

from tests.core.mock_simulation_object import MockSimulationObjectInterface

class Example(MockSimulationObjectInterface):

    def test_method(self, value):
        self.test_case.assertEqual(value, self.kwargs['expected'])

    def send_initial_events(self): pass

    @classmethod
    def register_subclass_handlers(this_class): pass

    @classmethod
    def register_subclass_sent_messages(this_class): pass


class TestMockSimulationObjectInterface(unittest.TestCase):

    def test(self):
        t = Example('name', self, a=1, expected=2)
        t.test_method(2)

        with self.assertRaises(ValueError) as context:
            Example('name', 'string')
        self.assertIn("'test_case' should be a unittest.TestCase instance", str(context.exception))
        