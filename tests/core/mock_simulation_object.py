""" An interface for a mock simulation object that can evaluate unit test assertions
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-02-06
:Copyright: 2018, Karr Lab
:License: MIT
"""

import six, abc
from builtins import super
from unittest import TestCase

from wc_sim.core.simulation_object import SimulationObject, SimulationObjectInterface


# TODO(Arthur): get coverage reports of the test_mock_simulation_object.py test of this module
# the obvious ways using 'coverage' or 'pytest' do not work
class MockSimulationObject(SimulationObject):
    """ An object to help test simulation objects
    """

    def __init__(self, name, test_case, **kwargs):
        """ Init a MockSimulationObject that can unittest a `SimulationObject`s behavior

        Use `self.test_case` and `self.kwargs` to evaluate unit tests

        Args:
            name (:obj:`str`): name for the `SimulationObject`
            test_case (:obj:`TestCase`): reference to the `TestCase` that launches the simulation
            kwargs (:obj:`dict`): arguments used by a test case
        """
        if not isinstance(test_case, TestCase):
            raise ValueError("'test_case' should be a unittest.TestCase instance, but it is a {}".format(
                type(test_case)))
        (self.test_case, self.kwargs) = (test_case, kwargs)
        super().__init__(name)

@six.add_metaclass(abc.ABCMeta)
class MockSimulationObjectInterface(MockSimulationObject, SimulationObjectInterface):  # pragma: no cover
    """ An ABC to help test simulation objects
    """

    @abc.abstractmethod
    def send_initial_events(self): pass

    @abc.abstractmethod
    def get_state(self): pass

    @classmethod
    @abc.abstractmethod
    def register_subclass_handlers(this_class): pass

    @classmethod
    @abc.abstractmethod
    def register_subclass_sent_messages(this_class): pass
