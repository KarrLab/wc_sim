"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2017-03-26
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

import sys
import unittest
import six
from builtins import super

from wc_sim.core.errors import SimulatorError
from wc_sim.core.simulation_object import EventQueue, SimulationObject, SimulationObjectInterface
from wc_sim.core.simulation_engine import SimulationEngine
from tests.core.some_message_types import InitMsg, Eg1
from wc_sim.core.shared_state_interface import SharedStateInterface
from wc_utils.util.misc import most_qual_cls_name

ALL_MESSAGE_TYPES = [InitMsg, Eg1]


# TODO(Arthur): consolidate in example_simulation_objects
class InactiveSimulationObject(SimulationObject, SimulationObjectInterface):

    def __init__(self):
        SimulationObject.__init__(self, 'inactive')

    def send_initial_events(self): pass

    def get_state(self): pass

    @classmethod
    def register_subclass_handlers(this_class): pass

    @classmethod
    def register_subclass_sent_messages(this_class): pass


class BasicExampleSimulationObject(SimulationObject, SimulationObjectInterface):

    def __init__(self, name):
        SimulationObject.__init__(self, name)
        self.num = 0

    def send_initial_events(self):
        self.send_event(1, self, InitMsg())

    def get_state(self):
        return "self.num: {}".format(self.num)

    @classmethod
    def register_subclass_handlers(this_class):
        SimulationObject.register_handlers(this_class,
            [(sim_msg_type, this_class.handle_event) for sim_msg_type in ALL_MESSAGE_TYPES])

    @classmethod
    def register_subclass_sent_messages(this_class):
        SimulationObject.register_sent_messages(this_class, ALL_MESSAGE_TYPES)


class ExampleSimulationObject(BasicExampleSimulationObject):

    def handle_event(self, event):
        self.send_event(2.0, self, InitMsg())

    @classmethod
    def register_subclass_handlers(this_class):
        super().register_subclass_handlers()

    @classmethod
    def register_subclass_sent_messages(this_class):
        super().register_subclass_sent_messages()


class InteractingSimulationObject(BasicExampleSimulationObject):

    def handle_event(self, event):
        self.num += 1
        # send an event to each InteractingSimulationObject
        for obj in self.simulator.simulation_objects.values():
            self.send_event(1, obj, InitMsg())

    @classmethod
    def register_subclass_handlers(this_class):
        super().register_subclass_handlers()

    @classmethod
    def register_subclass_sent_messages(this_class):
        super().register_subclass_sent_messages()


class CyclicalMessagesSimulationObject(SimulationObject, SimulationObjectInterface):
    """ Send events around a cycle of objects
    """

    def __init__(self, name, obj_num, cycle_size, test_case):
        SimulationObject.__init__(self, name)
        self.obj_num = obj_num
        self.cycle_size = cycle_size
        self.test_case = test_case
        self.num_msgs = 0

    def next_obj(self):
        next = (self.obj_num+1) % self.cycle_size
        return self.simulator.simulation_objects[obj_name(next)]

    def send_initial_events(self):
        # send event to next InteractingSimulationObject
        self.send_event(1, self.next_obj(), InitMsg())

    def get_state(self):
        return 'object state to be provided'

    def handle_event(self, event):
        self.num_msgs += 1
        self.test_case.assertEqual(self.time, float(self.num_msgs))
        # send event to next InteractingSimulationObject
        self.send_event(1, self.next_obj(), InitMsg())

    @classmethod
    def register_subclass_handlers(this_class):
        SimulationObject.register_handlers(this_class,
            [(sim_msg_type, this_class.handle_event) for sim_msg_type in ALL_MESSAGE_TYPES])

    @classmethod
    def register_subclass_sent_messages(this_class):
        SimulationObject.register_sent_messages(this_class, ALL_MESSAGE_TYPES)


NAME_PREFIX = 'sim_obj'

def obj_name(i):
    return '{}_{}'.format(NAME_PREFIX, i)


class TestSimulationEngine(unittest.TestCase):

    def setUp(self):
        # create simulator and register simulation object types
        self.simulator = SimulationEngine()
        self.simulator.register_object_types([ExampleSimulationObject, InteractingSimulationObject,
            CyclicalMessagesSimulationObject])
        self.simulator.reset()

    def test_one_object_simulation(self):
        obj = ExampleSimulationObject(obj_name(1))
        self.simulator.add_object(obj)
        self.simulator.initialize()
        self.assertEqual(self.simulator.simulate(5.0), 3)

    def test_one_object_simulation_neg_endtime(self):
        obj = ExampleSimulationObject(obj_name(1))
        self.simulator.add_object(obj)
        self.simulator.initialize()
        self.assertEqual(self.simulator.simulate(-1), 0)

    def test_simulation_engine_exceptions(self):
        obj = ExampleSimulationObject(obj_name(1))
        with self.assertRaises(ValueError) as context:
            self.simulator.delete_object(obj)
        self.assertIn("cannot delete simulation object '{}'".format(obj.name), str(context.exception))

        self.simulator.add_object(obj)
        with self.assertRaises(SimulatorError) as context:
            self.simulator.simulate(5.0)
        self.assertIn('Simulation has not been initialized', str(context.exception))

        with self.assertRaises(SimulatorError) as context:
            self.simulator.add_object(obj)
        self.assertIn("cannot add simulation object '{}'".format(obj.name), str(context.exception))

        self.simulator.delete_object(obj)
        try:
            self.simulator.add_object(obj)
        except:
            self.fail('should be able to add object after delete')

        self.simulator.initialize()
        event_queue = obj.event_queue
        event_queue.schedule_event(-1, -1, obj, obj, InitMsg())
        with self.assertRaises(AssertionError) as context:
            self.simulator.simulate(5.0)
        self.assertIn('find object time', str(context.exception))
        self.assertIn('> event time', str(context.exception))

        with self.assertRaises(SimulatorError) as context:
            self.simulator.initialize()
        self.assertEqual('Simulation has already been initialized', str(context.exception))

        self.simulator.reset()
        self.simulator.initialize()
        with self.assertRaises(SimulatorError) as context:
            self.simulator.simulate(5.0, epsilon=0)
        six.assertRegex(self, str(context.exception), "epsilon (.*) plus end time (.*) must exceed end time")

    def test_simulation_end(self):
        self.simulator.add_object(InactiveSimulationObject())
        self.simulator.initialize()
        # log "No events remain"
        self.simulator.simulate(5.0)

    def test_multi_object_simulation_and_reset(self):
        for i in range(1, 4):
            obj = ExampleSimulationObject(obj_name(i))
            self.simulator.add_object(obj)
        self.simulator.initialize()
        self.assertEqual(self.simulator.simulate(5.0), 9)

        self.simulator.reset()
        self.assertEqual(len(self.simulator.simulation_objects), 0)

    def test_multi_interacting_object_simulation(self):
        sim_objects = [InteractingSimulationObject(obj_name(i)) for i in range(1, 3)]
        self.simulator.load_objects(sim_objects)
        self.simulator.initialize()
        self.assertEqual(self.simulator.simulate(2.5), 4)

    def make_cyclical_messaging_network_sim(self, num_objs):
        # make simulation with cyclical messaging network
        sim_objects = [CyclicalMessagesSimulationObject(obj_name(i), i, num_objs, self)
            for i in range(num_objs)]
        self.simulator.load_objects(sim_objects)

    def test_cyclical_messaging_network(self):
        # test event times at simulation objects; this test should succeed with any
        # natural number for num_objs and any non-negative value of sim_duration
        self.make_cyclical_messaging_network_sim(10)
        self.simulator.initialize()
        self.simulator.simulate(20)

    def test_message_queues(self):
        self.make_cyclical_messaging_network_sim(4)
        self.simulator.add_object(InactiveSimulationObject())
        self.simulator.initialize()
        queues = self.simulator.message_queues()
        for sim_obj_name in self.simulator.simulation_objects:
            self.assertIn(sim_obj_name, queues)



class ExampleSharedStateObject(SharedStateInterface):

    def __init__(self, name, state):
        self.name = name
        self.state = state

    def get_name(self):
        return self.name

    def get_shared_state(self, time):
        return str(self.state)


class TestSimulationEngineLogging(unittest.TestCase):

    def setUp(self):
        self.example_shared_state_obj_name = 'example_shared_state_obj_name'
        self.example_shared_state_obj_state = [2, 'hi']
        self.example_shared_state_objs = \
            [ExampleSharedStateObject(self.example_shared_state_obj_name, self.example_shared_state_obj_state)]
        self.simulator = SimulationEngine(shared_state=self.example_shared_state_objs, debug_log=True)
        self.simulator.register_object_types([InteractingSimulationObject])
        self.simulator.reset()

    # test logging with InteractingSimulationObject and a SharedStateInterface object
    def test_logging(self):
        num_sim_obj = 2
        sim_objects = [InteractingSimulationObject(obj_name(i)) for i in range(1, num_sim_obj+1)]
        self.simulator.load_objects(sim_objects)
        self.simulator.initialize()
        self.assertEqual(self.simulator.simulate(2.5), 4)
        self.assertIn(self.example_shared_state_obj_name, self.simulator.log_simulation_state())
        self.assertIn(str(self.example_shared_state_obj_state), self.simulator.log_simulation_state())
        self.assertIn(InteractingSimulationObject.__name__, self.simulator.log_simulation_state())
        for i in range(1, num_sim_obj+1):
            self.assertIn(obj_name(i), self.simulator.log_simulation_state())
