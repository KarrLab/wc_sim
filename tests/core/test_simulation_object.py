"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2017-03-26
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

import unittest
import random
import six
import warnings
from builtins import super

from wc_sim.core.errors import SimulatorError
from wc_sim.core.simulation_object import (EventQueue, SimulationObject, ApplicationSimulationObject,
    ApplicationSimulationObjMeta)
from wc_sim.core.simulation_engine import SimulationEngine
from tests.core.some_message_types import InitMsg, Eg1, MsgWithAttrs, UnregisteredMsg
from tests.core.example_simulation_objects import (ALL_MESSAGE_TYPES, TEST_SIM_OBJ_STATE,
    ExampleSimulationObject)
from wc_utils.util.misc import most_qual_cls_name
from wc_utils.util.list import is_sorted
EVENT_HANDLERS = ApplicationSimulationObjMeta.EVENT_HANDLERS
MESSAGES_SENT = ApplicationSimulationObjMeta.MESSAGES_SENT


# TODO(Arthur): combine together example classes in tests.core.example_simulation_objects
class ImproperlyRegisteredSimulationObject(ApplicationSimulationObject):

    # register the event handler for each type of message received
    event_handlers = [(Eg1, 'handler')]

    # register the message types sent
    messages_sent = [InitMsg]

    def send_initial_events(self, *args): pass

    def get_state(self):
        return 'stateless object'

    def handler(self, event): pass


class TestEventQueue(unittest.TestCase):

    def setUp(self):
        self.event_queue = EventQueue()
        self.num_events = 5
        self.sender = sender = ExampleSimulationObject('sender')
        self.receiver = receiver = ExampleSimulationObject('receiver')
        for i in range(self.num_events):
            self.event_queue.schedule_event(i, i+1, sender, receiver, InitMsg())
        self.simulator = SimulationEngine()

    def test_next_event_time(self):
        empty_event_queue = EventQueue()
        self.assertEqual(float('inf'), empty_event_queue.next_event_time())
        self.assertEqual(1, self.event_queue.next_event_time())

    def test_render(self):
        self.assertIn('Empty', EventQueue().render())
        self.assertEqual(len(self.event_queue.render(as_list=True)), self.num_events+1)
        def get_event_times(eq_rendered_as_list):
            return [row[1] for row in eq_rendered_as_list[1:]]
        self.assertTrue(is_sorted(get_event_times(self.event_queue.render(as_list=True))))

        # test sorting
        test_eq = EventQueue()
        num_events = 10
        for i in range(num_events):
            test_eq.schedule_event(i, random.uniform(i, i+num_events), self.sender, self.receiver,
                MsgWithAttrs(2, 3))
        self.assertTrue(is_sorted(get_event_times(test_eq.render(as_list=True))))

        # test multiple message types
        test_eq = EventQueue()
        num_events = 20
        for i in range(num_events):
            msg = random.choice([InitMsg(), MsgWithAttrs(2, 3)])
            test_eq.schedule_event(i, i+1, self.sender, self.receiver, msg)
        self.assertEqual(len(test_eq.render(as_list=True)), num_events+1)
        self.assertTrue(is_sorted(get_event_times(test_eq.render(as_list=True))))
        for attr in MsgWithAttrs.__slots__:
            self.assertIn("\t{}:".format(attr), test_eq.render())


class ExampleSimulationObjectSubclass(ExampleSimulationObject):
    # register the message types sent
    messages_sent = [MsgWithAttrs]


class ASOwithMessages(ApplicationSimulationObject):
    # register the message types sent
    messages_sent = [MsgWithAttrs]


class FullASO(ApplicationSimulationObject):
    messages_sent = [MsgWithAttrs]
    def my_handler(self, event): pass
    event_handlers = [(Eg1, my_handler)]


class TestSimulationObjectRegistration(unittest.TestCase):

    def test_define_app_sim_obj(self):
        # test heritability
        self.assertEqual(ExampleSimulationObjectSubclass.messages_sent, [MsgWithAttrs])
        self.assertEqual(ExampleSimulationObjectSubclass.event_handlers, ExampleSimulationObject.event_handlers)

        full_ASO = FullASO('name')
        self.assertEqual(full_ASO.send_initial_events(), None)
        self.assertEqual(full_ASO.get_state(), '')

        with self.assertRaises(SimulatorError) as context:
            method_name = 'my_handler'
            class ASOwithBoth(ASOwithMessages):
                event_handlers = [(Eg1, method_name)]
        self.assertIn("ApplicationSimulationObject 'ASOwithBoth' definition must define '{}'.".format(method_name),
            str(context.exception))

        with self.assertRaises(SimulatorError) as context:
            class UnRegisteredSimulationObject(ApplicationSimulationObject):
                pass
        self.assertIn("ApplicationSimulationObject 'UnRegisteredSimulationObject' definition must provide",
            str(context.exception))

        with warnings.catch_warnings(record=True) as w:
            class UnRegisteredSimulationObject(ApplicationSimulationObject):
                messages_sent = [InitMsg]
            self.assertEqual(str(w[-1].message),
                "ApplicationSimulationObject 'UnRegisteredSimulationObject' definition does not provide '{}'.".format(
                    EVENT_HANDLERS))
            self.assertTrue(InitMsg in UnRegisteredSimulationObject.metadata.message_types_sent)
            self.assertFalse(UnRegisteredSimulationObject.metadata.event_handlers)

        with warnings.catch_warnings(record=True) as w:
            class UnRegisteredSimulationObject(ApplicationSimulationObject):
                event_handlers = [(InitMsg, 'handler')]
                def handler(self): pass
            self.assertEqual(str(w[-1].message),
                "ApplicationSimulationObject 'UnRegisteredSimulationObject' definition does not provide '{}'.".format(
                    MESSAGES_SENT))
            self.assertTrue(InitMsg in UnRegisteredSimulationObject.metadata.event_handlers)
            self.assertEqual(UnRegisteredSimulationObject.metadata.event_handlers[InitMsg],
                UnRegisteredSimulationObject.handler)

        warnings.simplefilter("ignore")
        class UnRegisteredSimulationObject(ApplicationSimulationObject):
            def handler(self): pass
            event_handlers = [(InitMsg, handler)]
        self.assertTrue(InitMsg in UnRegisteredSimulationObject.metadata.event_handlers)

        with self.assertRaises(SimulatorError) as context:
            class UnRegisteredSimulationObject(ApplicationSimulationObject):
                def handler(self): pass
                event_handlers = [(InitMsg, 3)]
        self.assertIn("ApplicationSimulationObject 'UnRegisteredSimulationObject' handler_name '3' must",
            str(context.exception))

        with self.assertRaises(SimulatorError) as context:
            # attempt to register the same message type multiple times
            class UnRegisteredSimulationObject(ApplicationSimulationObject):
                def handler(self): pass
                event_handlers = [(InitMsg, handler), (InitMsg, handler)]
        self.assertEqual(str(context.exception), "message type '{}' appears repeatedly".format(
            most_qual_cls_name(InitMsg)))

        with self.assertRaises(SimulatorError) as context:
            # attempt to register the same message type multiple times
            class UnRegisteredSimulationObject(ApplicationSimulationObject):
                not_callable = 'xxx'
                event_handlers = [(InitMsg, 'not_callable')]
        self.assertEqual(str(context.exception), "handler 'xxx' must be callable")


class TestSimulationObject(unittest.TestCase):

    def setUp(self):
        self.good_name = 'arthur'
        self.eso1 = ExampleSimulationObject(self.good_name)
        self.irso1 = ImproperlyRegisteredSimulationObject(self.good_name)
        self.simulator = SimulationEngine()
        self.o1 = ExampleSimulationObject('o1')
        self.o2 = ExampleSimulationObject('o2')
        self.simulator.add_objects([self.o1, self.o2])
        self.simulator.initialize()

    def test_attributes(self):
        self.assertEqual(self.good_name, self.eso1.name)
        self.assertEqual(0, self.eso1.time)
        self.assertEqual(0, self.eso1.num_events)
        self.assertEqual(None, self.eso1.simulator)

    def test_exceptions(self):
        with self.assertRaises(SimulatorError) as context:
            self.eso1.send_event_absolute(2, self.eso1, UnregisteredMsg())
        self.assertEqual(str(context.exception),
            "'{}' simulation objects not registered to send '{}' messages".format(
                most_qual_cls_name(self.eso1),
                UnregisteredMsg().__class__.__name__))

        with self.assertRaises(SimulatorError) as context:
            self.eso1.send_event_absolute(2, self.irso1, InitMsg())
        self.assertEqual(str(context.exception),
            "'{}' simulation objects not registered to receive '{}' messages".format(
                most_qual_cls_name(self.irso1),
                InitMsg().__class__.__name__))

    def test_get_receiving_priorities_dict(self):
        self.assertTrue(ExampleSimulationObject.metadata.event_handler_priorities[InitMsg] <
            ExampleSimulationObject.metadata.event_handler_priorities[Eg1])
        receiving_priorities = self.eso1.get_receiving_priorities_dict()
        self.assertTrue(receiving_priorities[InitMsg] < receiving_priorities[Eg1])

    def test_send_events(self):
        times=[2.0, 1.0, 0.5]
        # test both send_event methods
        for copy in [False, True]:
            for send_method in [self.o1.send_event, self.o1.send_event_absolute]:

                for t in times:
                    send_method(t, self.o2, Eg1(), copy=copy)

                tmp = sorted(times)
                while self.o2.event_queue.next_event_time() < float('inf'):
                    self.assertEqual(self.o2.event_queue.next_event_time(), tmp.pop(0))
                    self.o2.event_queue.next_events(self.o2)
                self.assertEqual(self.o2.event_queue.next_events(self.o2), [])

    def test_event_time_ties(self):
        self.o1.send_event(0, self.o2, Eg1())
        self.o1.send_event(2, self.o2, InitMsg())

        num=10
        self.o1.send_event(1, self.o2, InitMsg())
        for i in range(num):
            if random.choice([True, False]):
                self.o1.send_event(1, self.o2, Eg1())
            else:
                self.o1.send_event(1, self.o2, InitMsg())

        self.assertEqual(self.o2.event_queue.next_event_time(), 0)
        event_list = self.o2.event_queue.next_events(self.o2)
        self.assertEqual(event_list[0].event_time, 0)

        self.assertEqual(self.o2.event_queue.next_event_time(), 1)
        event_list = self.o2.event_queue.next_events(self.o2)
        # all InitMsg messages come before any Eg1 message,
        # and at least 1 InitMsg message exists
        expected_type = InitMsg
        switched = False
        for event in event_list:
            if not switched and event.message.__class__ == Eg1:
                expected_type = Eg1
            self.assertEqual(event.message.__class__, expected_type)

        self.assertEqual(self.o2.event_queue.next_event_time(), 2)

    def test_render_event_queue(self):
        rv = self.o1.render_event_queue()

        times=[2.0, 1.0, 0.5]
        for time in times:
            self.o1.send_event(time, self.o1, Eg1())
        rv = self.o1.render_event_queue()
        self.assertIn(self.o1.name, rv)
        # 1 extra row for the header
        self.assertEqual(len(rv.split('\n')), len(times)+1)
        for time in times:
            self.assertIn(str(time), rv)

    def test_event_exceptions(self):
        delay = -1.0
        with self.assertRaises(SimulatorError) as context:
            self.o1.send_event(delay, self.o2, Eg1())
        self.assertEqual(str(context.exception),
            "delay < 0 in send_event(): {}".format(str(delay)))

        event_time = -1
        with self.assertRaises(SimulatorError) as context:
            self.o1.send_event_absolute(event_time, self.o2, Eg1())
        six.assertRegex(self, str(context.exception),
            "event_time \(-1.*\) < current time \(0.*\) in send_event_absolute\(\)")

        eq = EventQueue()
        with self.assertRaises(SimulatorError) as context:
            eq.schedule_event(2, 1, None, None, '')
        self.assertEqual(str(context.exception),
            "receive_time < send_time in schedule_event(): {} < {}".format(1, 2))

        with self.assertRaises(SimulatorError) as context:
            eq.schedule_event(1, 2, None, None, 13)
        self.assertIn("message should be an instance of SimulationMessage but is a",
            str(context.exception))

        with self.assertRaises(SimulatorError) as context:
            self.o1.add(self.simulator)
        self.assertEqual(str(context.exception),
            "SimulationObject '{}' is already part of a simulator".format(self.o1.name))

    def test_misc(self):
        self.assertEqual(self.o1.get_state(), TEST_SIM_OBJ_STATE)
