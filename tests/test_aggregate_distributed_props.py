'''
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2017-03-15
:Copyright: 2017-2018, Karr Lab
:License: MIT
'''

import os
import unittest
import sys
import math
import statistics
from io import StringIO

from wc_sim.aggregate_distributed_props import (AggregateDistributedProps,
                                                               DistributedProperty, DistributedPropertyFactory)
from de_sim.simulation_object import SimulationObject
from de_sim.simulator import Simulator
from de_sim.event_message import EventMessage
from wc_sim import message_types
from wc_sim.config import core as config_core_multialgorithm

config_multialgorithm = config_core_multialgorithm.get_config()['wc_sim']['multialgorithm']


def sum_values_fn(values, **kwargs):
    return kwargs['constant']*sum(values)


class TestDistributedProperty(unittest.TestCase):
    '''Test data management, not dynamics, in `DistributedProperty`'''

    def setUp(self):
        self.period = 4
        self.CONSTANT = 2
        self.distributed_property = DistributedProperty(
            'name',
            self.period,
            [],
            sum_values_fn,
            **{'constant': self.CONSTANT})
        self.adp = AggregateDistributedProps('name')
        self.test_time = 3

    def test_all(self):
        # self.distributed_property.contributors == [], so this won't execute send_event()
        self.distributed_property.request_values(self.adp, self.test_time)

        NUM_VALUES = 3
        values = list(range(NUM_VALUES))
        contributors = ['cont_' + str(v) for v in values]
        self.distributed_property.contributors = contributors

        # test record_value()
        LOCAL_CONST = 4
        for index, contributor in enumerate(contributors[:-1]):
            self.distributed_property.record_value(
                contributor,
                self.test_time,
                LOCAL_CONST*values[index])
        self.assertEqual(set(self.distributed_property.value_history[self.test_time].values()),
                         set([LOCAL_CONST*v for v in range(NUM_VALUES-1)]))

        self.distributed_property.record_value(contributors[-1], self.test_time, LOCAL_CONST*values[-1])

        # test aggregate_values()
        self.assertEqual(self.distributed_property.aggregate_values(self.test_time),
                         self.CONSTANT*LOCAL_CONST*sum(values))

        earlier_time = self.test_time-1
        with self.assertRaises(ValueError) as context:
            self.distributed_property.get_aggregate_value(earlier_time)
        self.assertIn("looking for time {}; nearest times are 'start of list' and '{}'".format(
            earlier_time, self.test_time), str(context.exception))

        later_time = self.test_time+1
        with self.assertRaises(ValueError) as context:
            self.distributed_property.get_aggregate_value(later_time)
        self.assertIn("looking for time {}; nearest times are '{}' and 'end of list'".format(
            later_time, self.test_time), str(context.exception))


class PropertyProvider(SimulationObject):

    def __init__(self, name, test_property_hist):
        super().__init__(name)
        self.test_property_hist = test_property_hist

    def handle_get_historical_prop(self, event):
        # provide a property value to a requestor
        property_name = event.message.property_name
        time = event.message.time
        value = self.test_property_hist[property_name][time]
        self.send_event(0,
                        event.sending_object,
                        message_types.GiveProperty(property_name, time, value))

    event_handlers = [(message_types.GetHistoricalProperty, handle_get_historical_prop)]

    # register the message types sent
    messages_sent = [message_types.GiveProperty]


class GoGetProperty(EventMessage):
    "Self-clocking message for test property requestor"
    property_name: str
    time: float


class PropertyRequestor(SimulationObject):

    def __init__(self, name, property_name, period, aggregate_distributed_props, expected_history,
                 test_case):
        '''Init PropertyRequestor, a mock `SimulationObject` that asks for a property value and
            executes a unittest to evaluate whether the value is correct

        Args:
            name (str): name of this simulation object
            property_name (str): name of the property
            period (number): collection period for the property
            aggregate_distributed_props (obj `AggregateDistributedProps`): the `AggregateDistributedProps`
                that collects data for the property
            expected_history (obj `dict`): dict: time -> expected property value at time
            test_case (:obj:`unittest.TestCase`): reference to the TestCase that launches the simulation
        '''
        super().__init__(name)
        self.property_name = property_name
        self.period = period
        self.num_periods = 0
        self.aggregate_distributed_props = aggregate_distributed_props
        self.expected_history = expected_history
        self.test_case = test_case

    def init_before_run(self):
        self.send_event(self.period,
                        self,
                        GoGetProperty(self.property_name, self.period))

    def handle_give_property_event(self, event):
        '''PERFORM A UNIT TEST, evaluating whether the property value and time are correct'''
        time = event.message.time
        value = event.message.value
        self.test_case.assertEqual(value, self.expected_history[time])

    def handle_go_get_property_event(self, event):
        self.num_periods += 1
        measure_property_time = (self.num_periods+1)*self.period
        self.send_event_absolute(measure_property_time,
                                 self,
                                 GoGetProperty(event.message.property_name, measure_property_time))
        self.send_event(0,
                        self.aggregate_distributed_props,
                        message_types.GetHistoricalProperty(
                            event.message.property_name,
                            event.message.time))

    # register the event handler for each type of message received
    event_handlers = [
        # (message type, handler)
        (message_types.GiveProperty, handle_give_property_event),
        (GoGetProperty, handle_go_get_property_event)]

    # register the message types sent
    messages_sent = [message_types.GetHistoricalProperty, GoGetProperty]


def sum_values(values):
    return sum(values)


class TestAggregateDistributedProps(unittest.TestCase):

    def setUp(self):
        self.PERIOD = 10
        self.property_name = 'test_prop'
        self.aggregate_distributed_props = AggregateDistributedProps('aggregate_distributed_props')
        self.simulator = Simulator()

    def make_properties_and_providers(self, num_properties, num_providers, period, num_periods,
                                      value_hist_generator):
        '''Create the components of distributed property collection: `PropertyProvider`s,
        `DistributedProperty`s, and `PropertyRequestor`s
        '''

        # set up objects for testing
        expected_value_hist = {time: val for time, val in value_hist_generator()}

        property_providers = []
        for prop_num in range(num_properties):
            for prov_num in range(num_providers):
                property_providers.append(
                    PropertyProvider('property_{}_provider_{}'.format(prop_num, prov_num),
                                     {'property_{}'.format(prop_num): expected_value_hist}))

        properties = []
        requestors = []
        for prop_num in range(num_properties):
            property_name = 'property_{}'.format(prop_num)
            properties.append(DistributedProperty(
                property_name,
                period,
                property_providers[prop_num*num_providers:(prop_num+1)*num_providers],
                sum_values))

            # since the aggregation function is sum, expect distributed property to be
            # (number of providers)*value
            expected_requestor_value_history = {time: num_providers*val
                                                for time, val in expected_value_hist.items()}
            requestors.append(PropertyRequestor(
                'property_requestor_{}'.format(prop_num),
                property_name,
                period,
                self.aggregate_distributed_props,
                expected_requestor_value_history,
                self))

        return (properties, expected_value_hist, property_providers, requestors)

    @unittest.skip("aggregate_distributed_props not currently used")
    def test_aggregate_distributed_props(self):

        # test multiple concurrent properties over multiple time periods
        # these values are entirely arbitrary
        NUM_PROPERTIES = 3
        NUM_PROVIDERS = 4
        NUM_PERIODS = 6
        PERIOD = 5
        OFFSET = 3
        RATE = 2

        def value_hist_generator():
            # generates time,value tuples for a history
            # values are a linear function of time
            for i in range(NUM_PERIODS):
                yield PERIOD*i, OFFSET+RATE*i

        # create and register test SimulationObjects
        (props, expected_value_hist, providers, requestors) = self.make_properties_and_providers(
            NUM_PROPERTIES, NUM_PROVIDERS, PERIOD, NUM_PERIODS, value_hist_generator)

        self.simulator.add_objects(providers+requestors+[self.aggregate_distributed_props])

        for property in props:
            self.aggregate_distributed_props.add_property(property)

        # send initial events
        self.simulator.initialize()
        self.simulator.simulate((NUM_PERIODS-1)*PERIOD)

    def test_aggregate_distributed_props_errors1(self):
        '''
        send the simulator a bad message
        simulate & catch error
        '''
        self.simulator.add_object(self.aggregate_distributed_props)

        self.aggregate_distributed_props.send_event(
            0,
            self.aggregate_distributed_props,
            message_types.GetHistoricalProperty('no_such_property_name', 0))

        self.simulator.initialize()
        with self.assertRaises(ValueError) as context:
            self.simulator.simulate(10)
        self.assertIn('Error: unknown distributed property', str(context.exception))

    def test_aggregate_distributed_props_errors2(self):
        '''
        send the simulator a bad message
        simulate & catch error
        '''
        self.simulator.add_object(self.aggregate_distributed_props)

        prop_name = 'prop_name'
        period = 3
        distributed_property = DistributedProperty(
            prop_name,
            period,
            [],
            sum_values)
        self.aggregate_distributed_props.add_property(distributed_property)

        self.aggregate_distributed_props.send_event(
            2,
            self.aggregate_distributed_props,
            message_types.GetHistoricalProperty(prop_name, 1))

        self.simulator.initialize()
        with self.assertRaises(ValueError) as context:
            self.simulator.simulate(10)
        self.assertIn("Error: distributed property 'prop_name' not available for time",
                      str(context.exception))


test_lists = [
    [1],
    [33, 2, 1],
    [1, 2],
    [1, 20, 10, 100],
    [10, 1, 0, 1],
]


class TestDistributedPropertyFactory(unittest.TestCase):

    def test_distributed_property_factory(self):

        NO_SUCH_FN = 'no_such_fn'
        with self.assertRaises(ValueError) as context:
            DistributedPropertyFactory.make_distributed_property('F', 'f', 1, NO_SUCH_FN)
        self.assertIn("'{}' not callable or in builtins or math".format(NO_SUCH_FN),
                      str(context.exception))

        # builtin constant raises exception
        f = 'False'
        with self.assertRaises(ValueError) as context:
            DistributedPropertyFactory.make_distributed_property('F', 'f', 1, f)
        self.assertIn("'{}' not callable".format(f), str(context.exception))

        s = DistributedPropertyFactory.make_distributed_property('F', 'f', 1, 'sum')
        self.assertIs(s.aggregation_function, sum)

        fs = DistributedPropertyFactory.make_distributed_property('F', 'f', 1, 'fsum')
        self.assertIs(fs.aggregation_function, math.fsum)

        median_value = DistributedPropertyFactory.make_distributed_property(
            'MEDIAN',
            'median_value',
            2,    # seconds in period
            statistics.median)
        for test_list in test_lists:
            self.assertEqual(median_value.aggregation_function(test_list), statistics.median(test_list))
