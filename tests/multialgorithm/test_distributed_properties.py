'''
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-03-15
:Copyright: 2017, Karr Lab
:License: MIT
'''

import os, unittest

from wc_utils.config.core import ConfigManager
from wc_utils.util.misc import isclass_by_name as check_class
from wc_sim.multialgorithm.distributed_properties import AggregateDistributedProps, DistributedProperty
from wc_sim.core.simulation_object import EventQueue, SimulationObject, SimulationObjectInterface
from wc_sim.core.simulation_engine import SimulationEngine
from wc_sim.multialgorithm import message_types
from wc_sim.multialgorithm.config import paths as config_paths_multialgorithm

config_multialgorithm = ConfigManager(config_paths_multialgorithm.core).get_config()['wc_sim']['multialgorithm']
epsilon = config_multialgorithm['epsilon']

def sum_values_test(values, **kwargs):
    return kwargs['constant']*sum(values)

class TestDistributedProperty(unittest.TestCase):
    '''This just tests data management in `DistributedProperty`
    '''

    def setUp(self):
        self.CONSTANT = 2
        self.distributed_property = DistributedProperty(
            'name',
            None,
            [],
            sum_values_test,
            {'constant':self.CONSTANT})
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
        for index,contributor in enumerate(contributors[:-1]):
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

class PropertyProvider(SimulationObject, SimulationObjectInterface):

    # Sent message types
    SENT_MESSAGE_TYPES = [message_types.GiveProperty]

    # At any time instant, process messages in this order
    MESSAGE_TYPES_BY_PRIORITY = [message_types.GetProperty]

    def __init__(self, name, test_property_hist):
        super(PropertyProvider, self).__init__(name)
        self.test_property_hist = test_property_hist

    def send_initial_events(self):
        pass

    def handle_event(self, event_list):
        super(PropertyProvider, self).handle_event(event_list)

        for event_message in event_list:

            # switch/case on event message type
            if check_class(event_message.event_type, message_types.GetProperty):
                # provide a property value to a requestor
                property_name = event_message.event_body.property_name
                time = event_message.event_body.time
                value = self.test_property_hist[property_name][time]
                self.send_event(0,
                    event_message.sending_object,
                    message_types.GiveProperty,
                    event_body=message_types.GiveProperty.Body(property_name, time, value))

            else:
                assert False, "Shouldn't get here - event_type '{}' should be covered"\
                    " in the if statement above".format(event_message.event_type)

class GoGetProperty(object):
    '''Self-clocking message for test property requestor.'''

    class Body(object):
        def __init__(self, property_name, time):
            self.property_name = property_name
            self.time = time

        def __str__(self):
            '''Return string representation of a GoGetProperty message body'''
            return "GoGetProperty: get {} @ {}".format(self.property_name, self.time)


class PropertyRequestor(SimulationObject, SimulationObjectInterface):

    # Sent message types
    SENT_MESSAGE_TYPES = [message_types.GetProperty, GoGetProperty]

    # At any time instant, process messages in this order
    MESSAGE_TYPES_BY_PRIORITY = [message_types.GiveProperty, GoGetProperty]

    def __init__(self, name, period, aggregate_distributed_props, expected_history, test_case):
        '''Init PropertyRequestor, a mock `SimulationObject` that can execute a unittest

        Args:
            test_case (:obj:`unittest.TestCase`): reference to the TestCase that launches the simulation
        '''
        super(PropertyRequestor, self).__init__(name)
        self.period = period
        self.num_periods = 0
        self.aggregate_distributed_props = aggregate_distributed_props
        self.expected_history = expected_history
        self.test_case = test_case

    def send_initial_events(self):
        pass

    def handle_event(self, event_list):
        '''Perform a unit test.'''
        super(PropertyRequestor, self).handle_event(event_list)

        for event_message in event_list:
            if check_class(event_message.event_type, message_types.GiveProperty):
                '''Evaluate whether the property value and time are right'''
                time = event_message.event_body.time
                value = event_message.event_body.value
                self.test_case.assertEqual(value, self.expected_history[time])

            elif check_class(event_message.event_type, GoGetProperty):
                self.num_periods += 1
                measure_property_time = (self.num_periods+1)*self.period
                self.send_event_absolute(measure_property_time-epsilon,
                    self,
                    GoGetProperty,
                    event_body=GoGetProperty.Body(
                        event_message.event_body.property_name,
                        measure_property_time))
                self.send_event(epsilon,
                    self.aggregate_distributed_props,
                    message_types.GetProperty,
                    event_body=message_types.GetProperty.Body(
                        event_message.event_body.property_name,
                        event_message.event_body.time))

            else:
                raise SpeciesPopulationError("Error: event_message.event_type '{}' should "\
                "be covered in the if statement above".format(event_message.event_type))

def sum_values(values):
    return sum(values)

class TestAggregateDistributedProps(unittest.TestCase):

    def setUp(self):
        self.PERIOD = 10
        self.property_name = 'test_prop'

    def make_properties_and_providers(self, num_properties, num_providers, period, num_periods,
        value_hist_generator, aggregate_distributed_props):
        # set up objects for testing
    
        expected_value_hist = {time:val for time,val in value_hist_generator()}

        property_providers = []
        for prop_num in range(num_properties):
            for prov_num in range(num_providers):
                property_providers.append(
                    PropertyProvider('property_{}_provider_{}'.format(prop_num, prov_num),
                        {'property_{}'.format(prop_num):expected_value_hist}))

        properties = []
        requestors = []
        for prop_num in range(num_properties):
            property_name = 'property_{}'.format(prop_num)
            properties.append(DistributedProperty(
                property_name,
                period,
                property_providers[prop_num*num_providers:(prop_num+1)*num_providers],
                sum_values))
            requestors.append(PropertyRequestor(
                'property_requestor_{}'.format(prop_num),
                period,
                aggregate_distributed_props,
                {time:num_providers*val for time,val in expected_value_hist.items()},
                self))

        return (properties, expected_value_hist, property_providers, requestors)

    def test_aggregate_distributed_props(self):
        '''
            create and register test SimulationObjects
            send initial events
            run simulation
        '''

        NUM_PROPERTIES = 4
        NUM_PERIODS = 20
        PERIOD = 7
        OFFSET = 3
        RATE = 2

        def value_hist_generator():
            # generates time,value tuples for a history
            for i in range(NUM_PERIODS):
                yield PERIOD*i,OFFSET+RATE*i 

        aggregate_distributed_props = AggregateDistributedProps('aggregate_distributed_props')

        (props, expected_value_hist, providers, requestors) = self.make_properties_and_providers(
            NUM_PROPERTIES, 5, PERIOD, NUM_PERIODS, value_hist_generator, aggregate_distributed_props)

        simulator = SimulationEngine()
        for sim_obj in providers+requestors+[aggregate_distributed_props]:
            simulator.add_object(sim_obj)

        # register message types
        for simulation_object_type in [PropertyProvider, PropertyRequestor, AggregateDistributedProps]:
            simulator.register_sent_message_types(simulation_object_type,
                simulation_object_type.SENT_MESSAGE_TYPES)
            simulator.register_receiver_priorities(simulation_object_type,
                simulation_object_type.MESSAGE_TYPES_BY_PRIORITY)

        for property in props:
            aggregate_distributed_props.add_property(property)

        # initial request events
        for prop_num in range(NUM_PROPERTIES):
            requestors[prop_num].send_event(PERIOD-epsilon,
                requestors[prop_num],
                GoGetProperty,
                event_body=GoGetProperty.Body(
                    props[prop_num].name,
                    PERIOD))

        simulator.simulate((NUM_PERIODS-1)*PERIOD, epsilon)

    # todo: create asserts to locate errors in the log and stderr
    def test_some_aggregate_distributed_props_errors(self):
        '''
        make a `AggregateDistributedProps`
        send it some bad messages
        simulate
        '''
        aggregate_distributed_props = AggregateDistributedProps('aggregate_distributed_props')

        simulator = SimulationEngine()
        simulator.add_object(aggregate_distributed_props)

        # register message types
        for simulation_object_type in [AggregateDistributedProps]:
            simulator.register_sent_message_types(simulation_object_type,
                simulation_object_type.SENT_MESSAGE_TYPES)
            simulator.register_receiver_priorities(simulation_object_type,
                simulation_object_type.MESSAGE_TYPES_BY_PRIORITY)

        # Error: unknown distributed property
        aggregate_distributed_props.send_event(
            0,
            aggregate_distributed_props,
            message_types.GetProperty,
            event_body=message_types.GetProperty.Body('no_such_property_name', 0))

        prop_name = 'prop_name'
        period = 3
        distributed_property = DistributedProperty(
            prop_name,
            period,
            [],
            sum_values)

        aggregate_distributed_props.add_property(distributed_property)
        # Error: distributed property 'prop_name' not available for time
        aggregate_distributed_props.send_event(
            2,
            aggregate_distributed_props,
            message_types.GetProperty,
            event_body=message_types.GetProperty.Body(prop_name, 1))

        simulator.simulate(10)

    def test_some_aggregate_distributed_props_errors(self):
        '''
        make a `AggregateDistributedProps`
        send it some bad messages
        simulate
        '''
        aggregate_distributed_props = AggregateDistributedProps('aggregate_distributed_props')
 
        simulator = SimulationEngine()
        simulator.add_object(aggregate_distributed_props)
 
        # register message types
        for simulation_object_type in [AggregateDistributedProps]:
            simulator.register_sent_message_types(simulation_object_type,
                simulation_object_type.SENT_MESSAGE_TYPES)
            simulator.register_receiver_priorities(simulation_object_type,
                simulation_object_type.MESSAGE_TYPES_BY_PRIORITY)
 
        # Error: unknown distributed property
        aggregate_distributed_props.send_event(
            0,
            aggregate_distributed_props,
            message_types.GetProperty,
            event_body=message_types.GetProperty.Body('no_such_property_name', 0))
 
        prop_name = 'prop_name'
        period = 3
        distributed_property = DistributedProperty(
            prop_name,
            period,
            [],
            sum_values)
 
        aggregate_distributed_props.add_property(distributed_property)
        # Error: distributed property 'prop_name' not available for time
        aggregate_distributed_props.send_event(
            2,
            aggregate_distributed_props,
            message_types.GetProperty,
            event_body=message_types.GetProperty.Body(prop_name, 1))
 
        simulator.simulate(10)

