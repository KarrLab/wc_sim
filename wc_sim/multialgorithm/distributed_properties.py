'''Compute distributed properties of multi-algorithmic whole-cell model simulations.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-03-15
:Copyright: 2016-2017, Karr Lab
:License: MIT
'''

from scipy.constants import Avogadro
import sys
import six
from collections import defaultdict
import bisect

from wc_utils.util.misc import isclass_by_name as check_class
from wc_utils.config.core import ConfigManager
from wc_sim.core.simulation_object import Event, SimulationObject, SimulationObjectInterface
from wc_sim.multialgorithm.debug_logs import logs as debug_logs
from wc_sim.multialgorithm import message_types
from wc_sim.multialgorithm.config import paths as config_paths_multialgorithm
from .debug_logs import logs as debug_logs

config_multialgorithm = ConfigManager(config_paths_multialgorithm.core).get_config()['wc_sim']['multialgorithm']

class AggregateDistributedProps(SimulationObject, SimulationObjectInterface):
    '''Obtain and provide properties of a multi-algorithmic whole-cell model simulation
    that require retrieving distributed data.

    Computing an aggregate distributed property is driven by the data in a DistributedProperty. The
    current mechanism determines each distributed property periodically,
    at an interval provided by the DistributedProperty. It is obtained by requesting values
    from `SimulationObject`s that store parts of the property.

    A `AggregateDistributedProps` manages multiple distributed properties.

    Event messages used to obtain properties:
    * `AggregateProperty`: aggregate a property. Events sent by a `AggregateDistributedProps` to itself,
        regulating a distributed property's periodicity. 
    * `GetProperty`: request a property. A GetProperty message sent to a `AggregateDistributedProps`
        requests the aggregate property. One sent to another object requests the local property.
    * `GiveProperty`: a response to a `GetProperty` message, containing the property's value.

    Attributes:
        properties (:obj:`dict`): map: property.name -> property; properties managed by this
            `AggregateDistributedProps`
    '''

    # Sent message types
    SENT_MESSAGE_TYPES = [
        message_types.AggregateProperty,
        message_types.GiveProperty,
        message_types.GetProperty ]

    # At any time instant, process messages in this order
    MESSAGE_TYPES_BY_PRIORITY = SENT_MESSAGE_TYPES

    def __init__(self, name):
        '''Initialize an AggregateDistributedProps object

        For remaining args and exceptions, see `__init__()` documentation for
        `wc_sim.multialgorithm.SimulationObject`.
        '''
        SimulationObject.__init__(self, name)
        self.properties = {}

    def send_initial_events(self):
        pass

    def add_property(self, distributed_property):
        '''Add a DistributedProperty 

        Args:
            distributed_property (:obj:`DistributedProperty`): a DistributedProperty that will
                be managed by this `AggregateDistributedProps` 
        '''
        self.properties[distributed_property.name] = distributed_property
        self.handle_aggregate_property_event(distributed_property.name, initial_event=True)

    def handle_aggregate_property_event(self, property_name, initial_event=False):
        '''Handle a `message_types.AggregateProperty` simulation event

        Args:
            property_name (:obj:`str`): the property's name
            initial_event (:obj:`boolean`, optional): if `True`, initializing the property
        '''
        period = self.properties[property_name].period
        if not initial_event:
            self.properties[property_name].num_periods += 1
        num_periods = self.properties[property_name].num_periods
        # next event time calculated by a product of number of periods to avoid roundoff of an endless sum
        next_property_time = period*(num_periods+1)
        # obtain the distributed values
        self.properties[property_name].request_values(self, next_property_time)
        # send self-clocking AggregateProperty
        self.send_event(next_property_time-self.time,
            self,
            message_types.AggregateProperty,
            event_body=message_types.AggregateProperty.Body(property_name))
        
    def handle_event(self, event_list):
        '''Handle an AggregateDistributedProps simulation event.

        Process event messages for this AggregateDistributedProps.

        Args:
            event_list (:obj:`list` of :obj:`wc_sim.core.Event`): list of Events to process.
        '''
        # call handle_event() in class SimulationObject to perform generic tasks on the event list
        super(AggregateDistributedProps, self).handle_event(event_list)
        for event_message in event_list:

            # if the property value isn't available, return an error
            property_name = event_message.event_body.property_name
            if not property_name in self.properties:
                msg = "Error: unknown distributed property '{}'".format(property_name)
                print(msg, file=sys.stderr)
                debug_logs.get_log('wc.debug.file').debug(msg, sim_time=self.time)
                continue

            # switch/case on event message type
            if check_class(event_message.event_type, message_types.AggregateProperty):
                self.handle_aggregate_property_event(property_name)

            elif check_class(event_message.event_type, message_types.GetProperty):
                # provide an aggregate property value to a requestor
                time = event_message.event_body.time
                try:
                    value = self.properties[property_name].get_aggregate_value(time)
                except KeyError as e:
                    msg = "Error: distributed property '{}' not available for time {}: {}".format(
                        property_name, time, e)
                    print(msg, file=sys.stderr)
                    debug_logs.get_log('wc.debug.file').debug(msg, sim_time=self.time)
                    continue

                self.send_event(0,
                    event_message.sending_object,
                    message_types.GiveProperty,
                    event_body=message_types.GiveProperty.Body(property_name, time, value))

            elif check_class(event_message.event_type, message_types.GiveProperty):
                # record a property value from a contributor
                self.properties[property_name].record_value(
                    event_message.sending_object,
                    event_message.event_body.time,
                    event_message.event_body.value)

            else:
                assert False, "Shouldn't get here - {} should be covered"\
                    " in the if statement above".format(event_message.event_type)

class DistributedProperty(object):
    '''A distributed property
    
    Maintain the state of an aggregate distributed property. The property is a single value,
    collected periodically from a set of contributing `SimulationObject`s.

    Attributes:
        name (:obj:`str`): the property's name
        period (:obj:`float`): the periodicity at which the property is aggregated
        num_periods (:obj:`int`): the number of periods for which this property has been collected;
            used by `AggregateDistributedProps` to create event times that equal integral numbers
            of periods 
        contributors (:obj:`list` of `SimulationObject`): `SimulationObject`s which must be queried
            to establish this property
        value_history (:obj:`dict`): time -> `dict`: `SimulationObject` -> value; history of distributed
            values for this property
        aggregate_value_history (:obj:`dict`): time -> value; history of aggregated values for this
            property
        aggregation_function (:obj:`method`): static method that aggregates values for this property
        kwargs (:obj:`dict`): arguments keyword arguments for `aggregation_function`
    '''
    # todo: doc strings and error handling
    def __init__(self, name, period, contributors, aggregation_function, kwargs=None):
        self.name = name
        self.period = period
        self.num_periods = 0
        self.contributors = contributors
        self.value_history = {}
        self.aggregate_value_history = {}
        self.aggregation_function = aggregation_function
        self.kwargs = kwargs

    def request_values(self, this_simulation_object, time):
        # time cannot be later than the simulation time when the requests will be handled
        request_time = this_simulation_object.time + self.period
        if request_time < time:
            raise ValueError("request time ({}) later than simulation time of requests({})".format(
                time, request_time))
        self.value_history[time] = defaultdict(dict)
        epsilon = config_multialgorithm['epsilon']
        for contributor in self.contributors:
            this_simulation_object.send_event(self.period-epsilon, contributor,
                message_types.GetProperty,
                event_body=message_types.GetProperty.Body(self.name, time))

    def record_value(self, contributor, time, value):
        # todo: self.value_history & self.aggregate_value_history grow without bound and 
        # todo: should be garbage collected, at least before GVT
        self.value_history[time][contributor] = value
        # received values from all contributor?
        if len(self.value_history[time].keys()) == len(self.contributors):
            self.aggregate_value_history[time] = self.aggregate_values(time)
            # print("recorded aggregate_value at {}".format(time))

    def aggregate_values(self, time):
        if self.kwargs is None:
            return self.aggregation_function(self.value_history[time].values())
        return self.aggregation_function(self.value_history[time].values(), **self.kwargs)

    def get_aggregate_value(self, time):
        try:
            # todo: if permitted by the property, interpolate
            return self.aggregate_value_history[time]
        except KeyError as e:
            times = sorted(self.aggregate_value_history.keys())
            index = bisect.bisect_left(times, time)
            greater = lesser = 'not in list'
            if 0<=index-1:
                lesser = times[index-1]
            if index<len(times):
                greater = times[index]
            print(times[-100:])
            raise KeyError("KeyError in get_aggregate_value: looking for time {} "
                "nearest times are '{}' and '{}'".format(time, lesser, greater))


class MolecularMass(object):
    '''Uses DistributedProperty and AggregateDistributedProps to obtain the molecular mass of the cell.
    
    Simply sum the molecular masses of all cellular submodels and shared population objects.

    Approach:
        Create a molecular mass `DistributedProperty`
        Create a `AggregateDistributedProps` `SimulationObject`
        Add the `DistributedProperty` to the `AggregateDistributedProps`
    '''
    
    # todo: LocalPopulationStore must contain molecular weights; distribute the mass sum
    @staticmethod
    def sum_masses(masses):
        return sum(masses)

    @staticmethod
    def obtain_molecular_mass(self, period, population_stores,
        aggregate_distributed_props=None):
        '''Provide the molecular mass of a set of population stores
        '''
        distributed_property = DistributedProperty('obtain_molecular_mass',
            period,
            population_stores,
            MolecularMass.sum_masses,
            molecular_masses)
        if aggregate_distributed_props is None:
            aggregate_distributed_props = AggregateDistributedProps('molecular_mass')
        aggregate_distributed_props.add_property(distributed_property)
