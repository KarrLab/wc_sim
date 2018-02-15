'''Compute distributed properties of multi-algorithmic whole-cell model simulations.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2017-03-15
:Copyright: 2016-2018, Karr Lab
:License: MIT
'''

from scipy.constants import Avogadro
from collections import defaultdict
import bisect
import builtins, math, sys

from wc_utils.config.core import ConfigManager
from wc_sim.core.simulation_object import Event, SimulationObject, SimulationObjectInterface
from wc_sim.multialgorithm.debug_logs import logs as debug_logs
from wc_sim.multialgorithm import message_types
from wc_sim.multialgorithm.config import paths as config_paths_multialgorithm
from .debug_logs import logs as debug_logs

config_multialgorithm = ConfigManager(config_paths_multialgorithm.core).get_config()['wc_sim']['multialgorithm']

# TODO(Arthur): cover after MVP wc_sim done
class AggregateDistributedProps(SimulationObject, SimulationObjectInterface):   # pragma: no cover
    '''Obtain and provide properties of a multi-algorithmic whole-cell model simulation
    that require retrieving distributed data.

    Computing an aggregate distributed property is driven by the data in a DistributedProperty. The
    current mechanism determines each distributed property periodically,
    at an interval provided by the DistributedProperty. It is obtained by requesting values
    from `SimulationObject`'s that store parts of the property.

    A `AggregateDistributedProps` manages multiple distributed properties.

    Event messages used to obtain properties:

    * `AggregateProperty`: aggregate a property. Events sent by an `AggregateDistributedProps` to itself,
       regulating a distributed property's periodicity. 
    * `GetHistoricalProperty`: request a property. A GetHistoricalProperty message sent to an `AggregateDistributedProps`
       requests the aggregate property. One sent to another object requests the local property.
    * `GiveProperty`: a response to a `GetHistoricalProperty` message, containing the property's value.

    Attributes:
        properties (:obj:`dict`): map: property.name -> property; properties managed by this
            `AggregateDistributedProps`
    '''

    # Sent message types
    SENT_MESSAGE_TYPES = [
        message_types.AggregateProperty,
        message_types.GiveProperty,
        message_types.GetHistoricalProperty]

    def __init__(self, name):
        '''Initialize an AggregateDistributedProps object

        For remaining args and exceptions, see `__init__()` documentation for
        `wc_sim.multialgorithm.SimulationObject`.
        '''
        SimulationObject.__init__(self, name)
        self.properties = {}

    def send_initial_events(self): pass

    def get_state(self):
        return 'object state to be provided'

    def add_property(self, distributed_property):
        '''Add a DistributedProperty 

        Args:
            distributed_property (:obj:`DistributedProperty`): a DistributedProperty that will
                be managed by this `AggregateDistributedProps` 
        '''
        self.properties[distributed_property.name] = distributed_property
        self.process_aggregate_property_event(distributed_property.name, initial_event=True)

    def get_property(self, event_message):
        '''Obtain an event's property_name

        Args:
            event_message (:obj:`Event`): an event message about a `DistributedProperty`
        '''
        property_name = event_message.event_body.property_name
        # if the property value isn't available, return an error
        if not property_name in self.properties:
            msg = "Error: unknown distributed property '{}'".format(property_name)
            debug_logs.get_log('wc.debug.file').debug(msg, sim_time=self.time)
            raise ValueError(msg)
        return property_name

    def process_aggregate_property_event(self, property_name, initial_event=False):
        '''Process an `AggregateProperty` simulation event

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
            event_body=message_types.AggregateProperty(property_name))

    def handle_aggregate_property_event(self, event_message):
        '''Process an event message `AggregateProperty`

        Args:
            event_message (:obj:`Event`): an event message about a `DistributedProperty`
        '''
        property_name = self.get_property(event_message)
        self.process_aggregate_property_event(property_name)        

    def handle_get_historical_property_event(self, event_message):
        '''Provide an aggregate property value to a requestor

        Args:
            event_message (:obj:`Event`): an event message about a `DistributedProperty`

        Raises:
            :obj:`ValueError`: if the `DistributedProperty` is not available at the requested time
        '''
        property_name = self.get_property(event_message)
        time = event_message.event_body.time
        try:
            value = self.properties[property_name].get_aggregate_value(time)
        except ValueError as e:
            msg = "Error: distributed property '{}' not available for time {}: {}".format(
                property_name, time, e)
            debug_logs.get_log('wc.debug.file').debug(msg, sim_time=self.time)
            raise ValueError(msg)

        self.send_event(0,
            event_message.sending_object,
            message_types.GiveProperty,
            event_body=message_types.GiveProperty(property_name, time, value))

    def handle_give_property_event(self, event_message):
        '''Record a property value from a contributor

        Args:
            event_message (:obj:`Event`): an event message about a `DistributedProperty`
        '''
        property_name = self.get_property(event_message)
        self.properties[property_name].record_value(
            event_message.sending_object,
            event_message.event_body.time,
            event_message.event_body.value)

    @classmethod
    def register_subclass_handlers(this_class):
        '''Register the handlers for event messages received by an `AggregateDistributedProps`
        '''
        SimulationObject.register_handlers(this_class,
            [
                # At any time instant, process message types in this order
                (message_types.AggregateProperty, this_class.handle_aggregate_property_event),
                (message_types.GiveProperty, this_class.handle_give_property_event),
                (message_types.GetHistoricalProperty, this_class.handle_get_historical_property_event),
                # todo: add a handler for GetCurrentProperty
            ])

    @classmethod
    def register_subclass_sent_messages(this_class):
        SimulationObject.register_sent_messages(this_class, this_class.SENT_MESSAGE_TYPES)


class DistributedProperty(object):  # pragma: no cover; # TODO(Arthur): cover after MVP wc_sim done
    '''A distributed property
    
    Maintain the state of an aggregate distributed property. The property is a single value,
    collected periodically from a set of contributing `SimulationObject`'s.

    Attributes:
        name (:obj:`str`): the property's name
        period (:obj:`float`): the periodicity at which the property is aggregated
        num_periods (:obj:`int`): the number of periods for which this property has been collected;
            used by `AggregateDistributedProps` to create event times that equal integral numbers
            of periods 
        contributors (:obj:`list` of `SimulationObject`): `SimulationObject`'s which must be queried
            to establish this property
        value_history (:obj:`dict`): time -> `dict`: `SimulationObject` -> value; history of distributed
            values for this property
        aggregate_value_history (:obj:`dict`): time -> value; history of aggregated values for this
            property
        aggregation_function (:obj:`method`): static method that aggregates values for this property
        kwargs (:obj:`dict`): arguments keyword arguments for `aggregation_function`
    '''
    def __init__(self, name, period, contributors, aggregation_function, **kwargs):
        self.name = name
        self.period = period
        self.contributors = contributors
        self.aggregation_function = aggregation_function
        self.num_periods = 0
        self.value_history = {}
        self.aggregate_value_history = {}
        self.kwargs = kwargs

    def request_values(self, the_aggregate_distributed_props, time):
        '''Request a `DistributedProperty`'s value from all objects that contribute to the value

        Args:
            the_aggregate_distributed_props (:obj:`SimulationObject`): the `AggregateDistributedProps`
                requesting the values
            time (:obj:`float`): the time of the values

        Raises:
            :obj:`ValueError`: if the `DistributedProperty` is not available at the requested time
        '''
        # time cannot be later than the simulation time when the requests will be handled
        request_time = the_aggregate_distributed_props.time + self.period
        if request_time < time:
            raise ValueError("request time ({}) later than simulation time of requests({})".format(
                time, request_time))
        self.value_history[time] = defaultdict(dict)
        epsilon = config_multialgorithm['epsilon']
        for contributor in self.contributors:
            the_aggregate_distributed_props.send_event(self.period-epsilon, contributor,
                message_types.GetHistoricalProperty,
                event_body=message_types.GetHistoricalProperty(self.name, time))

    def record_value(self, contributor, time, value):
        '''Record a contributed value of a distributed property

        Also aggregate the property if values have been received from all contributors at `time`.

        Args:
            contributor (:obj:`float`): the :obj:`SimulationObject` contributing the value
            time (:obj:`float`): the value's time
            value (:obj:`float`): the value from `contributor` at `time`
        '''
        '''
            todo: since self.value_history & self.aggregate_value_history grow without bound they
            must be garbage collected, at least those before GVT
        '''
        self.value_history[time][contributor] = value
        # received values from all contributor?
        if len(self.value_history[time].keys()) == len(self.contributors):
            self.aggregate_value_history[time] = self.aggregate_values(time)

    def aggregate_values(self, time):
        '''Aggregate the value of this distributed property at time `time`

        Use this distributed property's aggregation function (`aggregation_function`) to
        perform the aggregation.

        Args:
            time (:obj:`float`): the value's time
        '''
        if self.kwargs:
            return self.aggregation_function(self.value_history[time].values(), **self.kwargs)
        return self.aggregation_function(self.value_history[time].values())

    def get_aggregate_value(self, time):
        '''Obtain a `DistributedProperty`'s value from the local value history

        Args:
            time (:obj:`float`): the value's time

        Raises:
            :obj:`ValueError`: if the `DistributedProperty` is not available at `time`
        '''
        try:
            # todo: if permitted by the property, interpolate
            return self.aggregate_value_history[time]
        except KeyError as e:
            # report on context of key that cannot be found
            times = sorted(self.aggregate_value_history.keys())
            index = bisect.bisect_left(times, time)
            lesser = 'start of list'
            greater = 'end of list'
            if 0<=index-1:
                lesser = times[index-1]
            if index<len(times):
                greater = times[index]
            raise ValueError("get_aggregate_value: looking for time {}; "
                "nearest times are '{}' and '{}'".format(time, lesser, greater))


class DistributedPropertyFactory(object):   # pragma: no cover; # TODO(Arthur): cover after MVP wc_sim done

    @staticmethod
    def make_distributed_property(property_name, distributed_property_name, period,
        aggregation_function, **kwargs):
        '''Create a partially instantiated `DistributedProperty`

        Args:
            property_name (:obj:`str`): the property's name, an ALL CAPS constant in this module
            distributed_property_name (:obj:`str`): the distributed property's name
            period (:obj:`float`): the periodicity at which the property is aggregated
            aggregation_function (:obj:`method`): static method that aggregates values for this property
            kwargs (:obj:`dict`): arguments keyword arguments for `aggregation_function`

        Raises:
            ValueError: if the aggregation function is not callable or is unknown
        '''

        # create property_name constant in this module
        current_module = sys.modules[__name__]
        setattr(current_module, property_name.upper(), property_name.lower())

        # pick the aggregation function from available functions, use the provided function, or raise error
        if callable(aggregation_function):
            _aggregation_function = aggregation_function
        elif hasattr(builtins, aggregation_function):
            _aggregation_function = getattr(builtins, aggregation_function)
        elif hasattr(math, aggregation_function):
            _aggregation_function = getattr(math, aggregation_function)
        else:
            raise ValueError("Aggregation function '{}' not callable or in builtins or math".format(
                aggregation_function))
        if not callable(_aggregation_function):
            raise ValueError("Aggregation function '{}' not callable".format(aggregation_function))

        # create distributed_property
        distributed_property = DistributedProperty(distributed_property_name,
                    period,
                    None,
                    _aggregation_function,
                    **kwargs)
        # return partially instantiated distributed_property
        return distributed_property
