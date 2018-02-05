""" Core discrete event simulation engine

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2016-06-01
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

import datetime

from wc_sim.core.simulation_object import SimulationObject
from wc_sim.core.errors import SimulatorError
from wc_sim.core.event import Event

# configure logging
from .debug_logs import logs as debug_logs

# TODO(Arthur): replace the O(n) iteration over simulation objects with a heap of them organized by next event time
# TODO(Arthur): measure engine's raw performance


class SimulationEngine(object):
    """ A simulation engine

    General-purpose simulation mechanisms, including the simulation scheduler.
    Architected as an OO simulation that could be parallelized.

    `SimulationEngine` contains and manipulates global simulation data.
    SimulationEngine registers all simulation objects types and all simulation objects.
    Following `simulate()` it runs the simulation, scheduling objects to execute events
    in non-decreasing time order; and generates debugging output.

    Attributes:
        time (:obj:`float`): the simulations's current time
        simulation_objects (:obj:`dict` of `SimulationObject`): all simulation objects, keyed by name
        __initialized (:obj:`bool`): whether the simulation has been initialized
    """

    def __init__(self):
        self.time = 0.0
        self.simulation_objects = {}
        self.log_with_time("SimulationEngine created")
        self.__initialized = False

    def register_object_types(self, simulation_object_types):
        """ Register simulation object types with the simulation

        The types of all simulation objects used by a simulation must be registered so that messages
        can be executed by vectoring to the right message handlers. This calls
        the methods `register_subclass_handlers()` and `register_subclass_sent_messages()` in each
        subclass of `SimulationObject` provided. A call to this method overwrites any
        previous registration.

        Args:
             simulation_object_types (:obj:`list`): the types of subclasses of `SimulationObject`s
                that will be used in the simulation
        """
        for simulation_object_type in simulation_object_types:
            simulation_object_type.register_subclass_handlers()
            simulation_object_type.register_subclass_sent_messages()

    def add_object(self, simulation_object):
        """ Add a simulation object instance to this simulation

        Args:
            simulation_object (:obj:`SimulationObject`): a simulation object instance that
                will be used by this simulation

        Raises:
            :obj:`SimulatorError`: if the simulation object's name is already in use
        """
        name = simulation_object.name
        if name in self.simulation_objects:
            raise SimulatorError("cannot add simulation object '{}', name already in use".format(name))
        simulation_object.add(self)
        self.simulation_objects[name] = simulation_object

    def delete_object(self, simulation_object):
        """ Delete a simulation object instance from this simulation

        Args:
            simulation_object (:obj:`SimulationObject`): a simulation object instance that is
                part of this simulation

        Raises:
            :obj:`SimulatorError`: if the simulation object is not part of this simulation
        """
        name = simulation_object.name
        if name not in self.simulation_objects:
            raise ValueError("cannot delete simulation object '{}', has not been added".format(name))
        simulation_object.delete()
        del self.simulation_objects[name]

    def load_objects(self, simulation_objects):
        """ Load simulation objects into the simulation

        Args:
            simulation_objects (:obj:`list` of `SimulationObject`): a list of simulation objects
        """
        for simulation_object in simulation_objects:
            self.add_object(simulation_object)

    def initialize(self):
        """ Initialize a simulation

        Call `send_initial_events()` in each simulation object that has been loaded.

        Raises:
            SimulatorError: if the simulation has already been initialized
        """
        if self.__initialized:
            raise SimulatorError('Simulation has already been initialized')
        for sim_obj in self.simulation_objects.values():
            sim_obj.send_initial_events()
        self.__initialized = True

    def reset(self):
        """ Reset this `SimulationEngine`

        Set simulation time to 0, delete all objects, and reset any prior initialization.
        """
        self.time = 0.0
        for simulation_object in list(self.simulation_objects.values()):
            self.delete_object(simulation_object)
        self.__initialized = False

    def message_queues(self):
        """ Return a string listing all message queues in the simulation

        Returns:
            :obj:`str`: a list of all message queues in the simulation and their messages
        """
        data = ['Event queues at {:6.3f}'.format(self.time)]
        for sim_obj in sorted(self.simulation_objects.values(), key=lambda sim_obj: sim_obj.name):
            data.append(sim_obj.name + ':')
            if sim_obj.event_queue.event_heap:
                data.append(Event.header())
                data.append(str(sim_obj.event_queue))
            else:
                data.append('Empty event queue')
            data.append('')
        return '\n'.join(data)

    def simulate(self, end_time, epsilon=None):
        """ Run the simulation

        Args:
            end_time (:obj:`float`): the time of the end of the simulation
            epsilon (:obj:`float`): small time interval used to control the order of near simultaneous
                events at different simulation objects; if provided, compare
                `epsilon` with `end_time` to ensure the ratio is not too large.

        Returns:
            :obj:`int`: the number of times a simulation object executes `_handle_event()`. This may
                be larger than the number of events sent, because simultaneous events are handled together.

        Raises:
            :obj:`SimulatorError`: if the ratio of `end_time` to `epsilon` is so large that `epsilon`
                is lost in roundoff error, or if the simulation has not been initialized
        """
        # TODO(Arthur): add optional logical termation condition(s)

        if not self.__initialized:
            raise SimulatorError("Simulation has not been initialized")

        # ratio of max simulation time to epsilon must not be so large that epsilon is lost
        # in roundoff error
        if not epsilon is None and not(epsilon + end_time > end_time):
            raise SimulatorError("epsilon ({:E}) plus end time ({:E}) must exceed end time".format(
                epsilon, end_time))

        # write header to a plot log
        # plot logging is controlled by configuration files pointed to by config_constants and by env vars
        plotting_logger = debug_logs.get_log('wc.plot.file')
        plotting_logger.debug('# {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()), sim_time=0)

        handle_event_invocations = 0
        self.log_with_time("Simulation to {} starting".format(end_time))
        while self.time <= end_time:

            # get the earliest next event in the simulation
            next_time = float('inf')
            self.log_with_time('Simulation Engine launching next object')
            for sim_obj in self.simulation_objects.values():

                if sim_obj.event_queue.next_event_time() < next_time:
                    next_time = sim_obj.event_queue.next_event_time()
                    next_sim_obj = sim_obj

            if float('inf') == next_time:
                self.log_with_time(" No events remain")
                break

            if end_time < next_time:
                self.log_with_time(" End time exceeded")
                break

            handle_event_invocations += 1

            self.time = next_time

            # assertion won't be violoated unless init message sent to negative time or
            # objects decrease their time.
            assert next_sim_obj.time <= next_time, ("Dispatching '{}', but find object time "
                "{} > event time {}.".format(next_sim_obj.name, next_sim_obj.time, next_time))

            # dispatch object that's ready to execute next event
            next_sim_obj.time = next_time
            next_sim_obj.__handle_event(next_sim_obj.event_queue.next_events(next_sim_obj))

        return handle_event_invocations

    def log_with_time(self, msg, local_call_depth=1):
        """Write a debug log message with the simulation time.
        """
        debug_logs.get_log('wc.debug.file').debug(msg, sim_time=self.time,
            local_call_depth=local_call_depth)
