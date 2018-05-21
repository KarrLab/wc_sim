""" Template simulation objects

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-05-16
:Copyright: 2018, Karr Lab
:License: MIT
"""

from wc_sim.core.simulation_message import SimulationMessage
from wc_sim.core.simulation_object import ApplicationSimulationObject
from wc_sim.core.errors import SimulatorError


class NextEvent(SimulationMessage):
    "Schedule the next event"


class TemplateCheckpointSimulationObject(ApplicationSimulationObject):
    """ Template self-clocking ApplicationSimulationObject

    Events occur at time 0, `period`, `2 x period`, ...

    Attributes:
        period (:obj:`float`): interval between events, in simulated seconds
    """

    def __init__(self, name, period):
        if period <= 0:
            raise SimulatorError("period must be positive, but is {}".format(period))
        self.period = period
        super().__init__(name)

    def schedule_next_event(self):
        """ Schedule the next event in `self.period` simulated seconds
        """
        self.send_event(self.period, self, NextEvent())

    def handle_event(self):
        """ Handle the periodic event

        Derived classes must override this method and actually handle the event
        """
        pass    # pragma: no cover     # must be overridden

    def send_initial_events(self):
        # create the initial event
        self.handle_event()
        self.schedule_next_event()

    def handle_simulation_event(self, event):
        self.handle_event()
        self.schedule_next_event()

    def get_state(self):
        return ''    # pragma: no cover

    event_handlers = [(NextEvent, handle_simulation_event)]

    # register the message type sent
    messages_sent = [NextEvent]