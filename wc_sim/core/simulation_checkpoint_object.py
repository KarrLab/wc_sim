""" A simulation object that produces periodic checkpoints

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-05-03
:Copyright: 2018, Karr Lab
:License: MIT
"""
import sys

from wc_sim.core.simulation_message import SimulationMessage
from wc_sim.core.simulation_object import ApplicationSimulationObject

class NextCheckpoint(SimulationMessage):
    "Schedule the next checkpoint"


# TODO(Arthur): a factory that generates self-clocking ApplicationSimulationObjects would be handy
class CheckpointSimulationObject(ApplicationSimulationObject):
    """ Create periodic checkpoints

    Attributes:
        checkpoint_period (:obj:`float`): interval between checkpoints, in simulated seconds
        checkpoint_dir (:obj:`str`): the directory in which to save checkpoints
    """

    def __init__(self, name, checkpoint_period, checkpoint_dir):
        self.checkpoint_period = checkpoint_period
        self.checkpoint_dir = checkpoint_dir
        super().__init__(name)

    def schedule_next_checkpoint(self):
        """ Schedule the next checkpoint in `self.checkpoint_period` simulated seconds
        """
        self.send_event(self.checkpoint_period, self, NextCheckpoint())

    def create_checkpoint(self):
        """ Create a checkpoint in the directory `self.checkpoint_dir`

        Derived classes must override this method and actually create a checkpoint
        """
        pass    # pragma: no cover     # must be overridden

    def send_initial_events(self):
        # create the initial checkpoint
        self.create_checkpoint()
        self.schedule_next_checkpoint()

    def handle_simulation_event(self, event):
        self.create_checkpoint()
        self.schedule_next_checkpoint()

    def get_state(self):
        return ''    # pragma: no cover     # must be overridden

    event_handlers = [(NextCheckpoint, handle_simulation_event)]

    # register the message type sent
    messages_sent = [NextCheckpoint]
