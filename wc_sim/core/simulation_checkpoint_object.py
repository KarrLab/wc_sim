""" A simulation object that produces periodic checkpoints

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-05-03
:Copyright: 2018, Karr Lab
:License: MIT
"""
import sys

from wc_sim.core.simulation_message import SimulationMessage
from wc_sim.core.simulation_object import ApplicationSimulationObject
from wc_sim.core.errors import SimulatorError
from wc_sim.log.checkpoint import Checkpoint
from wc_sim.core.sim_metadata import SimulationMetadata


class NextCheckpoint(SimulationMessage):
    "Schedule the next checkpoint"


# TODO(Arthur): a factory that generates self-clocking ApplicationSimulationObjects would be handy
class AbstractCheckpointSimulationObject(ApplicationSimulationObject):
    """ Abstract class that creates periodic checkpoints

    Attributes:
        checkpoint_period (:obj:`float`): interval between checkpoints, in simulated seconds
    """

    def __init__(self, name, checkpoint_period):
        if checkpoint_period <= 0:
            raise SimulatorError("checkpoint period must be positive, but is {}".format(checkpoint_period))
        self.checkpoint_period = checkpoint_period
        super().__init__(name)

    def schedule_next_checkpoint(self):
        """ Schedule the next checkpoint in `self.checkpoint_period` simulated seconds
        """
        self.send_event(self.checkpoint_period, self, NextCheckpoint())

    def create_checkpoint(self):
        """ Create a checkpoint

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
        return ''    # pragma: no cover

    event_handlers = [(NextCheckpoint, handle_simulation_event)]

    # register the message type sent
    messages_sent = [NextCheckpoint]


# TODO(Arthur): `access_state_obj` objects should be derived from an ABC that enforces a `get_checkpoint_state(time)` method
class CheckpointSimulationObject(AbstractCheckpointSimulationObject):
    """ Create periodic checkpoints to files

    Uses the `wc_sim.log` checkpoint implementation

    Attributes:
        checkpoint_dir (:obj:`str`): the directory in which to save checkpoints
        metadata (:obj:`SimulationMetadata`): simulation run metadata
        access_state_obj (:obj:`object`): an object whose `get_state()` returns the simulation's state
    """

    def __init__(self, name, checkpoint_period, checkpoint_dir, metadata, access_state_obj):
        self.checkpoint_dir = checkpoint_dir
        self.metadata = metadata
        self.access_state_obj = access_state_obj
        super().__init__(name, checkpoint_period)

    def create_checkpoint(self):
        """ Create a checkpoint in the directory `self.checkpoint_dir`
        """
        # TODO(Arthur): include the random state
        Checkpoint.set_checkpoint(self.checkpoint_dir,
            Checkpoint(self.metadata, self.time, self.access_state_obj.get_checkpoint_state(self.time), None))
