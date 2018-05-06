"""
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2018-05-03
:Copyright: 2017-2018, Karr Lab
:License: MIT
"""

import unittest

from wc_sim.core.simulation_engine import SimulationEngine
from wc_sim.core.simulation_checkpoint_object import CheckpointSimulationObject
from wc_sim.core.simulation_message import SimulationMessage
from wc_sim.core.simulation_object import ApplicationSimulationObject


class PeriodicCheckpointSimuObj(CheckpointSimulationObject):
    """ Test checkpointing by simplistically saving them to a list
    """

    def __init__(self, name, checkpoint_period, checkpoint_dir, simulation_state,
        shared_checkpoints):
        self.simulation_state = simulation_state
        self.shared_checkpoints = shared_checkpoints
        super().__init__(name, checkpoint_period, checkpoint_dir)

    def create_checkpoint(self):
        self.shared_checkpoints.append((self.time, self.simulation_state.get()))


class MessageSentToSelf(SimulationMessage):
    "A message that's sent to self"


class PeriodicLinearUpdatingSimuObj(ApplicationSimulationObject):
    """ Sets a shared value to a linear function of the simulation time
    """

    def __init__(self, name, delay, simulation_state, a, b):
        self.delay = delay
        self.simulation_state = simulation_state
        self.a = a
        self.b = b
        super().__init__(name)

    def send_initial_events(self):
        self.send_event(self.delay, self, MessageSentToSelf())

    def handle_simulation_event(self, event):
        self.simulation_state.set(self.a*self.time + self.b)
        self.send_event(self.delay, self, MessageSentToSelf())

    def get_state(self): return ''

    # register the event handler and message type sent
    event_handlers = [(MessageSentToSelf, handle_simulation_event)]
    messages_sent = [MessageSentToSelf]


class SharedValue(object):

    def __init__(self, init_val):
        self.value = init_val

    def set(self, val):
        self.value = val

    def get(self):
        return self.value


class TestCheckpointSimulationObject(unittest.TestCase):

    def test_checkpoint_simulation_object(self):
        '''
        Run a simulation with a subclass of CheckpointSimulationObject and another object.
        Take checkpoints and test them.
        '''
        self.simulator = SimulationEngine()
        a = 4
        b = 3
        state = SharedValue(b)
        update_period = 3
        updating_obj = PeriodicLinearUpdatingSimuObj('updating_obj', update_period, state, a, b)

        checkpoints = []
        checkpoint_period = 10
        checkpointing_obj = PeriodicCheckpointSimuObj('checkpointing_obj', checkpoint_period, None,
            state, checkpoints)
        self.simulator.add_objects([updating_obj, checkpointing_obj])
        self.simulator.initialize()
        run_time = 100
        self.simulator.run(run_time)
        checkpointing_obj.create_checkpoint()
        for i in range(1 + int(run_time/checkpoint_period)):
            time, value = checkpoints[i]
            self.assertEqual(time, i*checkpoint_period)
            # updating_obj sets the shared value to a*time + b, at the instants 0, update_period, 2*update_period, ...
            # checkpointing_obj samples the value at times unsynchronized with updating_obj
            # therefore, for 0<a, the sampled values are at most a*update_period less than the line a*time + b
            linear_prediction = a*checkpoint_period*i + b
            self.assertTrue(linear_prediction - a*update_period <= value <= linear_prediction)
