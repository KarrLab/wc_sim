""" Checkpointing log

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2017-08-30
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

import math
import numpy
import os
import pickle
import re

# .. todo :: use hdf rather than pickle


class Checkpoint(object):
    """ Represents a simulation checkpoint

    Attributes:
        metadata (:obj:`wc.sim.metadata.Metadata`):
        time (:obj:`float`):
        state (:obj:`object`): simulated state
        random_state (:obj:`object`): state of random number generator
    """

    def __init__(self, metadata, time, state, random_state):
        self.metadata = metadata
        self.time = time
        self.state = state
        self.random_state = random_state

    @staticmethod
    def set_checkpoint(dirname, checkpoint):
        """ Save a checkpoint to the directory `dirname`.

        Args:
            checkpoint (:obj:`Checkpoint`): checkpoint
            dirname (:obj:`str`): directory to read/write checkpoint data
        """

        file_name = Checkpoint.get_file_name(dirname, checkpoint.time)

        with open(file_name, 'wb') as file:
            pickle.dump(checkpoint, file)

    @staticmethod
    def get_checkpoint(dirname, time=None):
        """ Get most recent checkpoint before time `time` from the checkpoint directory `dirname`.
        For example if `time` = 1.99 s and there are checkpoints at 1.0 s, 1.5 s, and 2.0 s,
        return the checkpoint from 1.5 s.

        Args:
            dirname (:obj:`str`): directory to read/write checkpoint data
            time (:obj:`float`): time in seconds of desired checkpoint

        Returns:
            :obj:`Checkpoint`: most recent checkpoint before time `time`
        """

        # get list of checkpoints
        checkpoint_times = Checkpoint.list_checkpoints(dirname)

        # select closest checkpoint
        if time is None:
            nearest_time = checkpoint_times[-1]
        else:
            nearest_time = checkpoint_times[numpy.argmax(numpy.array(checkpoint_times) < time)]

        file_name = Checkpoint.get_file_name(dirname, nearest_time)

        # load and return this checkpoint
        with open(file_name, 'rb') as file:
            return pickle.load(file)

    @staticmethod
    def list_checkpoints(dirname):
        """ Get sorted list of times of saved checkpoints in checkpoint directory `dirname`.

        Args:
            dirname (:obj:`str`): directory to read/write checkpoint data

        Returns:
            :obj:`list`: sorted list of times of saved checkpoints
        """

        # find checkpoint times
        checkpoint_times = []
        for file_name in os.listdir(dirname):
            match = re.match('^(\d+\.\d{6,6}).pickle$', file_name)
            if os.path.isfile(os.path.join(dirname, file_name)) and match:
                checkpoint_times.append(float(match.group(1)))

        # sort by time
        checkpoint_times.sort()

        # return list of checkpoint times
        return checkpoint_times

    @staticmethod
    def get_file_name(dirname, time):
        """ Get file name for checkpoint at time `time`

        Args:
            dirname (:obj:`str`): directory to read/write checkpoint data
            time (:obj:`float`): time in seconds

        Returns:
            :obj:`str`: file name for checkpoint at time `time`
        """

        return os.path.join(dirname, '{:0.6f}.pickle'.format(math.floor(time * 1e6) / 1e6))


class CheckpointLogger(object):
    """ Checkpoint logger

    Attributes:
        dirname (:obj:`str`): directory to write checkpoint data
        step (:obj:`float`): simulation time between checkpoints in seconds
        _next_checkpoint (:obj:`float`): time in seconds of next checkpoint
        metadata (:obj:`wc.sim.metadata.Metadata`): simulation run metadata
    """

    def __init__(self, dirname, step, initial_time, metadata):
        """
        Args:
            dirname (:obj:`str`): directory to write checkpoint data
            step (:obj:`float`): simulation time between checkpoints in seconds
            initial_time (:obj:`float`): starting simulation time
            metadata (:obj:`wc.sim.metadata.Metadata`): simulation run metadata
        """

        next_checkpoint = math.ceil(initial_time / step) * step
        if next_checkpoint == initial_time:
            next_checkpoint += step

        if not os.path.isdir(dirname):
            os.makedirs(dirname)

        self.dirname = dirname
        self.step = step
        self._next_checkpoint = next_checkpoint
        self.metadata = metadata

    def checkpoint_periodically(self, time, state, random_state):
        """ Periodically store checkpoint

        Args:
            time (:obj:`float`): simulation time in seconds
            state (:obj:`object`): simulated state (e.g. species counts)
            random_state (:obj:`numpy.random.RandState`): random number generator state
        """

        if time >= self._next_checkpoint:
            self.checkpoint(time, state, random_state)
            self._next_checkpoint += self.step

    def checkpoint(self, time, state, random_state):
        """ Store checkpoint at time `time` with state `state` and ranodom number generator state `random_state`

        Args:
            time (:obj:`float`): simulation time in seconds
            state (:obj:`object`): simulated state (e.g. species counts)
            random_state (:obj:`numpy.random.RandState`): random number generator state
        """

        Checkpoint.set_checkpoint(self.dirname, Checkpoint(self.metadata, time, state, random_state.get_state()))
