""" Classes to record and retrieve simulation results.

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2017-07-25
:Copyright: 2016, Karr Lab
:License: MIT
"""

# todo: build distributed, multi-state logger
# todo: implement log reader

import numpy
import pickle


class Writer(object):
    """ Logs simulation results to file using pickle.

    Currently, this includes:
    * Time (s)
    * Volume (L)
    * Mass (g)
    * Species counts (molecules)

    Attributes:
        state (:obj:`wc.sim.state.Model`): compiled model whose simulation will be logged
        num_time_steps (:obj:`int`): number of time steps to keep in history
        log_path (:obj:`str`): path where simulation results should be stored
        _simulation_results (:obj:`dict`): data structure to represent simulation results
        _current_results_index (:obj:`int`): index of slice of simulation_results to which results should be written
    """

    # .. todo:: Use pytables rather than pickle
    # .. todo:: implement parallelizable logging system

    STATES = (
        'volume',
        'growth',
        'species_counts',
    )

    def __init__(self, state, num_time_steps, log_path):
        """ Construct a log writer.

        Args:
            state (:obj:`wc.sim.state.Model`): model state
            num_time_steps (:obj:`int`): Number of time steps to log
            log_path (:obj:`str`): Path to log file
        """
        self.state = state
        self.num_time_steps = num_time_steps
        self.log_path = log_path

        self._simulation_results = None
        self._current_results_index = -1

    def start(self):
        """ Starts simulation results log.
        * Allocates memory to store results
        * Records initial simulated state
        """

        # Allocate memory to store results
        n_steps = self.num_time_steps
        self._simulation_results = {
            'time': numpy.full((1, 1, n_steps + 1), numpy.nan),
        }
        for state in self.STATES:
            size = getattr(self.state, state).shape or (1, 1)            
            self._simulation_results[state] = numpy.full(list(size) + [n_steps + 1], numpy.nan)

        # Initial index of current log slice
        self._current_results_index = -1

        # Record initial simulated state
        self.append(0.)

    def append(self, time):
        """ Appends current simulated state to results log.

        Args:
            time (:obj:`float`): simulation time
        """

        # Advance index of current log slice
        self._current_results_index += 1

        # Record simulated state
        self._simulation_results['time'][:, :, self._current_results_index] = time
        for state in self.STATES:
            self._simulation_results[state][:, :, self._current_results_index] = getattr(self.state, state)

    def close(self):
        """ Finalizes simulation results log.
        * Writes simulation results to file
        * Deallocates memory
        """

        # Write results to file
        with open(self.log_path, 'wb') as file:
            pickle.dump(self._simulation_results, file)

        # Deallocate memory
        self._simulation_results = None


class Reader(object):
    """ Reads logged simulation results from pickle files

    Attributes:
        log_path (:obj:`str`): path to log
    """

    def __init__(self, log_path):
        """
        Args:
            log_path (:obj:`str`): path to log
        """
        self.log_path = log_path

    def run(self):
        """ Read a log        

        Returns:
            :obj:`dict`: logged results
        """
        with open(self.log_path, 'rb') as file:
            return pickle.load(file)
