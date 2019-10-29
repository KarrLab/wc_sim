""" Store and retrieve combined results of a multialgorithmic simulation run

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2018-05-20
:Copyright: 2018-2019, Karr Lab
:License: MIT
"""
import numpy
import os
import pandas
import pickle

from wc_utils.util.misc import as_dict
from de_sim.checkpoint import Checkpoint
from de_sim.sim_metadata import SimulationMetadata
from wc_sim.multialgorithm_errors import MultialgorithmError


class RunResults(object):
    """ Store and retrieve combined results of a multialgorithmic simulation run

    Attributes:
        results_dir (:obj:`str`): pathname of a directory containing a simulation run's checkpoints and/or
            HDF5 file storing the combined results
        run_results (:obj:`dict`): dictionary of RunResults components, indexed by component name
    """
    # component stored in a RunResults instance and the HDF file it manages
    COMPONENTS = {
        'populations',          # predicted populations of species at all checkpoint times
        'aggregate_states',     # predicted aggregate states of the cell over the simulation
        'observables',          # predicted values of all observables over the simulation
        'functions',            # predicted values of all functions over the simulation
        'random_states',        # states of the simulation's random number geerators over the simulation
        'metadata',             # the simulation's global metadata
    }
    HDF5_FILENAME = 'run_results.h5'

    def __init__(self, results_dir):
        """ Create a `RunResults`

        Args:
            results_dir (:obj:`str`): directory storing checkpoints and/or HDF5 file with
                the simulation run results
        """
        self.results_dir = results_dir
        self.run_results = {}

        # if the HDF file containing the run results exists, open it
        if os.path.isfile(self._hdf_file()):
            self._load_hdf_file()

        # else create the HDF file from the stored metadata and sequence of checkpoints
        else:

            # create the HDF file containing the run results
            population_df, observables_df, functions_df, aggregate_states_df, random_states_s = self.convert_checkpoints()
            # populations
            population_df.to_hdf(self._hdf_file(), 'populations')
            # observables
            observables_df.to_hdf(self._hdf_file(), 'observables')
            # functions
            functions_df.to_hdf(self._hdf_file(), 'functions')
            # aggregate states
            aggregate_states_df.to_hdf(self._hdf_file(), 'aggregate_states')
            # random states
            random_states_s.to_hdf(self._hdf_file(), 'random_states')

            # metadata
            metadata_s = self.convert_metadata()
            metadata_s.to_hdf(self._hdf_file(), 'metadata')

            self._load_hdf_file()

    def _hdf_file(self):
        """ Provide the pathname of the HDF5 file storing the combined results

        Returns:
            :obj:`str`: the pathname of the HDF5 file storing the combined results
        """
        return os.path.join(self.results_dir, self.HDF5_FILENAME)

    def _load_hdf_file(self):
        """ Load run results from the HDF file
        """
        for component in self.COMPONENTS:
            self.run_results[component] = pandas.read_hdf(self._hdf_file(), component)

    def get(self, component):
        """ Read and provide the specified `component`

        Args:
            component (:obj:`str`): the name of the component to return

        Returns:
            :obj:`pandas.DataFrame`, or `pandas.Series`: a pandas object containing a component of
                this `RunResults`, as specified by `component`

        Raises:
            :obj:`MultialgorithmError`: if `component` is not an element of `RunResults.COMPONENTS`
        """
        if component not in RunResults.COMPONENTS:
            raise MultialgorithmError("component '{}' is not an element of {}".format(component,
                RunResults.COMPONENTS))
        return self.run_results[component]

    def convert_metadata(self):
        """ Convert the saved simulation metadata into a pandas series

        Returns:
            :obj:`pandas.Series`: the simulation metadata
        """
        simulation_metadata = SimulationMetadata.read_metadata(self.results_dir)
        return pandas.Series(as_dict(simulation_metadata))

    # todo: provide get functionality that hides the internal structure of state components
    @staticmethod
    def get_state_components(state):
        return (state['population'], state['observables'], state['functions'], state['aggregate_state'])

    def convert_checkpoints(self):
        """ Convert the data in saved checkpoints into pandas dataframes for loading into hdf

        Returns:
            :obj:`tuple` of pandas objects: dataframes of the components of a simulation checkpoint history
                population_df, observables_df, functions_df, aggregate_states_df, random_states_s
        """
        # create pandas objects for species populations, aggregate states and simulation random states
        checkpoints = Checkpoint.list_checkpoints(self.results_dir)
        first_checkpoint = Checkpoint.get_checkpoint(self.results_dir, time=0)
        species_pop, observables, functions, aggregate_state = self.get_state_components(first_checkpoint.state)

        species_ids = species_pop.keys()
        population_df = pandas.DataFrame(index=checkpoints, columns=species_ids, dtype=numpy.float64)

        observable_ids = observables.keys()
        observables_df = pandas.DataFrame(index=checkpoints, columns=observable_ids, dtype=numpy.float64)

        function_ids = functions.keys()
        functions_df = pandas.DataFrame(index=checkpoints, columns=function_ids, dtype=numpy.float64)

        compartments = list(aggregate_state['compartments'].keys())
        properties = list(aggregate_state['compartments'][compartments[0]].keys())
        compartment_property_tuples = list(zip(compartments, properties))
        columns = pandas.MultiIndex.from_tuples(compartment_property_tuples, names=['compartment', 'property'])
        aggregate_states_df = pandas.DataFrame(index=checkpoints, columns=columns)
        random_states_s = pandas.Series(index=checkpoints)

        # load these pandas objects
        for time in Checkpoint.list_checkpoints(self.results_dir):

            checkpoint = Checkpoint.get_checkpoint(self.results_dir, time=time)
            species_populations, observables, functions, aggregate_state = self.get_state_components(checkpoint.state)

            for species_id, population in species_populations.items():
                population_df.loc[time, species_id] = population

            for observable_id, observable in observables.items():
                observables_df.loc[time, observable_id] = observable

            for function_id, function in functions.items():
                functions_df.loc[time, function_id] = function

            # todo: could add cell aggregate properties to aggregate_states_df
            compartment_states = aggregate_state['compartments']
            for compartment_id, agg_states in compartment_states.items():
                for property, value in agg_states.items():
                    aggregate_states_df.loc[time, (compartment_id, property)] = value

            random_states_s[time] = pickle.dumps(checkpoint.random_state)

        return (population_df, observables_df, functions_df, aggregate_states_df, random_states_s)
