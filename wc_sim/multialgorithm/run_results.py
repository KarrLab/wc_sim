""" Store and retrieve combined results of a multialgorithmic simulation run

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-05-20
:Copyright: 2018, Karr Lab
:License: MIT
"""
import numpy
import os
import pandas

from wc_utils.util.misc import as_dict
from wc_sim.log.checkpoint import Checkpoint
from wc_sim.core.sim_metadata import SimulationMetadata
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError


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
        'random_states',        # states of the simulation's pRNGs over the simulation
        'metadata',             # the simulation's global metadata
    }
    HDF5_FILENAME = 'run_results.h5'

    def __init__(self, results_dir, metadata=None):
        """ Create a RunResults

        Args:
            results_dir (:obj:`str`): directory storing checkpoints and/or HDF5 file with
                the simulation run results
            metadata (:obj:`SimulationMetadata`, optional): metadata for the simulation run;
                required if an HDF5 file combining simulation run results has not been already created
        """
        self.results_dir = results_dir
        self.run_results = {}

        # if the HDF file containing the run results exists, open it
        if os.path.isfile(self._hdf_file()):
            self._load_hdf_file()

        # else create the HDF file from the sequence of checkpoints
        else:
            if metadata is None:
                raise MultialgorithmError("'metadata' must be provided to create an HDF5 file")

            population_df, aggregate_states_df, random_states_s = self.convert_checkpoints()

            # create the HDF file containing the run results
            # populations
            population_df.to_hdf(self._hdf_file(), 'populations')
            # aggregate states
            aggregate_states_df.to_hdf(self._hdf_file(), 'aggregate_states')
            # random states
            random_states_s.to_hdf(self._hdf_file(), 'random_states')
            # metadata
            # create temporary dummy metadata
            dummy_metadata_s = pandas.Series('temporary dummy metadata', index=['test'])
            dummy_metadata_s.to_hdf(self._hdf_file(), 'metadata')

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

    def convert_checkpoints(self):
        """ Convert the data in saved checkpoints into pandas dataframes

        Returns:
            :obj:`tuple` of pandas objects: dataframes of the components of a simulation checkpoint history
                population_df, aggregate_states_df, random_states_s
        """
        # create pandas objects for species populations, aggregate states and simulation random states
        checkpoints = Checkpoint.list_checkpoints(self.results_dir)
        first_checkpoint = Checkpoint.get_checkpoint(self.results_dir, time=0)
        species_pop, aggregate_state = first_checkpoint.state

        species_ids = species_pop.keys()
        population_df = pandas.DataFrame(index=checkpoints, columns=species_ids, dtype=numpy.float64)

        compartments = list(aggregate_state['compartments'].keys())
        properties = list(aggregate_state['compartments'][compartments[0]].keys())
        compartment_property_tuples = list(zip(compartments, properties))
        columns = pandas.MultiIndex.from_tuples(compartment_property_tuples, names=['compartment', 'property'])
        aggregate_states_df = pandas.DataFrame(index=checkpoints, columns=columns)
        random_states_s = pandas.Series(index=checkpoints)

        # load these pandas objects
        for time in Checkpoint.list_checkpoints(self.results_dir):

            checkpoint = Checkpoint.get_checkpoint(self.results_dir, time=time)
            species_populations, aggregate_state = checkpoint.state
            for species_id,population in species_populations.items():
                population_df.loc[time, species_id] = population

            compartment_states = aggregate_state['compartments']
            for compartment_id,agg_states in compartment_states.items():
                for property,value in agg_states.items():
                    aggregate_states_df.loc[time, (compartment_id, property)] = value

            random_states_s[time] = pickle.dumps(checkpoint.random_state)

        return (population_df, aggregate_states_df, random_states_s)
