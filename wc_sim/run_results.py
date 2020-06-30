""" Store and retrieve combined results of a multialgorithmic simulation run

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2018-05-20
:Copyright: 2018-2019, Karr Lab
:License: MIT
"""

from scipy.constants import Avogadro
import dataclasses
import h5py
import numpy
import os
import pandas
import pickle

from de_sim.checkpoint import Checkpoint, AccessCheckpoints
from de_sim.simulation_metadata import SimulationMetadata
from wc_sim.metadata import WCSimulationMetadata
from wc_sim.model_utilities import ModelUtilities
from wc_sim.multialgorithm_errors import MultialgorithmError
from wc_utils.util.dict import DictUtil


class RunResults(object):
    """ Store and retrieve combined results of a multialgorithmic simulation run

    HDF5 uses attributes to store “self-describing” data. The official HDF5 way to store metadata is in the
    *attribute* field attached to each data object. Since a simulation's metadata applies to all :obj:`RunResults`
    components, it is stored in a `metadata` `Group` object in the HDF5 file.

    Attributes:
        results_dir (:obj:`str`): pathname of a directory containing a simulation run's checkpoints and/or
            HDF5 file storing the combined results
        run_results (:obj:`dict`): dictionary of RunResults components, indexed by component name
    """
    # components stored in a RunResults instance and the HDF file it manages
    COMPONENTS = {
        'populations',          # predicted populations of species at all checkpoint times
        'aggregate_states',     # predicted aggregate states of the cell at all checkpoint times
        'observables',          # predicted values of all observables at all checkpoint times
        'functions',            # predicted values of all functions at all checkpoint times
        'random_states',        # states of the simulation's random number generators at all checkpoint times
    }

    # components computed from stored components; map from component name to the method that computes it
    COMPUTED_COMPONENTS = {
        'volumes': 'get_volumes',   # volumes of all compartments at all checkpoint times
        'masses': 'get_masses',     # masses of all compartments at all checkpoint times
    }

    HDF5_FILENAME = 'run_results.h5'
    METADATA_GROUP = 'metadata'
    METADATA_CLASS_TO_NAME = {SimulationMetadata: 'de_sim_metadata',
                              WCSimulationMetadata: 'wc_sim_metadata'}

    def __init__(self, results_dir):
        """ Create a `RunResults`

        Args:
            results_dir (:obj:`str`): directory storing checkpoints and/or HDF5 file with
                the simulation run results
        """
        if results_dir is None:
            raise MultialgorithmError('results_dir is None')
        if not os.path.isdir(results_dir):
            raise MultialgorithmError(f'results_dir {results_dir} must be a directory')
        self.results_dir = results_dir
        self.run_results = {}

        # if an HDF file containing the run results does not exist, then
        # create it from the stored metadata and sequence of checkpoints
        if not os.path.isfile(self._hdf_file()):

            # create the HDF file containing the run results
            population_df, observables_df, functions_df, aggregate_states_df, random_states_s = \
                self.convert_checkpoints()

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
            self.convert_metadata(SimulationMetadata)
            self.convert_metadata(WCSimulationMetadata)

        # load the data in the HDF file containing the run results
        self._load_hdf_file()

    @classmethod
    def _prepare_computed_components(cls):
        """ Check and initialize the `COMPUTED_COMPONENTS`

        Raises:
            :obj:`MultialgorithmError`: if a value in `self.COMPUTED_COMPONENTS` is not a method
                in `RunResults`
        """
        for component, method in cls.COMPUTED_COMPONENTS.items():
            if hasattr(cls, method):
                cls.COMPUTED_COMPONENTS[component] = getattr(cls, method)
            else:
                raise MultialgorithmError("'{}' in COMPUTED_COMPONENTS is not a method in {}".format(
                    method, cls.__name__))

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

    def _check_component(self, component):
        """ Raise an exception if the component is empty

        Args:
            component (:obj:`str`): the name of the component to check

        Raises:
            :obj:`MultialgorithmError`: if `component` is empty
        """
        data = self.get(component)
        if data.empty:
            raise MultialgorithmError(f"'{component}' component is empty")

    def get(self, component):
        """ Provide the specified `component`

        Args:
            component (:obj:`str`): the name of the component to return

        Returns:
            :obj:`object`: an object containing a component of this `RunResults`, as specified by `component`;
                simulation time series data are :obj:`pandas.DataFrame` or `pandas.Series` instances;
                simulation metadata are :obj:`dict` instances.

        Raises:
            :obj:`MultialgorithmError`: if `component` is not an element of `RunResults.COMPONENTS`
                or `RunResults.COMPUTED_COMPONENTS`
        """
        all_components = RunResults.COMPONENTS.union(RunResults.COMPUTED_COMPONENTS)
        if component not in all_components:
            raise MultialgorithmError(f"component '{component}' is not an element of {all_components}")
        if component in RunResults.COMPUTED_COMPONENTS:
            return RunResults.COMPUTED_COMPONENTS[component](self)
        return self.run_results[component]

    def get_concentrations(self, compartment_id=None):
        """ Get species concentrations at checkpoint times

        Args:
            compartment_id (:obj:`str`, optional): if provided, obtain concentrations for species in
                `compartment_id`; otherwise, return the concentrations of all species

        Returns:
            :obj:`pandas.DataFrame`: the concentrations of species at checkpoint times, filtered
                by `compartment_id` if it's provided

        Raises:
            :obj:`MultialgorithmError`: if no species are in the compartment
        """
        populations = self.get('populations')
        self._check_component('populations')
        if compartment_id is None:
            # iterate over species in populations, dividing by the right compartment
            # (as of 0.25.3 pandas doesn't support joins between two MultiIndexes)
            pop_div_vol = populations.copy()
            for species_id in populations.columns.values:
                _, compartment_id = ModelUtilities.parse_species_id(species_id)
                pop_div_vol.loc[:, species_id] = pop_div_vol.loc[:, species_id] / \
                    self.get_volumes(compartment_id=compartment_id)
            concentrations = pop_div_vol / Avogadro
            return(concentrations)

        else:
            compartment_vols = self.get_volumes(compartment_id=compartment_id)
            # filter to populations for species in compartment_id
            filter = f'\[{compartment_id}\]$'
            filtered_populations = populations.filter(regex=filter)
            if filtered_populations.empty:  # pragma: no cover
                raise MultialgorithmError(f"No species found in compartment '{compartment_id}'")
            concentrations = filtered_populations.div(compartment_vols, axis='index') / Avogadro
            return(concentrations)

    def get_times(self):
        """ Get simulation times of results data

        Returns:
            :obj:`numpy.ndarray`: simulation times of results data
        """
        self._check_component('populations')
        return self.get('populations').index.values

    def aggregate_state_properties(self):
        """ Get the names of the aggregate state properties

        Returns:
            :obj:`set`: the names of the aggregate state properties in a `RunResults`
        """
        self._check_component('aggregate_states')
        aggregate_states_df = self.get('aggregate_states')
        return set(aggregate_states_df.columns.get_level_values('property').values)

    def get_properties(self, compartment_id, property=None):
        """ Get a compartment's aggregate state properties or property at checkpoint times

        Args:
            compartment_id (:obj:`str`): the id of the compartment
            property (:obj:`str`, optional): if provided, the property to return; otherwise,
                return all properties

        Returns:
            :obj:`pandas.DataFrame`: a compartment's properties or property at all checkpoint times
        """
        aggregate_states_df = self.get('aggregate_states')
        self._check_component('aggregate_states')
        if property is not None:
            return aggregate_states_df[compartment_id][property]
        return aggregate_states_df[compartment_id]

    def get_volumes(self, compartment_id=None):
        """ Get the compartment volumes at checkpoint times

        Args:
            compartment_id (:obj:`str`, optional): if provided, return the compartment's volumes;
                otherwise, return the volumes of all compartments

        Returns:
            :obj:`pandas.DataFrame`: the volumes of one compartment or all compartments at all checkpoint times
        """
        if compartment_id is not None:
            return self.get_properties(compartment_id, 'volume')
        aggregate_states = self.get('aggregate_states')
        self._check_component('aggregate_states')
        volumes = aggregate_states.loc[:, aggregate_states.columns.get_level_values('property') == 'volume']
        return volumes

    def get_masses(self, compartment_id=None):
        """ Get the compartment masses at checkpoint times

        Args:
            compartment_id (:obj:`str`, optional): if provided, return the compartment's masses;
                otherwise, return the masses of all compartments

        Returns:
            :obj:`pandas.DataFrame`: the masses of a compartment or all compartments at all checkpoint times
        """
        if compartment_id is not None:
            return self.get_properties(compartment_id, 'mass')
        aggregate_states = self.get('aggregate_states')
        self._check_component('aggregate_states')
        masses = aggregate_states.loc[:, aggregate_states.columns.get_level_values('property') == 'mass']
        return masses

    def convert_metadata(self, metadata_class):
        """ Convert the saved simulation metadata into HDF5 attributes on a `metadata` Group

        Args:
            metadata_class (:obj:`EnhancedDataClass`): the class that stored the metadata
        """
        metadata = metadata_class.read_dataclass(self.results_dir)

        # open the metadata file
        metadata_file = self._hdf_file()
        with h5py.File(metadata_file, 'a') as hdf5_file:
            # open or make a group in metadata_file called 'metadata'
            if self.METADATA_GROUP in hdf5_file:
                metadata_group = hdf5_file[self.METADATA_GROUP]
            else:
                metadata_group = hdf5_file.create_group(self.METADATA_GROUP)

            # make a nested dict of the metadata
            metadata_as_dict = dataclasses.asdict(metadata)
            metadata_class_name = self.METADATA_CLASS_TO_NAME[metadata_class]
            flat_metadata_as_dict = DictUtil.flatten_dict(metadata_as_dict)

            for key, value in flat_metadata_as_dict.items():
                # make a dotted string for each value in the metadata
                # metadata_as_dict keys cannot contain '.' because they're attribute names, therefore
                # '.' can be used as a separator in a name derived from concatenated keys
                separator = '.'
                name = f'{metadata_class_name}{separator}{separator.join(key)}'

                # make an attribute for each value
                if value is None:
                    value = 'NA'
                if not isinstance(value, (int, float, str)):
                    value = str(value)
                metadata_group.attrs[name] = value

    def get_metadata(self):
        """ Get simulation metadata from the HDF5 file

        Returns:
            :obj:`dict`: a nested dictionary of metadata corresponding to the attributes in
                :obj:`SimulationMetadata` and :obj:`WCSimulationMetadata`
        """
        hdf5_file = h5py.File(self._hdf_file(), 'r')
        metadata_attrs = hdf5_file[self.METADATA_GROUP].attrs
        metadata_attrs_as_dict = {}
        for key, value in metadata_attrs.items():
            metadata_attrs_as_dict[key] = value
        return DictUtil.expand_dict(metadata_attrs_as_dict)

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
        access_checkpoints = AccessCheckpoints(self.results_dir)
        checkpoints = access_checkpoints.list_checkpoints()
        first_checkpoint = access_checkpoints.get_checkpoint(time=checkpoints[0])
        species_pop, observables, functions, aggregate_state = self.get_state_components(first_checkpoint.state)

        species_ids = species_pop.keys()
        population_make_df = MakeDataFrame(checkpoints, species_ids)

        observable_ids = observables.keys()
        observables_make_df = MakeDataFrame(checkpoints, observable_ids)

        function_ids = functions.keys()
        function_make_df = MakeDataFrame(checkpoints, function_ids)

        compartments = list(aggregate_state['compartments'].keys())
        properties = list(aggregate_state['compartments'][compartments[0]].keys())
        compartment_property_tuples = list(zip(compartments, properties))
        columns = pandas.MultiIndex.from_tuples(compartment_property_tuples, names=['compartment', 'property'])
        aggregate_states_df = pandas.DataFrame(index=checkpoints, columns=columns, dtype=numpy.float64)
        random_states_s = pandas.Series(index=checkpoints)

        # load these pandas objects
        for time in access_checkpoints.list_checkpoints():

            checkpoint = access_checkpoints.get_checkpoint(time=time)
            species_populations, observables, functions, aggregate_state = self.get_state_components(checkpoint.state)

            population_make_df.add(time, species_populations)

            observables_make_df.add(time, observables)

            function_make_df.add(time, functions)

            compartment_states = aggregate_state['compartments']
            for compartment_id, agg_states in compartment_states.items():
                for property, value in agg_states.items():
                    aggregate_states_df.loc[time, (compartment_id, property)] = value

            random_states_s[time] = pickle.dumps(checkpoint.random_state)

        population_df = population_make_df.finish()
        observables_df = observables_make_df.finish()
        functions_df = function_make_df.finish()
        return (population_df, observables_df, functions_df, aggregate_states_df, random_states_s)

    # TODO(Arthur): exact caching: add metadata elements
    def __eq__(self, other):
        """ Determine whether two `RunResults` objects are semantically equal

        Args:
            other (:obj:`Object`): other object

        Returns:
            :obj:`bool`: true if `RunResults` objects are semantically equal
        """
        if other.__class__ is not self.__class__:
            return False

        for component in RunResults.COMPONENTS:
            if component != 'random_states' and not self.run_results[component].equals(other.run_results[component]):
                return False

        return True

    # TODO(Arthur): exact caching: add approximate match
    def semantically_equal(self, other, debug=False):
        """ Are the predictions and metadata in two :obj:`RunResults` objects semantically equal?

        Uses `numpy.allclose()` to compare predictions.
        Uses `semantically_equal` to ignore semantically insignificant metadata, such as the timestamp of a simulation's
        execution or its runtime.
        Compares all `RunResults.COMPONENTS` except 'random_states', which is ignored.

        Args:
            other (:obj:`Object`): other object
            debug (:obj:`bool`): whether to output debugging info when the objects are not semantically equal

        Returns:
            :obj:`bool`: :obj:`True` if `self` and `other` are semantically equal, :obj:`False` otherwise
        """
        if other.__class__ is not self.__class__:
            return False

        for component in RunResults.COMPONENTS:
            # TODO(Arthur): exact caching: tune tolerances for each component
            if component != 'random_states':
                if not self.dataframes_are_close(self.get(component), other.get(component)):
                    if debug:
                        print(f"{component} components are not close")
                    return False

        # use original metadata classes so their semantically_equal() methods can be used
        for metadata_class in self.METADATA_CLASS_TO_NAME:
            self_metadata = metadata_class.read_dataclass(self.results_dir)
            other_metadata = metadata_class.read_dataclass(other.results_dir)
            if not self_metadata.semantically_equal(other_metadata):
                if debug:
                    print(f"metadata {metadata_class.__name__} instances are not semantically equal")
                return False

        return True

    @staticmethod
    def dataframes_are_close(df1, df2, rtol=None, atol=None):
     """ Indicate whether two Pandas DataFrames that contain floats are almost equal

     Return :obj:`True` if rows and columns are identical and values are all close.

     Args:
         df1 (:obj:`pandas.DataFrame`): one DataFrame
         df2 (:obj:`pandas.DataFrame`): another DataFrame
         rtol (:obj:`float`, optional): relative tolerance; if provided, passed to `numpy.allclose()`
            which compares DataFrame values
         atol (:obj:`float`, optional): absolute tolerance, if provided, passed to `numpy.allclose()`
            which compares DataFrame values

     Returns:
         :obj:`bool`: :obj:`True` if the DataFrames are semantically equal

    Raises:
        :obj:`ValueError`: if either argument is not a :obj:`pandas.DataFrame`
     """
     if not isinstance(df1, pandas.DataFrame):
         raise ValueError(f"df1 is a(n) '{type(df1).__name__}', not a Pandas DataFrame")
     if not isinstance(df2, pandas.DataFrame):
         raise ValueError(f"df2 is a(n) '{type(df2).__name__}', not a Pandas DataFrame")
     if not df1.columns.equals(df2.columns):
         return False
     if not df1.index.equals(df2.index):
         return False
     kwargs = {}
     if rtol is not None:
        kwargs['rtol'] = rtol
     if atol is not None:
        kwargs['atol'] = atol
     return numpy.allclose(df1.values, df2.values, **kwargs)


class MakeDataFrame(object):
    """ Efficiently make a Pandas dataframe that contains a 2D numpy.ndarray
    """

    def __init__(self, times, columns):
        """ Initialize

        Args:
            times (:obj:`list`): the times of all checkpoints, in order
            columns (:obj:`list`): the column names of the Pandas dataframe, in order
        """
        self.times = times
        self.times_index = {time: i for i, time in enumerate(times)}
        self.columns = columns
        self.column_index = {column: i for i, column in enumerate(columns)}
        self.ndarray = numpy.zeros((len(times), len(columns)), dtype=numpy.float64)

    def add(self, time, iterator):
        """ Add data from another checkpoint

        Args:
            time (:obj:`float`): time of the checkpoint
            iterator (:obj:`dict`): `iterator.items()` iterates over (column, value) pairs for the data
                at time `time`
        """
        for column, value in iterator.items():
            self.ndarray[self.times_index[time]][self.column_index[column]] = value

    def finish(self):
        """ Efficiently make a Pandas dataframe from the `NumPy` array loaded by `add` calls

        Returns:
            :obj:`pandas.DataFrame`: a dataframe
        """
        return pandas.DataFrame(data=self.ndarray, index=self.times, columns=self.columns, dtype=numpy.float64)


RunResults._prepare_computed_components()
