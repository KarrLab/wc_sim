"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-09-17
:Copyright: 2018, Karr Lab
:License: MIT
"""

from collections import namedtuple
from enum import Enum
from inspect import currentframe, getframeinfo
from matplotlib.backends.backend_pdf import PdfPages
from pprint import pprint, pformat
from scipy.constants import Avogadro
import inspect
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import re
import shutil
import tempfile
import time
import traceback
import warnings

from wc_lang.core import ReactionParticipantAttribute, Model
from wc_lang.io import Reader
from wc_onto import onto
from wc_sim.config import core as config_core_multialgorithm
from wc_sim.model_utilities import ModelUtilities
from wc_sim.multialgorithm_errors import MultialgorithmError
from wc_sim.run_results import RunResults
from wc_sim.simulation import Simulation
from wc_sim.testing.make_models import MakeModel
from wc_utils.util.dict import DictUtil
from wc_utils.util.misc import geometric_iterator
from wc_utils.util.ontology import are_terms_equivalent

config_multialgorithm = config_core_multialgorithm.get_config()['wc_sim']['multialgorithm']


class Error(Exception):
    """ Base class for exceptions involving `wc_sim` verification

    Attributes:
        message (:obj:`str`): the exception's message
    """
    def __init__(self, message=None):
        super().__init__(message)


class VerificationError(Error):
    """ Exception raised for errors in `wc_sim.verify`

    Attributes:
        message (:obj:`str`): the exception's message
    """
    def __init__(self, message=None):
        super().__init__(message)


class WcSimVerificationWarning(UserWarning):
    """ `wc_sim` Verification warning """
    pass


class ODETestIterators(object):
    """ Convenient iterators for exploring ODE submodel and solver parameter spaces
    """

    @staticmethod
    def ode_test_generator(ode_time_step_factors=None, tolerance_ranges=None):
        """ Generate a nested iteration of test arguments for an ODE submodel

        Iterates over ODE time step factor and solver tolerances

        Args:
            ode_time_step_factors (:obj:`list` of :obj:`float`): factors by which the ODE time step will
                be multiplied
            tolerance_ranges (:obj:`dict`): ranges for absolute and relative ODE tolerances;
                see `default_tolerance_ranges` below for the dict's structure; `rtol`, `atol` or both
                may be provided; configured defaults are used for tolerance(s) that are not provided

        Returns:
            :obj:`iterator` of :obj:`dict`: a generator of `kwargs` :obj:`dict`\ s for ODE time step factor,
                a relative tolerance for ODE solver, and an absolute tolerance for the ODE solver
        """
        # if necessary, provide defaults
        if ode_time_step_factors is None:
            ode_time_step_factors = [1.0]

        default_rtol = config_multialgorithm['rel_ode_solver_tolerance']
        default_atol = config_multialgorithm['abs_ode_solver_tolerance']
        default_tolerance_ranges = {'rtol': dict(min=default_rtol, max=default_rtol),
                                    'atol': dict(min=default_atol, max=default_atol)}
        if tolerance_ranges is None:
            tolerance_ranges = default_tolerance_ranges
        if 'rtol' not in tolerance_ranges:
            tolerance_ranges['rtol'] = default_tolerance_ranges['rtol']
        if 'atol' not in tolerance_ranges:
            tolerance_ranges['atol'] = default_tolerance_ranges['atol']

        for ode_time_step_factor in ode_time_step_factors:
            rtol_iterator = geometric_iterator(tolerance_ranges['rtol']['min'],
                                               tolerance_ranges['rtol']['max'],
                                               10)

            for rtol in rtol_iterator:
                atol_iterator = geometric_iterator(tolerance_ranges['atol']['min'],
                                                   tolerance_ranges['atol']['max'],
                                                   10)

                for atol in atol_iterator:
                    # return kwargs for ode_time_step_factor, rtol, and atol
                    ode_test_params = dict(ode_time_step_factor=ode_time_step_factor,
                                           rtol=rtol,
                                           atol=atol)
                    yield ode_test_params


class VerificationTestCaseType(Enum):
    """ Types of test cases, and the directory that stores them """
    CONTINUOUS_DETERMINISTIC = 'semantic'       # algorithms like ODE
    DISCRETE_STOCHASTIC = 'stochastic'          # algorithms like SSA
    MULTIALGORITHMIC = 'multialgorithmic'       # multiple integration algorithms
    FLUX_BALANCE_STEADY_STATE = 'fba'           # algorithms like FBA


class VerificationTestReader(object):
    """ Read a model verification test case

    Read and access settings and expected results of an SBML test suite test case

    Attributes:
        expected_predictions_df (:obj:`pandas.DataFrame`): the test case's expected predictions
        expected_predictions_file (:obj:`str`): pathname of the test case's expected predictions
        model (:obj:`wc_lang.Model`): the test case's `wc_lang` model
        model_filename (:obj:`str`): pathname of the test case's model file
        settings_file (:obj:`str`): pathname of the test case's settings file
        test_case_dir (:obj:`str`): pathname of the directory storing the test case
        test_case_num (:obj:`str`): the test case's unique ID number
        test_case_type (:obj:`VerificationTestCaseType`): the test case's type
    """
    SBML_FILE_SUFFIX = '.xml'
    def __init__(self, test_case_type_name, test_case_dir, test_case_num):
        if test_case_type_name not in VerificationTestCaseType.__members__:
            raise VerificationError("Unknown VerificationTestCaseType: '{}'".format(test_case_type_name))

        self.test_case_type = VerificationTestCaseType[test_case_type_name]
        self.test_case_dir = test_case_dir
        self.test_case_num = test_case_num

    def read_settings(self):
        """ Read a test case's settings into a key-value dictionary

        Returns:
            :obj:`dict`: key-value pairs for the test case's settings
        """
        self.settings_file = settings_file = os.path.join(self.test_case_dir, self.test_case_num+'-settings.txt')
        settings = {}
        errors = []
        try:
            with open(settings_file, 'r') as f:
                for line in f:
                    if line.strip():
                        key, value = line.strip().split(':', maxsplit=1)
                        if key in settings:
                            errors.append("duplicate key '{}' in settings file '{}'".format(key, settings_file))
                        settings[key] = value.strip()
        except Exception as e:
            errors.append("could not read settings file '{}': {}".format(settings_file, e))
        if errors:
            raise VerificationError('; '.join(errors))

        # convert settings values
        # convert all numerics to floats
        for key, value in settings.items():
            try:
                settings[key] = float(value)
            except:
                pass
        # split into lists
        for key in ['variables', 'amount', 'concentration']:
            if key in settings and settings[key]:
                settings[key] = re.split(r'\W+', settings[key])
        # convert meanRange and sdRange into numeric tuples
        for key in ['meanRange', 'sdRange']:
            if key in settings and settings[key]:
                settings[key] = eval(settings[key])
        return settings

    def read_expected_predictions(self):
        """ Get the test case's expected predictions

        Returns:
            :obj:`pandas.DataFrame`: the test case's expected predictions
        """
        self.expected_predictions_file = expected_predictions_file = os.path.join(
            self.test_case_dir, self.test_case_num+'-results.csv')
        expected_predictions_df = pd.read_csv(expected_predictions_file)
        # expected predictions should contain data for all time steps
        times = np.linspace(self.settings['start'], self.settings['duration'], num=int(self.settings['steps'] + 1))
        if not np.allclose(times, expected_predictions_df.time):
            raise VerificationError("times in settings '{}' differ from times in expected predictions '{}'".format(
                self.settings_file, expected_predictions_file))

        if self.test_case_type == VerificationTestCaseType.CONTINUOUS_DETERMINISTIC:
            # expected predictions should contain time and the amount or concentration of each species
            # todo: use more SBML test suite cases and obtain expected predictions as amount or concentration
            expected_columns = set(self.settings['amount'])
            actual_columns = set(expected_predictions_df.columns.values)
            if expected_columns - actual_columns:
                raise VerificationError("some amounts missing from expected predictions '{}': {}".format(
                    expected_predictions_file, expected_columns - actual_columns))

        if self.test_case_type in [VerificationTestCaseType.DISCRETE_STOCHASTIC,
                                   VerificationTestCaseType.MULTIALGORITHMIC]:
            # expected predictions should contain the mean and sd of each variable in 'amount'
            expected_columns = set()
            for amount in self.settings['amount']:
                expected_columns.add(amount+'-mean')
                expected_columns.add(amount+'-sd')
            if expected_columns - set(expected_predictions_df.columns.values):
                raise VerificationError("mean or sd of some amounts missing from expected predictions '{}': {}".format(
                    expected_predictions_file, expected_columns - set(expected_predictions_df.columns.values)))

        return expected_predictions_df

    def species_columns(self):
        """ Get the names of the species columns

        Returns:
            :obj:`set`: the names of the species columns
        """
        return set(self.expected_predictions_df.columns.values) - {'time'}

    def slope_of_predictions(self):
        """ Determine the expected derivatives of species from the expected populations

        Returns:
            :obj:`pandas.DataFrame`: expected derivatives inferred from the expected populations
        """
        if self.test_case_type == VerificationTestCaseType.CONTINUOUS_DETERMINISTIC:
            # obtain linear estimates of derivatives
            results = self.expected_predictions_df
            times = results.time
            derivatives = pd.DataFrame(np.nan, index=results.index.copy(deep=True),
                                       columns=results.columns.copy(deep=True),
                                       dtype=np.float64)
            derivatives.time = times.copy()
            for species in self.species_columns():
                species_pops = results[species].values
                for i in range(len(times)-1):
                    derivatives.loc[i, species] = \
                        (species_pops[i+1] - species_pops[i]) / (times[i+1] - times[i])

            return derivatives

    def read_model(self, model_file_suffix='-wc_lang.xlsx'):
        """  Read a model into a `wc_lang` representation

        Args:
            model_file_suffix (:obj:`str`, optional): the name suffix for the model

        Returns:
            :obj:`wc_lang.Model`: the root of the test case's `wc_lang` model

        Raises:
            :obj:`VerificationError`: if an SBML model is read
        """
        self.model_filename = os.path.join(self.test_case_dir, self.test_case_num + model_file_suffix)
        if self.model_filename.endswith(self.SBML_FILE_SUFFIX):
            raise VerificationError(f"SBML files not supported: model filename: '{self.model_filename}'")
        return Reader().run(self.model_filename, validate=True)[Model][0]

    def get_species_id(self, species_type):
        """ Get the species id of a species type

        Raises an error if the given species type is contained in multiple compartments. I believe
        that this doesn't occur for models in the SBML test suite.

        Args:
            species_type (:obj:`str`): the species type of the species in the SBML test case

        Returns:
            :obj:`str`: the species id of the species type

        Raises:
            :obj:`VerificationError`: if multiple species ids are found for `species_type`, or
                if no species id is found for `species_type`
        """
        species_id = None
        for species in self.model.get_species():
            possible_species_type_id, _ = ModelUtilities.parse_species_id(species.id)
            if possible_species_type_id == species_type:
                if species_id is not None:
                        raise VerificationError(f"multiple species ids for species_type {species_type}: "
                                                f"{species_id} and {species.id}")
                else:
                    species_id = species.id
        if species_id is None:
            raise VerificationError(f"no species id found for species_type '{species_type}'")
        return species_id

    def run(self):
        """ Read the verification test
        """
        self.settings = self.read_settings()
        self.expected_predictions_df = self.read_expected_predictions()
        self.model = self.read_model()
        # todo: check that the variances on the model's distributions are zero, or set them

    def __str__(self):
        rv = []
        rv.append(pformat(self.settings))
        rv.append(pformat(self.expected_predictions_df))
        for attr in ['expected_predictions_file',
                     'model_filename',
                     'settings_file',
                     'test_case_dir',
                     'test_case_num',
                     'test_case_type',]:
            if hasattr(self, attr):
                rv.append(f'{attr}: {getattr(self, attr)}')
        return '\n'.join(rv)


class ResultsComparator(object):
    """ Compare simulated population predictions against expected populations

    Attributes:
        verification_test_reader (:obj:`VerificationTestReader`): the test case's reader
        simulation_run_results (:obj:`RunResults` or :obj:`list` of :obj:`RunResults`): simulation run results;
            results for 1 run of a CONTINUOUS_DETERMINISTIC integration, or a :obj:`list` of results
            for multiple runs of a stochastic integrator
        n_runs (:obj:`int`): number off runs of a stochastic integrator
        default_tolerances (:obj:`dict`): default tolerance specifications for ODE integrator
        simulation_pop_means (:obj:`dict`): map from species id to ndarray of mean populations over a
            simulation trajectory
    """
    TOLERANCE_MAP = dict(
        rtol='relative',
        atol='absolute'
    )

    def __init__(self, verification_test_reader, simulation_run_results):
        self.verification_test_reader = verification_test_reader
        self.simulation_run_results = simulation_run_results
        # obtain default tolerances in np.allclose()
        self.default_tolerances = VerificationUtilities.get_default_args(np.allclose)

    def prepare_tolerances(self):
        """ Prepare tolerance dictionary

        Use values from `verification_test_reader.settings` if available, otherwise from
        `numpy.allclose()`s defaults.

        Returns:
            :obj:`dict`: kwargs for `rtol` and `atol` tolerances for use by `numpy.allclose()`
        """
        kwargs = {}
        for np_name, testing_name in self.TOLERANCE_MAP.items():
            kwargs[np_name] = self.default_tolerances[np_name]
            if testing_name in self.verification_test_reader.settings:
                try:
                    tolerance = float(self.verification_test_reader.settings[testing_name])
                    kwargs[np_name] = tolerance
                except:
                    pass
        return kwargs

    def quantify_stoch_diff(self, evaluate=False):
        """ Quantify the difference between stochastic simulation population(s) and expected population(s)

        Used to tune multialgorithmic models

        Args:
            evaluate (:obj:`bool`, optional): if `False` return an array of Z scores for each species;
                if `True` return mean Z score for each species

        Returns:
            :obj:`dict`: map from each species to its difference from the expcected population;
                returns the normalized Z score for DISCRETE_STOCHASTIC and MULTIALGORITHMIC models
        """
        differences = {}
        if self.verification_test_reader.test_case_type in [VerificationTestCaseType.DISCRETE_STOCHASTIC,
                                                            VerificationTestCaseType.MULTIALGORITHMIC]:
            # TODO: warn if values lack precision; want int64 integers and float64 floats
            # see https://docs.scipy.org/doc/numpy-1.15.0/reference/arrays.scalars.html
            # use warnings.warn("", WcSimVerificationWarning)

            ### compute Z scores of mean differences ###
            self.n_runs = n_runs = len(self.simulation_run_results)

            self.simulation_pop_means = {}
            for species_type in self.verification_test_reader.settings['amount']:
                # extract n x 1 correct mean and sd np arrays
                correct_df = self.verification_test_reader.expected_predictions_df
                e_mean = correct_df.loc[:, species_type+'-mean'].values
                e_sd = correct_df.loc[:, species_type+'-sd'].values
                # errors if e_sd or e_mean < 0
                if np.any(e_mean < 0):
                    raise VerificationError("e_mean contains negative value(s)")
                if np.any(e_sd < 0):
                    raise VerificationError("e_sd contains negative value(s)")

                # avoid division by 0 when sd==0 by masking off SDs that are very close to 0
                mask = np.isclose(np.zeros_like(e_sd), e_sd, atol=1e-14, rtol=0)
                if 2 < np.count_nonzero(mask) or 0.3 < np.count_nonzero(mask) / len(mask):
                    raise VerificationError(f"e_sd contains too many zero(s): {np.count_nonzero(mask)}")

                # load simul. runs into 2D np array to find mean and SD
                species_id = self.verification_test_reader.get_species_id(species_type)
                pop_mean, pop_sd = SsaEnsemble.results_mean_n_sd(self.simulation_run_results, species_id)
                self.simulation_pop_means[species_type] = pop_mean
                Z = np.zeros_like(pop_mean)
                Z[~mask] = math.sqrt(n_runs) * (pop_mean[~mask] - e_mean[~mask]) / e_sd[~mask]
                differences[species_type] = Z

            if evaluate:
                # find mean diff for each species
                for species_type, Z in differences.items():
                    differences[species_type] = np.mean(Z)
                return differences
            return differences

    def differs(self):
        """ Evaluate whether the species amounts predicted by simulation run(s) differ from the correct amounts

        Returns:
            :obj:`obj`: `False` if amounts in the simulation run(s) and the correct amounts are equal
                within tolerances, otherwise :obj:`list`: of species types whose amounts differ
        """
        differing_values = []
        if self.verification_test_reader.test_case_type == VerificationTestCaseType.CONTINUOUS_DETERMINISTIC:
            kwargs = self.prepare_tolerances()
            # todo: fix: if expected values in settings are in 'amounts' then SB moles, not concentrations
            concentrations_df = self.simulation_run_results.get_concentrations()
            # for each prediction, determine if its trajectory is close enough to the expected predictions
            for species_type in self.verification_test_reader.settings['amount']:
                species_id = self.verification_test_reader.get_species_id(species_type)
                if not np.allclose(self.verification_test_reader.expected_predictions_df[species_type].values,
                    concentrations_df[species_id].values, **kwargs):
                    differing_values.append(species_type)
            return differing_values or False

        if self.verification_test_reader.test_case_type in [VerificationTestCaseType.DISCRETE_STOCHASTIC,
                                                            VerificationTestCaseType.MULTIALGORITHMIC]:
            """ Test mean population over multiple runs

            Follow algorithm in
            github.com/sbmlteam/sbml-test-suite/blob/master/cases/stochastic/DSMTS-userguide-31v2.pdf,
            from Evans, et al. The SBML discrete stochastic models test suite, Bioinformatics, 24:285-286, 2008.
            """

            ### test means ###
            mean_min_Z, mean_max_Z = self.verification_test_reader.settings['meanRange']
            differences = self.quantify_stoch_diff()
            for species_type, Z in differences.items():

                # compare with mean_range
                if np.any(Z < mean_min_Z) or np.any(mean_max_Z < Z):
                    differing_values.append(species_type)

            ### test sds ###
            # TODO: test sds

            return differing_values or False


class SsaEnsemble(object):
    """ Handle an SSA ensemble
    """

    SUBMODEL_ALGORITHM = 'SSA'

    @staticmethod
    def convert_ssa_submodels_to_nrm(model):
        for submodel in model.submodels:
            if are_terms_equivalent(submodel.framework, onto['WC:stochastic_simulation_algorithm']):
                submodel.framework = onto['WC:next_reaction_method']

    @staticmethod
    def run(model, simul_kwargs, tmp_results_dir, num_runs):
        """ Simulate a stochastic model for multiple runs

        Args:
            model (:obj:`wc_lang.Model`): a model to simulate
            simul_kwargs (:obj:`dict`): kwargs for `Simulation.run()`
            tmp_results_dir (:obj:`str`): path of tmp directory for storing results
            num_runs (:obj:`int`): number of Monte Carlo runs to make

        Returns:
            :obj:`list`: of :obj:`RunResults`: a :obj:`RunResults` for each simulation run
        """
        if SsaEnsemble.SUBMODEL_ALGORITHM == 'NRM':
            SsaEnsemble.convert_ssa_submodels_to_nrm(model)

        simulation_run_results = []
        progress_factor = 1
        for i in range(1, num_runs+1):
            if i == progress_factor:
                print("starting run {} of {} of model {}".format(i, num_runs, model.id))
                progress_factor *= 2
            simul_kwargs['results_dir'] = tempfile.mkdtemp(dir=tmp_results_dir)
            simulation = Simulation(model)
            results_dir = simulation.run(**simul_kwargs).results_dir
            simulation_run_results.append(RunResults(results_dir))
            # remove results_dir after RunResults created
            shutil.rmtree(simul_kwargs['results_dir'])
        return simulation_run_results

    @staticmethod
    def results_mean_n_sd(simulation_run_results, species_id):
        """ Obtain the mean and standard deviation time courses of a species across multiple runs

        Args:
            simulation_run_results (:obj:`list`: of :obj:`RunResults`): results for each simulation run
            species_id (:obj:`str`): id of a species

        Returns:
            :obj:`tuple`: a pair of numpy arrays, for the mean and standard deviation time courses
        """
        times = simulation_run_results[0].get_times()
        n_times = len(times)
        n_runs = len(simulation_run_results)
        run_results_array = np.empty((n_times, n_runs))
        for idx, run_result in enumerate(simulation_run_results):
            run_pop_df = run_result.get('populations')
            run_results_array[:, idx] = run_pop_df.loc[:, species_id].values
        # take mean & sd at each time
        return run_results_array.mean(axis=1), run_results_array.std(axis=1)


class CaseVerifier(object):
    """ Verify or evaluate a test case

    Attributes:
        default_num_stochastic_runs (:obj:`int`): default number of Monte Carlo runs for SSA simulations
        num_runs (:obj:`int`): actual number of Monte Carlo runs for an SSA test
        results_comparator (:obj:`ResultsComparator`): object that compares expected and actual predictions
        simulation_run_results (:obj:`RunResults`): results for a simulation run
        test_case_dir (:obj:`str`): directory containing the test case
        tmp_results_dir (:obj:`str`): temporary directory for simulation results
        verification_test_reader (:obj:`VerificationTestReader`): the test case's reader
        comparison_result (:obj:`obj`): `False` if populations in the expected result and simulation run
            are equal within tolerances, otherwise :obj:`list`: of species types whose populations differ
    """

    # maximum number of attempts to verify an SSA model
    MAX_TRIES = 3

    def __init__(self, test_cases_root_dir, test_case_type, test_case_num,
                 default_num_stochastic_runs=None):
        """ Read model, config and expected predictions

        Args:
            test_cases_root_dir (:obj:`str`): pathname of directory containing test case files
            test_case_type (:obj:`str`): the type of case, `CONTINUOUS_DETERMINISTIC`
                `DISCRETE_STOCHASTIC`, or `MULTIALGORITHMIC`
            test_case_num (:obj:`str`): unique id of a verification case
            num_stochastic_runs (:obj:`int`, optional): number of Monte Carlo runs for an SSA test
        """
        self.test_case_dir = os.path.join(test_cases_root_dir,
                                          VerificationTestCaseType[test_case_type].value, test_case_num)
        self.verification_test_reader = VerificationTestReader(test_case_type, self.test_case_dir,
                                                               test_case_num)
        self.verification_test_reader.run()
        self.default_num_stochastic_runs = default_num_stochastic_runs
        if default_num_stochastic_runs is None:
            self.default_num_stochastic_runs = config_multialgorithm['num_ssa_verification_sim_runs']

    def verify_model(self, num_discrete_stochastic_runs=None, discard_run_results=True, plot_file=None,
                     ode_time_step_factor=None, tolerances=None, evaluate=False):
        """ Verify a model

        Args:
            num_discrete_stochastic_runs (:obj:`int`, optional): number of stochastic runs
            discard_run_results (:obj:`bool`, optional): whether to discard run results
            plot_file (:obj:`str`, optional): path of plotted results, if desired
            ode_time_step_factor (:obj:`float`, optional): factor by which to multiply the ODE time step
            tolerances (:obj:`dict`, optional): if testing tolerances, values of ODE solver tolerances
            evaluate (:obj:`bool`, optional): control the return value

        Returns:
            :obj:`obj`: if `evaluate` is `False`, then return `False` if populations in the expected
                result and simulation run are equal within tolerances, otherwise :obj:`list`: of species
                types whose populations differ;
                if `evaluate` is `True`, and the model contains a stochastic submmodel, then return
                a :obj:`dict` containing the mean Z-score for each species

        Raises:
            :obj:`VerificationError`: if 'duration' or 'steps' are missing from settings, or
                settings requires simulation to start at time other than 0, or
                `evaluate` is `True` and test_case_type is CONTINUOUS_DETERMINISTIC
        """

        # check settings
        required_settings = ['duration', 'steps']
        settings = self.verification_test_reader.settings
        errors = []
        for setting in required_settings:
            if setting not in settings:
                errors.append("required setting '{}' not provided".format(setting))
            elif not isinstance(settings[setting], float):
                errors.append("required setting '{}' not a float".format(setting))
        if errors:
            raise VerificationError('; '.join(errors))
        if 'start' in settings and settings['start'] != 0:
            raise VerificationError("non-zero start setting ({}) not supported".format(settings['start']))

        if self.verification_test_reader.test_case_type == VerificationTestCaseType.CONTINUOUS_DETERMINISTIC \
            and evaluate:
                raise VerificationError("evaluate is True and test_case_type is CONTINUOUS_DETERMINISTIC")

        # prepare for simulation
        self.tmp_results_dir = tmp_results_dir = tempfile.mkdtemp()
        simul_kwargs = dict(time_max=settings['duration'],
                            checkpoint_period=settings['duration']/settings['steps'],
                            results_dir=tmp_results_dir,
                            verbose= False)

        simul_kwargs['ode_time_step'] = settings['duration']/settings['steps']
        if ode_time_step_factor is not None:
            simul_kwargs['ode_time_step'] *= ode_time_step_factor

        if self.verification_test_reader.test_case_type == VerificationTestCaseType.CONTINUOUS_DETERMINISTIC:
            ## 1. run simulation
            simulation = Simulation(self.verification_test_reader.model)
            if tolerances:
                simul_kwargs['options'] = dict(OdeSubmodel=dict(options=dict(tolerances=tolerances)))
            results_dir = simulation.run(**simul_kwargs).results_dir
            self.simulation_run_results = RunResults(results_dir)

            ## 2. compare results
            self.results_comparator = ResultsComparator(self.verification_test_reader, self.simulation_run_results)
            self.comparison_result = self.results_comparator.differs()

        if self.verification_test_reader.test_case_type in [VerificationTestCaseType.DISCRETE_STOCHASTIC,
                                                            VerificationTestCaseType.MULTIALGORITHMIC]:
            # make multiple simulation runs with different random seeds
            if num_discrete_stochastic_runs is not None and 0 < num_discrete_stochastic_runs:
                num_runs = num_discrete_stochastic_runs
            else:
                num_runs = self.default_num_stochastic_runs
            self.num_runs = num_runs

            ## retry on failure
            # if failure, rerun and evaluate; correct simulations will fail to verify
            # (100*(p-value threshold)) percent of the time
            # assuming p-value << 1, two failures indicate a likely error
            # if failure repeats, report it
            try_num = 1
            while try_num <= self.MAX_TRIES:
                ## 1. run simulation
                self.simulation_run_results = \
                    SsaEnsemble.run(self.verification_test_reader.model, simul_kwargs, tmp_results_dir, num_runs)

                ## 2. compare results
                self.results_comparator = ResultsComparator(self.verification_test_reader,
                                                            self.simulation_run_results)
                self.comparison_result = self.results_comparator.differs()
                if evaluate:
                    self.evaluation = self.results_comparator.quantify_stoch_diff(evaluate=evaluate)
                # if model & simulation verify or evaluating, don't retry
                if not self.comparison_result or evaluate:
                    break
                try_num += 1

        ## 3 plot comparison of actual and expected trajectories
        if plot_file:
            self.plot_model_verification(plot_file)
        # TODO: optionally, save results
        # TODO: output difference between actual and expected trajectory

        ## 4. cleanup
        if discard_run_results:
            shutil.rmtree(self.tmp_results_dir)

        if evaluate:
            return self.evaluation
        return self.comparison_result

    def get_model_summary(self):
        """ Obtain a text summary of the test model
        """
        mdl = self.verification_test_reader.model
        summary = ['Model Summary:']
        summary.append("model '{}':".format(mdl.id))
        for cmpt in mdl.compartments:
            summary.append(f"comp {cmpt.id}:\nmean init V {cmpt.init_volume.mean}, "
                           f"{cmpt.biological_type.name.replace(' compartment', '')}, "
                           f"{cmpt.physical_type.name.replace(' compartment', '')}")
        reaction_participant_attribute = ReactionParticipantAttribute()
        for sm in mdl.submodels:
            summary.append("submodel {}:".format(sm.id))
            summary.append("framework {}:".format(sm.framework.name.title()))
            for rxn in sm.reactions:
                summary.append("rxn & rl '{}': {}, {}".format(rxn.id,
                    reaction_participant_attribute.serialize(rxn.participants),
                    rxn.rate_laws[0].expression.serialize()))
        for param in mdl.get_parameters():
            summary.append("param: {}={} ({})".format(param.id, param.value, param.units))
        for func in mdl.get_functions():
            summary.append("func: {}={} ({})".format(func.id, func.expression, func.units))
        return summary

    def get_test_case_summary(self):
        """ Summarize the test case
        """
        # TODO: include # events, run time, perhaps details on error of failed tests
        summary = ['Test Case Summary']
        if self.comparison_result:
            summary.append("Failing species: {}".format(', '.join(self.comparison_result)))
        else:
            summary.append('All species verify')
        if hasattr(self, 'num_runs'):
            summary.append("Num simul runs: {}".format(self.num_runs))
        summary.append("Test case type: {}".format(self.verification_test_reader.test_case_type.name))
        summary.append("Test case number: {}".format(self.verification_test_reader.test_case_num))
        return summary

    def plot_model_verification(self, plot_file, max_runs_to_plot=100, presentation_qual=False):
        """Plot a model verification run

        Args:
            plot_file (:obj:`str`): name of file in which to save the plot
            max_runs_to_plot (:obj:`int`, optional): maximum number of runs to show when plotting
                Monte Carlo data
            presentation_qual (:obj:`bool`, optional): whether to produce presentation quality plot,
                without debugging information; or not

        Returns:
            :obj:`str`: location of the plot
        """
        # TODO: configure max_runs_to_plot in config file
        # TODO: use matplotlib 3; use the more flexible OO API instead of pyplot
        num_species_types = len(self.verification_test_reader.settings['amount'])
        if num_species_types == 1:
            n_rows = 1
            n_cols = 1
        elif num_species_types == 2:
            n_rows = 1
            n_cols = 2
        elif 2 < num_species_types <= 4:
            n_rows = 2
            n_cols = 2
        else:
            # TODO: better handle situation of more than 4 plots
            raise ValueError(f'cannot plot more than 4 species_types num_species_types: {num_species_types}')

        legend_fontsize = 9 if presentation_qual else 5
        plot_num = 1
        if self.verification_test_reader.test_case_type == VerificationTestCaseType.CONTINUOUS_DETERMINISTIC:
            times = self.simulation_run_results.get_times()
            simulated_concentrations = self.simulation_run_results.get_concentrations()
            for species_type in self.verification_test_reader.settings['amount']:
                plt.subplot(n_rows, n_cols, plot_num)

                # linestyles, designed so simulated and expected curves which are equal will both be visible
                dashdotted = (0, (2, 3, 3, 2))
                loosely_dashed = (0, (4, 6))
                linewidth = 0.8
                expected_plot_kwargs = dict(color='red', linestyle=dashdotted,
                                            linewidth=linewidth)
                simulated_plot_kwargs = dict(color='blue', linestyle=loosely_dashed,
                                             linewidth=linewidth)

                # plot expected predictions
                expected_mean_df = self.verification_test_reader.expected_predictions_df.loc[:, species_type]
                correct_mean, = plt.plot(times, expected_mean_df.values, **expected_plot_kwargs)

                # plot simulation populations
                species = self.verification_test_reader.get_species_id(species_type)
                species_simulated_conc = simulated_concentrations.loc[:, species]
                simul_pops, = plt.plot(times, species_simulated_conc, **simulated_plot_kwargs)

                units = 'M'
                plt.ylabel(f'{species_type} ({units})', fontsize=10)
                plt.xlabel('time (s)', fontsize=10)
                plt.legend((simul_pops, correct_mean),
                    (f'{species_type} simulation run', 'Correct concentration'),
                    loc='lower left', fontsize=legend_fontsize)
                plot_num += 1

        if self.verification_test_reader.test_case_type in [VerificationTestCaseType.DISCRETE_STOCHASTIC,
                                                              VerificationTestCaseType.MULTIALGORITHMIC]:
            times = self.simulation_run_results[0].get_times()

            for species_type in self.verification_test_reader.settings['amount']:
                plt.subplot(n_rows, n_cols, plot_num)

                # plot simulation pops
                # TODO: try making individual runs visible by slightly varying color and/or width
                species = self.verification_test_reader.get_species_id(species_type)
                for rr in self.simulation_run_results[:max_runs_to_plot]:
                    pop_time_series = rr.get('populations').loc[:, species]
                    simul_pops, = plt.plot(times, pop_time_series, 'b-', linewidth=0.1)

                # plot expected predictions
                expected_kwargs = dict(color='red', linewidth=1)
                expected_mean_df = self.verification_test_reader.expected_predictions_df.loc[:, species_type+'-mean']
                correct_mean, = plt.plot(times, expected_mean_df.values, **expected_kwargs)
                # mean +/- 3 sd
                expected_kwargs['linestyle'] = 'dashed'
                expected_sd_df = self.verification_test_reader.expected_predictions_df.loc[:, species_type+'-sd']
                # TODO: range for mean (-3, +3) should be taken from settings data
                correct_mean_plus_3sd, = plt.plot(times, expected_mean_df.values + 3 * expected_sd_df \
                                                  / math.sqrt(self.results_comparator.n_runs), **expected_kwargs)
                correct_mean_minus_3sd, = plt.plot(times, expected_mean_df.values - 3 * expected_sd_df \
                                                   / math.sqrt(self.results_comparator.n_runs), **expected_kwargs)

                # plot mean simulation pop
                model_kwargs = dict(color='green')
                model_mean, = plt.plot(times, self.results_comparator.simulation_pop_means[species_type], **model_kwargs)

                plt.ylabel('{} (molecules)'.format(species_type), fontsize=10)
                plt.xlabel('time (s)', fontsize=10)
                num_runs = len(self.simulation_run_results)
                runs_note = ''
                if max_runs_to_plot < num_runs:
                    runs_note = " ({} of {} runs)".format(max_runs_to_plot, num_runs)
                plt.legend((simul_pops, model_mean, correct_mean, correct_mean_plus_3sd),
                    ('{} runs{}'.format(species_type, runs_note),
                        'Mean of {} runs'.format(num_runs),
                        'Correct mean', '+/- 3Z thresholds'),
                    loc='lower left', fontsize=legend_fontsize)
                plot_num += 1

        summary = self.get_model_summary()
        middle = len(summary)//2
        x_pos = 0.05
        y_pos = 0.85
        for lb, ub in [(0, middle), (middle, len(summary))]:
            if not presentation_qual:
                text = plt.figtext(x_pos, y_pos, '\n'.join(summary[lb:ub]), fontsize=5)
            # TODO: position text automatically
            x_pos += 0.3
        test_case_summary = self.get_test_case_summary()
        if not presentation_qual:
            plt.figtext(0.8, y_pos, '\n'.join(test_case_summary), fontsize=5)

        if presentation_qual:
            test_reader = self.verification_test_reader
            suptitle = "{} case {}, '{}': {}".format(
                test_reader.test_case_type.name.replace('_', ' ').title(),
                test_reader.test_case_num,
                self.verification_test_reader.model.name,
                'failed' if self.comparison_result else 'verified'
                )
            plt.suptitle(suptitle, fontsize=10)
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        else:
            plt.tight_layout(rect=[0, 0.03, 1, 0.8])
        fig = plt.gcf()
        fig.savefig(plot_file)
        plt.close(fig)
        return "Wrote: {}".format(plot_file)


class VerificationResultType(Enum):
    """ Types of verification results """
    CASE_UNREADABLE = 'could not read case'
    FAILED_VERIFICATION_RUN = 'verification run failed'
    SLOW_VERIFICATION_RUN = 'verification run timed out'
    CASE_DID_NOT_VERIFY = 'case did not verify'
    CASE_VERIFIED = 'case verified'
    VERIFICATION_UNKNOWN = 'verification not evaluated'


VerificationRunResult = namedtuple('VerificationRunResult',
                                   ['cases_dir', 'case_type', 'case_num', 'result_type',
                                    'duration', 'quant_diff', 'params', 'error'])
# make quant_diff, error and params optional: see https://stackoverflow.com/a/18348004
VerificationRunResult.__new__.__defaults__ = (None, None, None,)
VerificationRunResult.__doc__ += ': results for one verification test run'
VerificationRunResult.cases_dir.__doc__ = 'directory containing the case(s)'
VerificationRunResult.case_type.__doc__ = 'type of the case, a name in `VerificationTestCaseType`'
VerificationRunResult.case_num.__doc__ = 'case number'
VerificationRunResult.result_type.__doc__ = "a VerificationResultType: the result's classification"
VerificationRunResult.duration.__doc__ = 'time it took to run the test'
VerificationRunResult.quant_diff.__doc__ = ('mean Z-score difference between correct means and actual '
                                            'simulation predictions')
VerificationRunResult.params.__doc__ = 'optional, parameters used by the test'
VerificationRunResult.error.__doc__ = 'optional, error message for the test'


class VerificationSuite(object):
    """ Manage a suite of verification tests of `wc_sim`\ 's dynamic behavior

    Attributes:
        cases_dir (:obj:`str`): path to cases directory
        plot_dir (:obj:`str`): path to directory of plots
        results (:obj:`list` of :obj:`VerificationRunResult`): a result for each test
    """

    def __init__(self, cases_dir, plot_dir=None):
        if not os.path.isdir(cases_dir):
            raise VerificationError("cannot open cases_dir: '{}'".format(cases_dir))
        self.cases_dir = cases_dir
        if plot_dir and not os.path.isdir(plot_dir):
            raise VerificationError("cannot open plot_dir: '{}'".format(plot_dir))
        self.plot_dir = plot_dir
        self._empty_results()

    def _empty_results(self):
        """ Discard all results of test runs
        """
        self.results = []

    def get_results(self):
        """ Provide results of test runs

        Returns:
            :obj:`list` of :obj:`VerificationRunResult`: results of previous `_run_test()`\ s, in
                execution order
        """
        return self.results

    def _record_result(self, case_type_name, case_num, result_type, duration, **kwargs):
        """ Record the result of a test run

        Args:
            case_type_name (:obj:`str`): the type of case, a name in `VerificationTestCaseType`
            case_num (:obj:`str`): unique id of a verification case
            result_type (:obj:`VerificationResultType`): the result's classification
            duration (:obj:`float`): time it took to run the test (sec)
            in `kwargs`:
            quant_diff (:obj:`str`, optional): quantified difference between actual and expected
            params (:obj:`str`, optional): parameters used by the test, if any
            error (:obj:`str`, optional): description of the error, if any
        """
        if type(result_type) != VerificationResultType:
            raise VerificationError("result_type must be a VerificationResultType, not a '{}'".format(
                                    type(result_type).__name__))
        self.results.append(VerificationRunResult(self.cases_dir, case_type_name, case_num,
                                                  result_type, duration, **kwargs))

    # attributes in a VerificationRunResult to dump in `dump_results`
    RESULTS_ATTRIBUTES_TO_DUMP = ['case_type', 'case_num', 'duration', 'quant_diff']
    def dump_results(self, errors=False):
        """ Provide results of tests run by `_run_test`

        Args:
            errors (:obj:`bool`, optional): if set, provide results that have errors; otherwise,
                provide results that don't have errors

        Returns:
            :obj:`list` of :obj:`dict` of :obj:`obj`: results summarized as depth-one dict for each run
        """
        formatted_results = []
        for result in self.results:
            row = {}
            for attr in self.RESULTS_ATTRIBUTES_TO_DUMP:
                row[attr] = getattr(result, attr)
            row['result_type'] = result.result_type.name
            if result.params:
                params = DictUtil.flatten_dict(result.params)
                for k, v in params.items():
                    row[k] = v

            # if errors is set, just provide results with errors
            if errors and result.error:
                    row['error'] = result.error
                    formatted_results.append(row)

            # if errors is False, just provide results without errors
            if not errors and not result.error:
                # results without errors
                formatted_results.append(row)

        return formatted_results

    def _run_test(self, case_type_name, case_num, num_stochastic_runs=None,
                  ode_time_step_factor=None, rtol=None, atol=None, verbose=False, evaluate=False):
        """ Run one test case and record the result

        The results from running a test case are recorded by calling `self._record_result()`.
        It records results in the list `self.results`. All types of results, including exceptions,
        are recorded.

        Args:
            case_type_name (:obj:`str`): the type of case, a name in `VerificationTestCaseType`
            case_num (:obj:`str`): unique id of a verification case
            num_stochastic_runs (:obj:`int`, optional): number of Monte Carlo runs for an SSA test
            ode_time_step_factor (:obj:`float`, optional): factor by which to change the ODE time step
            rtol (:obj:`float`, optional): relative tolerance for the ODE solver
            atol (:obj:`float`, optional): absolute tolerance for the ODE solver
            verbose (:obj:`bool`, optional): whether to produce verbose output
            evaluate (:obj:`bool`, optional): whether to quantitatively evaluate the test case;
                if `False`, then indicate whether the test passed in the saved result's `result_type`;
                otherwise, provide the mean Z-score divergence in the result's `quant_diff`

        Returns:
            :obj:`None`: results are recorded in `self.results`
        """
        try:
            case_verifier = CaseVerifier(self.cases_dir, case_type_name, case_num)
        except:
            tb = traceback.format_exc()
            self._record_result(case_type_name, case_num, VerificationResultType.CASE_UNREADABLE,
                                float('nan'), error=tb)
            return

        # assemble kwargs
        kwargs = {}

        if num_stochastic_runs is None:
            kwargs['num_discrete_stochastic_runs'] = config_multialgorithm['num_ssa_verification_sim_runs']
        else:
            kwargs['num_discrete_stochastic_runs'] = num_stochastic_runs

        kwargs['ode_time_step_factor'] = ode_time_step_factor
        plot_name_params = [f"ode_{ode_time_step_factor}"]

        if rtol is not None or atol is not None:
            tolerances_dict = {}
            if rtol is not None:
                tolerances_dict['rtol'] = rtol
                plot_name_params.append(f"rtol_{rtol}")
            if atol is not None:
                tolerances_dict['atol'] = atol
                plot_name_params.append(f"atol_{atol}")
            kwargs['tolerances'] = tolerances_dict

        if self.plot_dir:
            plot_name_append = '_'.join(plot_name_params)
            plot_file = os.path.join(self.plot_dir,
                                     f"{case_type_name}_{case_num}_{plot_name_append}_verification_test.pdf")
            kwargs['plot_file'] = plot_file

        if verbose:
            print("Verifying {} case {}".format(case_type_name, case_num))

        try:
            start_time = time.process_time()
            if evaluate:
                kwargs['evaluate'] = True
            verification_result = case_verifier.verify_model(**kwargs)

            run_time = time.process_time() - start_time
            results_kwargs = {}
            if evaluate:
                results_kwargs['quant_diff'] = verification_result
                # since evaluate, don't worry about setting the VerificationRunResult.result_type
                result_type = VerificationResultType.VERIFICATION_UNKNOWN
            else:
                if verification_result:
                    result_type = VerificationResultType.CASE_DID_NOT_VERIFY
                    results_kwargs['error'] = verification_result
                else:
                    result_type = VerificationResultType.CASE_VERIFIED

            self._record_result(case_type_name, case_num, result_type, run_time, params=kwargs,
                                **results_kwargs)

        except Exception as e:
            run_time = time.process_time() - start_time
            tb = traceback.format_exc()
            self._record_result(case_type_name, case_num, VerificationResultType.FAILED_VERIFICATION_RUN,
                                run_time, params=kwargs, error=tb)

    def _run_tests(self, case_type_name, case_num, num_stochastic_runs=None,
                  ode_time_step_factors=None, tolerance_ranges=None, verbose=False,
                  empty_results=False, evaluate=False):
        """ Run one or more tests, possibly iterating over over ODE time step factors and solver tolerances

        Args:
            case_type_name (:obj:`str`): the type of case, a name in `VerificationTestCaseType`
            case_num (:obj:`str`): unique id of a verification case
            num_stochastic_runs (:obj:`int`, optional): number of Monte Carlo runs for an SSA test
            ode_time_step_factors (:obj:`list` of :obj:`float`): factors by which the ODE time step will
                be multiplied
            tolerance_ranges (:obj:`dict`): ranges for absolute and relative ODE tolerances;
                see `default_tolerance_ranges` in `ode_test_generator` for the dict's structure;
                `rtol`, `atol` or both may be provided; configured defaults are used for tolerance(s)
                that are not provided
            verbose (:obj:`bool`, optional): whether to produce verbose output
            empty_results (:obj:`bool`, optional): whether to empty the list of verification run results
            evaluate (:obj:`bool`, optional): whether to quantitatively evaluate the test case(s)

        Returns:
            :obj:`list`: of :obj:`VerificationRunResult`: the results for this :obj:`VerificationSuite`
        """
        if empty_results:
            self._empty_results()
        ode_test_iterator = ODETestIterators.ode_test_generator(ode_time_step_factors=ode_time_step_factors,
                                                                tolerance_ranges=tolerance_ranges)
        for test_kwargs in ode_test_iterator:
            self._run_test(case_type_name, case_num, num_stochastic_runs=num_stochastic_runs,
                           verbose=verbose, evaluate=evaluate, **test_kwargs)
        return self.results

    # default ranges for analyzing ODE solver sensitivity to tolerances
    DEFAULT_MIN_RTOL = 1E-15
    DEFAULT_MAX_RTOL = 1E-3
    DEFAULT_MIN_ATOL = 1E-15
    DEFAULT_MAX_ATOL = 1E-6
    @staticmethod
    def tolerance_ranges_for_sensitivity_analysis():
        """ Get the default tolerance ranges for analyzing ODE solver sensitivity to tolerances

        Returns:
            :obj:`dict`: the default tolerance ranges for sensitivity analysis
        """
        tolerance_ranges = {'rtol': dict(min=VerificationSuite.DEFAULT_MIN_RTOL,
                                         max=VerificationSuite.DEFAULT_MAX_RTOL),
                            'atol': dict(min=VerificationSuite.DEFAULT_MIN_ATOL,
                                         max=VerificationSuite.DEFAULT_MAX_ATOL)}
        return tolerance_ranges

    def run(self, test_case_type_name=None, cases=None, num_stochastic_runs=None,
            ode_time_step_factors=None, tolerance_ranges=None, verbose=False, empty_results=True,
            evaluate=False):
        """ Run all requested test cases

        If `test_case_type_name` is not specified, then all cases for all
        :obj:`VerificationTestCaseType`\ s are verified.
        If `cases` are not specified, then all cases with the specified `test_case_type_name` are
        verified.

        Args:
            test_case_type_name (:obj:`str`, optional): the type of case, a name in
                `VerificationTestCaseType`
            cases (:obj:`list` of :obj:`str`, optional): list of unique ids of verification cases
            num_stochastic_runs (:obj:`int`, optional): number of Monte Carlo runs for an SSA test
            ode_time_step_factors (:obj:`list` of :obj:`float`): factors by which the ODE time step will
                be multiplied
            tolerance_ranges (:obj:`dict`): ranges for absolute and relative ODE tolerances;
                see `default_tolerance_ranges` in `ode_test_generator` for the dict's structure;
                `rtol`, `atol` or both may be provided; configured defaults are used for tolerance(s)
                that are not provided
            verbose (:obj:`bool`, optional): whether to produce verbose output
            empty_results (:obj:`bool`, optional): whether to empty the list of verification run results
            evaluate (:obj:`bool`, optional): whether to quantitatively evaluate the test case(s)

        Returns:
            :obj:`list`: of :obj:`VerificationRunResult`: the results for this :obj:`VerificationSuite`
        """
        if empty_results:
            self._empty_results()
        if isinstance(cases, str):
            raise VerificationError("cases should be an iterator over case nums, not a string")
        if cases and not test_case_type_name:
            raise VerificationError('if cases provided then test_case_type_name must be provided too')
        if test_case_type_name:
            if test_case_type_name not in VerificationTestCaseType.__members__:
                raise VerificationError("Unknown VerificationTestCaseType: '{}'".format(test_case_type_name))
            if cases is None:
                cases = os.listdir(os.path.join(self.cases_dir,
                                                VerificationTestCaseType[test_case_type_name].value))
            for case_num in cases:
                self._run_tests(test_case_type_name, case_num, num_stochastic_runs=num_stochastic_runs,
                                ode_time_step_factors=ode_time_step_factors, tolerance_ranges=tolerance_ranges,
                                verbose=verbose, evaluate=evaluate)
        else:
            for verification_test_case_type in VerificationTestCaseType:
                for case_num in os.listdir(os.path.join(self.cases_dir, verification_test_case_type.value)):
                    self._run_tests(verification_test_case_type.name, case_num,
                                    num_stochastic_runs=num_stochastic_runs,
                                    ode_time_step_factors=ode_time_step_factors,
                                    tolerance_ranges=tolerance_ranges, verbose=verbose, evaluate=evaluate)
        return self.results

    ODE_TIME_STEP_FACTORS = [0.05, 0.1, 1.0]
    def run_multialg(self, cases, ode_time_step_factors=None, tolerances=False, verbose=None,
                     evaluate=True):
        """ Test a suite of multialgorithmic models

        Initial approach:

        * use SBML stochastic models with multiple reactions to test multialgorithmic simulation
        * evaluate correctness using DISCRETE_STOCHASTIC expected results and verification code
        * try various settings for ODE time step, tolerances, etc.

        Args:
            cases (:obj:`list` of :obj:`str`): list of unique ids of verification cases
            ode_time_step_factors (:obj:`list` of :obj:`float`, optional): factors by which the ODE
                time step will be multiplied
            tolerance_ranges (:obj:`dict`): ranges for absolute and relative ODE tolerances;
            verbose (:obj:`bool`, optional): whether to produce verbose output
            evaluate (:obj:`bool`, optional): whether to quantitatively evaluate the test case(s)

        Returns:
            :obj:`list`: of :obj:`VerificationRunResult`: the results for this :obj:`VerificationSuite`
        """
        if ode_time_step_factors is None:
            ode_time_step_factors = self.ODE_TIME_STEP_FACTORS
        tolerance_ranges = None
        if tolerances:
            tolerance_ranges = self.tolerance_ranges_for_sensitivity_analysis()
        return self.run(test_case_type_name=VerificationTestCaseType.MULTIALGORITHMIC.name,
                        cases=cases,
                        num_stochastic_runs=10,
                        ode_time_step_factors=ode_time_step_factors,
                        tolerance_ranges=tolerance_ranges,
                        empty_results=False, verbose=verbose, evaluate=evaluate)


class MultialgModelVerificationFuture(object):    # pragma: no cover
    """
    long-term approach
        input model or pair of equivalent models, a correct model that can be run w SSA, and a hybrid model that also uses ODE
        use the correct model to create a correct SSA verification case:
            make settings file
            correct means & SDs (consider only 1 initial concentration):
                run model Monte Carlo using only SSA
                ** convert into <model_name>-results.csv

        evaluate hybrid simulation of the hybrid model:
            prepare correct results:
                ** convert correct SSA results for species to be modeled only by ODE to a deterministic ODE correct result
                    for these deterministic species, create settings and <model_name_deterministic>-results.csv
                    create a test case for species modeled by SSA or SSA & ODE by removing the deterministic species
            ** run the hybrid model: high flux reactions by ODE, and low flux by SSA
                verify deterministic species against their correct results
                verify hybrid and SSA-only species against their correct results

        verification case dirs needed:
            1) correct SSA: model and settings for running the correct SSA, which will GENERATE its local *.results.csv
            2) hybrid-semantic: model and settings for running the hybrid model, and comparing its ODE predictions
            3) hybrid-stochastic: model and settings for running the hybrid model, and comparing its hybrid&SSA predictions

        if time permits, try various settings of checkpoint interval, time step factor, etc.
    """
    def __init__(self, verification_dir, case_name, ssa_model_file, ssa_settings, hybrid_model_file, hybrid_settings):
        self.verification_dir = verification_dir
        self.case_name = case_name

        '''
        # TODO: add later
        self.continuous_typed_case_verifier = self.TypedCaseVerifier(self.verification_dir, case_name,
            hybrid_model_file, hybrid_settings, 'semantic')
        '''
        self.discrete_typed_case_verifier = self.TypedCaseVerifier(self.verification_dir, case_name,
            hybrid_model_file, hybrid_settings, 'MULTIALGORITHMIC')
        self.correct_typed_case_verifier = self.TypedCaseVerifier(self.verification_dir, case_name,
            ssa_model_file, ssa_settings, 'MULTIALGORITHMIC', correct=True)
        self.tmp_results_dir = tempfile.mkdtemp()

    class TypedCaseVerifier(object):
        # represent a verification case in a MultialgModelVerificationFuture, one of correct,
        # deterministic (ODE) or discrete (SSA)
        def __init__(self, root_dir, case_name, model_file, settings, case_type, correct=False):
            if case_type not in VerificationTestCaseType.__members__:
                raise MultialgorithmError("bad case_type: '{}'".format(case_type))
            # create a special 'correct' sub-dir
            if correct:
                root_dir = os.path.join(root_dir, 'correct')
            self.root_dir = root_dir
            self.case_name = case_name
            # create directory
            # follow directory structure: root dir, type, case num (name)
            self.case_dir = os.path.join(root_dir, VerificationTestCaseType[case_type], case_name)
            os.makedirs(self.case_dir, exist_ok=True)
            # copy in model file, renamed as 'case_name-wc_lang.xlsx
            self.model_file = os.path.join(self.case_dir, "{}-wc_lang.xlsx".format(case_name))
            shutil.copyfile(model_file, self.model_file)
            # read model
            self.model = Reader().run(self.model_file, strict=False)
            # create settings file
            self.write_settings_file(settings)
            # read settings
            self.verification_test_reader = VerificationTestReader(case_type, self.case_dir, case_name)
            self.verification_test_reader.settings = self.verification_test_reader.read_settings()
            # create results csv from run results
            # run verification
            if case_type == VerificationTestCaseType.DISCRETE_STOCHASTIC:
                pass

            if case_type == VerificationTestCaseType.CONTINUOUS_DETERMINISTIC:
                pass

        def get_typed_case_dir(self):
            return self.case_dir

        def get_model(self):
            return self.model

        def get_settings(self):
            return self.verification_test_reader.settings

        def get_simul_kwargs(self):
            simul_kwargs = dict(time_max=self.get_settings()['duration'],
                checkpoint_period=self.get_settings()['duration']/self.get_settings()['steps'],
                verbose=False)
            return simul_kwargs

        def write_settings_file(self, settings):
            """ given settings in a dict, write it to a file
            Args:
                settings (:obj:`dict`): settings
            """
            settings_file = os.path.join(self.case_dir, '{}-settings.txt'.format(self.case_name))
            with open(settings_file, 'w') as settings_fd:
                for key, value in settings.items():
                    settings_fd.write("{}: {}\n".format(key, value))
            return settings_file

        def __str__(self):
            pass

    # building blocks
    def convert_correct_results(self, run_results):
        """ Convert set of simulation results into a stochastic results csv

        Args:
            run_results (:obj:`list` of `RunResult`):
        """

        # allocate results mean & SD
        settings = self.correct_typed_case_verifier.get_settings()
        pop_means = {}
        pop_sds = {}
        for species_type in settings['amount']:
            # todo: fix: obtain species_id
            pop_means[species_type], pop_sds[species_type] = SsaEnsemble.results_mean_n_sd(run_results, species_id)

        # construct dataframe
        times = run_results[0].get_times()
        species_type_stat = []
        for species_type in settings['amount']:
            for stat in ['mean', 'sd']:
                species_type_stat.append("{}-{}".format(species_type, stat))
        correct_results_df = pd.DataFrame(index=times, columns=species_type_stat, dtype=np.float64)
        correct_results_df.index.name = 'time'
        for species_type in settings['amount']:
            for stat in ['mean', 'sd']:
                column = "{}-{}".format(species_type, stat)
                if stat == 'mean':
                    correct_results_df[column] = pop_means[species_type]
                if stat == 'sd':
                    correct_results_df[column] = pop_sds[species_type]

        # output as csv
        results_filename = os.path.join(self.correct_typed_case_verifier.get_typed_case_dir(),
            "{}-results.csv".format(self.case_name))
        correct_results_df.to_csv(results_filename)
        return results_filename

    def get_correct_results(self, num_runs, verbose=True):
        # get correct results from ssa model
        model = self.correct_typed_case_verifier.get_model()
        correct_run_results = SsaEnsemble.run(model,
                                              self.correct_typed_case_verifier.get_simul_kwargs(),
                                              self.tmp_results_dir, num_runs)
        if verbose:
            print("made {} runs of {}".format(len(correct_run_results), model.id))
        return correct_run_results

    def run(self, num_runs=200, make_correct_predictions=False):
        """ Evaluate hybrid simulation of the hybrid model
        """
        if make_correct_predictions:
            correct_results = self.get_correct_results(num_runs)
            self.convert_correct_results(correct_results)

        '''
        set up verification
            set up SSA run (break SSA execution above into reusable method)
            execute correct SSA run to generate SSA ensemble
            Convert set of simulation results into a stochastic results
            # convert ODE only species in stochastic results into hybrid-semantic results
            convert SSA species in stochastic results into hybrid-stochastic results
            copy models and settings into 'hybrid-semantic' and 'hybrid-stochastic' dirs
        run verification
            generate ensemble of hybrid model runs
            verify
            # filter results to ODE and hybrid/SSA
            # verify each
        '''


class VerificationUtilities(object):

    @staticmethod
    def get_default_args(func):
        """ Get the names and default values of function's keyword arguments

        From https://stackoverflow.com/a/12627202

        Args:
            func (:obj:`Function`): a Python function

        Returns:
            :obj:`dict`: a map: name -> default value
        """
        signature = inspect.signature(func)
        return {
            k: v.default
            for k, v in signature.parameters.items()
            if v.default is not inspect.Parameter.empty
        }
