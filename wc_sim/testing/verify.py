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
from pprint import pprint
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

from wc_lang.core import ReactionParticipantAttribute
from wc_lang.io import Reader
from wc_sim.config import core as config_core_multialgorithm
from wc_sim.multialgorithm_errors import MultialgorithmError
from wc_sim.run_results import RunResults
from wc_sim.simulation import Simulation
from wc_sim.testing.make_models import MakeModel

config_multialgorithm = config_core_multialgorithm.get_config()['wc_sim']['multialgorithm']


# TODO: doc strings
class Error(Exception):
    """ Base class for exceptions involving `wc_sim` validation

    Attributes:
        message (:obj:`str`): the exception's message
    """
    def __init__(self, message=None):
        super().__init__(message)


class ValidationError(Error):
    """ Exception raised for errors in `wc_sim.validate`

    Attributes:
        message (:obj:`str`): the exception's message
    """
    def __init__(self, message=None):
        super().__init__(message)


class WcSimValidationWarning(UserWarning):
    """ `wc_sim` Validation warning """
    pass


class ValidationTestCaseType(Enum):
    """ Types of test cases """
    CONTINUOUS_DETERMINISTIC = 1    # algorithms like ODE
    DISCRETE_STOCHASTIC = 2         # algorithms like SSA

# TODO: make the dirs the values of the Enum and get rid of TEST_CASE_TYPE_TO_DIR
TEST_CASE_TYPE_TO_DIR = {
    'CONTINUOUS_DETERMINISTIC': 'semantic',
    'DISCRETE_STOCHASTIC': 'stochastic'}


class ValidationTestReader(object):
    """ Read a model validation test case """
    SBML_FILE_SUFFIX = '.xml'
    def __init__(self, test_case_type, test_case_dir, test_case_num):
        if test_case_type not in ValidationTestCaseType.__members__:
            raise ValidationError("Unknown ValidationTestCaseType: '{}'".format(test_case_type))
        else:
            self.test_case_type = ValidationTestCaseType[test_case_type]
        self.test_case_dir = test_case_dir
        self.test_case_num = test_case_num

    def read_settings(self):
        """ Read a test case's settings into a key-value dictionary """
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
            raise ValidationError('; '.join(errors))

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
        self.expected_predictions_file = expected_predictions_file = os.path.join(
            self.test_case_dir, self.test_case_num+'-results.csv')
        expected_predictions_df = pd.read_csv(expected_predictions_file)
        # expected predictions should contain data for all time steps
        times = np.linspace(self.settings['start'], self.settings['duration'], num=self.settings['steps']+1)
        if not np.allclose(times, expected_predictions_df.time):
            raise ValidationError("times in settings '{}' differ from times in expected predictions '{}'".format(
                self.settings_file, expected_predictions_file))

        if self.test_case_type == ValidationTestCaseType.CONTINUOUS_DETERMINISTIC:
            # expected predictions should contain the mean and sd of each variable in 'amount'
            expected_columns = set(self.settings['amount'])
            actual_columns = set(expected_predictions_df.columns.values)
            if expected_columns - actual_columns:
                raise ValidationError("some amounts missing from expected predictions '{}': {}".format(
                    expected_predictions_file, expected_columns - actual_columns))

        if self.test_case_type == ValidationTestCaseType.DISCRETE_STOCHASTIC:
            # expected predictions should contain the mean and sd of each variable in 'amount'
            expected_columns = set()
            for amount in self.settings['amount']:
                expected_columns.add(amount+'-mean')
                expected_columns.add(amount+'-sd')
            if expected_columns - set(expected_predictions_df.columns.values):
                raise ValidationError("mean or sd of some amounts missing from expected predictions '{}': {}".format(
                    expected_predictions_file, expected_columns - set(expected_predictions_df.columns.values)))

        return expected_predictions_df

    def read_model(self):
        """  Read a model into a `wc_lang` representation. """
        self.model_filename = model_filename = os.path.join(
            self.test_case_dir, self.test_case_num+'-wc_lang.xlsx')
        if model_filename.endswith(self.SBML_FILE_SUFFIX):   # pragma: no cover
            raise ValidationError("Reading SBML files not supported: model filename '{}'".format(model_filename))
        return Reader().run(self.model_filename, strict=False)

    def run(self):
        self.settings = self.read_settings()
        self.expected_predictions_df = self.read_expected_predictions()
        self.model = self.read_model()


# the compartment for test cases
TEST_CASE_COMPARTMENT = 'c'

class ResultsComparator(object):
    """ Compare simulated and expected predictions """
    TOLERANCE_MAP = dict(
        rtol='relative',
        atol='absolute'
    )

    def __init__(self, validation_test_reader, simulation_run_results):
        self.validation_test_reader = validation_test_reader
        self.strip_compartments(simulation_run_results)
        self.simulation_run_results = simulation_run_results
        # obtain default tolerances in np.allclose()
        self.default_tolerances = ValidationUtilities.get_default_args(np.allclose)

    @staticmethod
    def get_species(species_type):
        return "{}[{}]".format(species_type, TEST_CASE_COMPARTMENT)

    @staticmethod
    def get_species_type(string):
        # if string is a species get the species type, otherwise return the string
        if re.match(r'\w+\[\w+\]$', string, flags=re.ASCII):
            return string.split('[')[0]
        return string

    @classmethod
    def strip_compartments(cls, simulation_run_results):
        # if they're present, strip compartments from simulation run results; assumes all validation tests happen in one compartment
        # note that original run results are changed; assumes that is OK because they're used just for validation
        if isinstance(simulation_run_results, RunResults):
            populations_df = simulation_run_results.get('populations')
            col_names = list(populations_df.columns)
            populations_df.columns = list(map(cls.get_species_type, col_names))
        elif isinstance(simulation_run_results, list):
            for run_result in simulation_run_results:
                cls.strip_compartments(run_result)
        else:
            raise ValidationError("wrong type for simulation_run_results '{}'".format(
                type(simulation_run_results)))

    def prepare_tolerances(self):
        """ Prepare tolerance dictionary

        Use values from `validation_test_reader.settings` if available, otherwise from `numpy.allclose()`s defaults

        Returns:
            :obj:`dict`: kwargs for `rtol` and `atol` tolerances for use by `numpy.allclose()`
        """
        kwargs = {}
        for np_name, testing_name in self.TOLERANCE_MAP.items():
            kwargs[np_name] = self.default_tolerances[np_name]
            if testing_name in self.validation_test_reader.settings:
                try:
                    tolerance = float(self.validation_test_reader.settings[testing_name])
                    kwargs[np_name] = tolerance
                except:
                    pass
        return kwargs

    @staticmethod
    def zero_to_inf(np_array):
        """ replace 0s with inf """
        infs = np.full(np_array.shape, float('inf'))
        return np.where(np_array != 0, np_array, infs)

    def differs(self):
        """ Evaluate whether simulation runs(s) differ from their expected species population prediction(s)

        Returns:
            :obj:`obj`: `False` if populations in the expected result and simulation run are equal
                within tolerances, otherwise :obj:`list`: of species with differing values
        """
        differing_values = []
        if self.validation_test_reader.test_case_type == ValidationTestCaseType.CONTINUOUS_DETERMINISTIC:
            kwargs = self.prepare_tolerances()
            concentrations_df = self.simulation_run_results.get('concentrations')
            # for each prediction, determine whether its trajectory is close enough to the expected predictions
            for species_type in self.validation_test_reader.settings['amount']:
                if not np.allclose(self.validation_test_reader.expected_predictions_df[species_type].values,
                    concentrations_df[species_type].values, **kwargs):
                    differing_values.append(species_type)
            return differing_values or False

        if self.validation_test_reader.test_case_type == ValidationTestCaseType.DISCRETE_STOCHASTIC:
            """ Test mean and sd population over multiple runs

            Follow algorithm in
            github.com/sbmlteam/sbml-test-suite/blob/master/cases/stochastic/DSMTS-userguide-31v2.pdf,
            from Evans, et al. The SBML discrete stochastic models test suite, Bioinformatics, 24:285-286, 2008.
            """
            # TODO: warn if values lack precision; want int64 integers and float64 floats
            # see https://docs.scipy.org/doc/numpy-1.15.0/reference/arrays.scalars.html
            # use warnings.warn("", WcSimValidationWarning)

            ### test means ###
            mean_range = self.validation_test_reader.settings['meanRange']
            self.n_runs = n_runs = len(self.simulation_run_results)

            self.simulation_pop_means = {}
            for species_type in self.validation_test_reader.settings['amount']:
                # extract nx1 correct mean and sd np arrays
                correct_df = self.validation_test_reader.expected_predictions_df
                e_mean = correct_df.loc[:, species_type+'-mean'].values
                e_sd = correct_df.loc[:, species_type+'-sd'].values
                # errors if e_sd or e_mean < 0
                if np.any(e_mean < 0):
                    raise ValidationError("e_mean contains negative value(s)")
                if np.any(e_sd < 0):
                    raise ValidationError("e_sd contains negative value(s)")

                # avoid division by 0 and sd=0; replace 0s in e_sd with inf
                # TODO: instead of this, remove 0s in e_sd and corresponding pop_mean & e_mean rows;
                # if too many removed, raise error
                e_sd = self.zero_to_inf(e_sd)

                # load simul. runs into 2D np array to find mean and SD
                pop_mean, pop_sd = SsaEnsemble.results_mean_n_sd(self.simulation_run_results, species_type)
                self.simulation_pop_means[species_type] = pop_mean
                Z = math.sqrt(n_runs) * (pop_mean - e_mean) / e_sd

                # compare with mean_range
                if np.any(Z < mean_range[0]) or np.any(mean_range[1] < Z):
                    differing_values.append(species_type)

            ### test sds ###
            # TODO: test sds
            return differing_values or False


class SsaEnsemble(object):
    # handle an SSA ensemble

    @staticmethod
    def run(model, simul_kwargs, tmp_results_dir, num_runs):
        simulation_run_results = []
        progress_factor = 1
        for i in range(num_runs):
            if i == progress_factor:
                print("starting run {} of {} of model {}".format(i, num_runs, model.id))
                progress_factor *= 2
            simul_kwargs['results_dir'] = tempfile.mkdtemp(dir=tmp_results_dir)
            # TODO: provide method(s) in Simulation and classes it uses (SimulationEngine) to reload() a simulation,
            # that is, do another monte Carlo simulation with a different seed
            simulation = Simulation(model)
            _, results_dir = simulation.run(**simul_kwargs)
            simulation_run_results.append(RunResults(results_dir))
        return simulation_run_results

    @staticmethod
    def results_mean_n_sd(simulation_run_results, species_type):
        times = simulation_run_results[0].get_times()
        n_times = len(times)
        n_runs = len(simulation_run_results)
        run_results_array = np.empty((n_times, n_runs))
        for idx, run_result in enumerate(simulation_run_results):
            run_pop_df = run_result.get('populations')
            run_results_array[:, idx] = run_pop_df.loc[:, species_type].values
        # take mean & sd at each time
        return run_results_array.mean(axis=1), run_results_array.std(axis=1)


class CaseValidator(object):
    """ Validate a test case """
    def __init__(self, test_cases_root_dir, test_case_type, test_case_num,
        default_num_stochastic_runs=config_multialgorithm['num_ssa_validation_sim_runs'],
        time_step_factor=None):
        # read model, config and expected predictions
        self.test_case_dir = os.path.join(test_cases_root_dir, TEST_CASE_TYPE_TO_DIR[test_case_type],
            test_case_num)
        self.validation_test_reader = ValidationTestReader(test_case_type, self.test_case_dir, test_case_num)
        self.validation_test_reader.run()
        self.default_num_stochastic_runs = default_num_stochastic_runs
        self.time_step_factor = time_step_factor

    def validate_model(self, num_discrete_stochastic_runs=None, discard_run_results=True, plot_file=None):
        """ Validate a model
        """
        # check settings
        required_settings = ['duration', 'steps']
        settings = self.validation_test_reader.settings
        errors = []
        for setting in required_settings:
            if setting not in settings:
                errors.append("required setting '{}' not provided".format(setting))
            elif not isinstance(settings[setting], float):
                errors.append("required setting '{}' not a float".format(setting))
        if errors:
            raise ValidationError('; '.join(errors))
        if 'start' in settings and settings['start'] != 0:
            raise ValidationError("non-zero start setting ({}) not supported".format(settings['start']))

        # prepare for simulation
        self.tmp_results_dir = tmp_results_dir = tempfile.mkdtemp()
        simul_kwargs = dict(end_time=settings['duration'],
            checkpoint_period=settings['duration']/settings['steps'],
            results_dir=tmp_results_dir,
            verbose= False)

        ## 1. run simulation
        factor = 1
        if self.time_step_factor is not None:
            factor = self.time_step_factor
        simul_kwargs['time_step'] = factor * settings['duration']/settings['steps']

        if self.validation_test_reader.test_case_type == ValidationTestCaseType.CONTINUOUS_DETERMINISTIC:
            simulation = Simulation(self.validation_test_reader.model)
            _, results_dir = simulation.run(**simul_kwargs)
            self.simulation_run_results = RunResults(results_dir)

        if self.validation_test_reader.test_case_type == ValidationTestCaseType.DISCRETE_STOCHASTIC:
            '''
            TODO: retry on failure
                if failure, retry "evaluate whether mean of simulation trajectories match expected trajectory"
                    # simulations generating the correct trajectories will fail validation (100*(p-value threshold)) percent of the time
                if failure again, report failure    # assuming p-value << 1, two failures indicates likely errors
            '''
            # TODO: convert to probabilistic test with multiple runs and p-value
            # make multiple simulation runs with different random seeds
            if num_discrete_stochastic_runs is not None:
                num_runs = num_discrete_stochastic_runs
            else:
                num_runs = self.default_num_stochastic_runs
            self.num_runs = num_runs
            self.simulation_run_results = \
                SsaEnsemble.run(self.validation_test_reader.model, simul_kwargs, tmp_results_dir, num_runs)

        ## 2. compare results
        self.results_comparator = ResultsComparator(self.validation_test_reader, self.simulation_run_results)
        self.comparison_result = self.results_comparator.differs()

        ## 3 plot comparison of actual and expected trajectories
        if plot_file:
            self.plot_model_validation(plot_file)
        # TODO: optionally, save results
        # TODO: output difference between actual and expected trajectory

        ## 4. cleanup
        if discard_run_results:
            shutil.rmtree(self.tmp_results_dir)
        return self.comparison_result

    def get_model_summary(self):
        """Summarize the test model
        """
        mdl = self.validation_test_reader.model
        summary = ['Model Summary:']
        summary.append("model {}:\n    {}".format(mdl.id, mdl.name))
        for cmpt in mdl.compartments:
            summary.append("compartment {}: {}, init. vol. {}, type {}".format(cmpt.id, cmpt.name,
                cmpt.initial_volume, cmpt.biological_type))
        reaction_participant_attribute = ReactionParticipantAttribute()
        for sm in mdl.submodels:
            summary.append("submodel {}: in {}".format(sm.id, sm.compartment.id))
            for rxn in sm.reactions:
                summary.append("reaction and rate law {}: {}, {}".format(rxn.id,
                    reaction_participant_attribute.serialize(rxn.participants),
                    rxn.rate_laws[0].equation.serialize()))
        for param in mdl.get_parameters():
            summary.append("param: {}={} ({})".format(param.id, param.value, param.units))
        return summary

    def get_test_case_summary(self):
        """Summarize the test case
        """
        # TODO: include # events, run time, perhaps details on error of failed tests
        summary = ['Test Case Summary']
        if self.comparison_result:
            summary.append("Failing species types: {}".format(', '.join(self.comparison_result)))
        else:
            summary.append('All species types validate')
        if hasattr(self, 'num_runs'):
            summary.append("Num simul runs: {}".format(self.num_runs))
        summary.append("Test case type: {}".format(self.validation_test_reader.test_case_type.name))
        summary.append("Test case number: {}".format(self.validation_test_reader.test_case_num))
        summary.append("Time step factor: {}".format(self.time_step_factor))
        return summary

    def plot_model_validation(self, plot_file, max_runs_to_plot=100, presentation_qual=True):
        """Plot a model validation run
        """
        # TODO: configure max_runs_to_plot in config file
        # TODO: use matplotlib 3; use the more flexible OO API instead of pyplot
        num_species_types = len(self.validation_test_reader.settings['amount'])
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
            print('cannot plot more than 4 species_types')
            return
        legend_fontsize = 9 if presentation_qual else 5
        plot_num = 1
        if self.validation_test_reader.test_case_type == ValidationTestCaseType.CONTINUOUS_DETERMINISTIC:
            times = self.simulation_run_results.get('concentrations').index
            for species_type in self.validation_test_reader.settings['amount']:
                plt.subplot(n_rows, n_cols, plot_num)

                # plot expected predictions
                expected_kwargs = dict(color='red', linewidth=1.2)
                expected_mean_df = self.validation_test_reader.expected_predictions_df.loc[:, species_type]
                correct_mean, = plt.plot(times, expected_mean_df.values, **expected_kwargs)

                # plot simulation pops
                # plot second, so appears on top, and narrower width so expected shows
                pop_time_series = self.simulation_run_results.get('concentrations').loc[:, species_type]
                simul_pops, = plt.plot(times, pop_time_series, 'b-', linewidth=0.8)

                plt.ylabel('{} (M)'.format(species_type), fontsize=10)
                plt.xlabel('time (s)', fontsize=10)
                plt.legend((simul_pops, correct_mean),
                    ('{} simulation run'.format(species_type), 'Correct concentration'),
                    loc='lower left', fontsize=legend_fontsize)
                plot_num += 1

        if self.validation_test_reader.test_case_type == ValidationTestCaseType.DISCRETE_STOCHASTIC:
            times = self.simulation_run_results[0].get('populations').index

            for species_type in self.validation_test_reader.settings['amount']:
                plt.subplot(n_rows, n_cols, plot_num)

                # plot simulation pops
                # TODO: try making individual runs visible by slightly varying color and/or width
                for rr in self.simulation_run_results[:max_runs_to_plot]:
                    pop_time_series = rr.get('populations').loc[:, species_type]
                    simul_pops, = plt.plot(times, pop_time_series, 'b-', linewidth=0.1)

                # plot expected predictions
                expected_kwargs = dict(color='red', linewidth=1)
                expected_mean_df = self.validation_test_reader.expected_predictions_df.loc[:, species_type+'-mean']
                correct_mean, = plt.plot(times, expected_mean_df.values, **expected_kwargs)
                # mean +/- 3 sd
                expected_kwargs['linestyle'] = 'dashed'
                expected_sd_df = self.validation_test_reader.expected_predictions_df.loc[:, species_type+'-sd']
                # TODO: take range -3, +3 should be taken from settings data
                correct_mean_plus_3sd, = plt.plot(times, expected_mean_df.values + 3 * expected_sd_df / math.sqrt(self.results_comparator.n_runs),
                    **expected_kwargs)
                correct_mean_minus_3sd, = plt.plot(times, expected_mean_df.values - 3 * expected_sd_df / math.sqrt(self.results_comparator.n_runs),
                    **expected_kwargs)

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
                        'Mean of {} {} runs'.format(num_runs, species_type),
                        'Correct mean', '+/- 3Z thresholds'),
                    loc='lower left', fontsize=legend_fontsize)
                plot_num += 1

        summary = self.get_model_summary()
        middle = len(summary)//2
        x_pos = 0.1
        y_pos = 0.9
        for lb, ub in [(0, middle), (middle, len(summary))]:
            if not presentation_qual:
                text = plt.figtext(x_pos, y_pos, '\n'.join(summary[lb:ub]), fontsize=4)
            # TODO: position text automatically
            x_pos += 0.35
        test_case_summary = self.get_test_case_summary()
        if not presentation_qual:
            plt.figtext(0.8, y_pos, '\n'.join(test_case_summary), fontsize=4)

        if presentation_qual:
            test_reader = self.validation_test_reader
            suptitle = "{} case {}, '{}': {}".format(
                test_reader.test_case_type.name.replace('_', ' ').title(),
                test_reader.test_case_num,
                self.validation_test_reader.model.name,
                'failed' if self.comparison_result else 'validated'
                )
            plt.suptitle(suptitle, fontsize=10)
            # plt.title('')
        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig = plt.gcf()
        fig.savefig(plot_file)
        plt.close(fig)
        return "Wrote: {}".format(plot_file)


class ValidationResultType(Enum):
    """ Types of validation results """
    CASE_UNREADABLE = 'could not read case'
    FAILED_VALIDATION_RUN = 'validation run failed'
    SLOW_VALIDATION_RUN = 'validation run timed out'
    CASE_DID_NOT_VALIDATE = 'case did not validate'
    CASE_VALIDATED = 'case validated'


ValidationRunResult = namedtuple('ValidationRunResult', 'cases_dir, case_type_sub_dir, case_num, result_type, error')
# make dynamic_expression optional: see https://stackoverflow.com/a/18348004
ValidationRunResult.__new__.__defaults__ = (None, )
# TODO: add doc strings, like
# ValidationRunResult.__doc__ += ': directory storing all test cases'

class ValidationSuite(object):
    """Run suite of validation tests of `wc_sim`'s dynamic behavior """

    def __init__(self, cases_dir, plot_dir=None):
        if not os.path.isdir(cases_dir):
            raise ValidationError("cannot open cases_dir: '{}'".format(cases_dir))
        self.cases_dir = cases_dir
        if plot_dir and not os.path.isdir(plot_dir):
            raise ValidationError("cannot open plot_dir: '{}'".format(plot_dir))
        self.plot_dir = plot_dir
        self._reset_results()

    def _reset_results(self):
        self.results = []

    def get_results(self):
        return self.results

    def _record_result(self, case_type_sub_dir, case_num, result_type, error=None):
        """Record a result_type
        """
        # TODO: if an error occurs record more, like the results dir
        if type(result_type) != ValidationResultType:
            raise ValidationError("result_type must be a ValidationResultType, not a '{}'".format(type(result_type).__name__))
        if error:
            self.results.append(ValidationRunResult(self.cases_dir, case_type_sub_dir, case_num, result_type, error))
        else:
            self.results.append(ValidationRunResult(self.cases_dir, case_type_sub_dir, case_num, result_type))

    def _run_test(self, case_type_name, case_num, num_stochastic_runs=20, time_step_factor=None,
        verbose=False):
        """Run one test case and report the result
        """
        if verbose:
            start_time = time.process_time()
        try:
            case_validator = CaseValidator(self.cases_dir, case_type_name, case_num,
                time_step_factor=time_step_factor)
            if verbose:
                print(MakeModel.model_to_str(case_validator.validation_test_reader.model))
        except:
            tb = traceback.format_exc()
            self._record_result(case_type_name, case_num, ValidationResultType.CASE_UNREADABLE, tb)
            return
        try:
            kwargs = {}
            if self.plot_dir:
                plot_file = os.path.join(self.plot_dir, "{}_{}_validation_test.pdf".format(case_type_name, case_num))
                kwargs['plot_file'] = plot_file
            if num_stochastic_runs:
                kwargs['num_discrete_stochastic_runs'] = num_stochastic_runs
            # TODO: timeout excessively long validation runs
            if verbose:
                notice = "Validating {} case {}".format(case_type_name, case_num)
                if num_stochastic_runs is not None:
                    notice += " with {} runs".format(num_stochastic_runs)
                print(notice)
            validation_result = case_validator.validate_model(**kwargs)
            if verbose:
                run_time = time.process_time() - start_time
                print("run time: {:8.3f}".format(run_time))

        except:
            tb = traceback.format_exc()
            self._record_result(case_type_name, case_num, ValidationResultType.FAILED_VALIDATION_RUN, tb)
            return
        if validation_result:
            self._record_result(case_type_name, case_num, ValidationResultType.CASE_DID_NOT_VALIDATE, validation_result)
        else:
            self._record_result(case_type_name, case_num, ValidationResultType.CASE_VALIDATED)

    def run(self, test_case_type_name=None, cases=None, num_stochastic_runs=None, time_step_factor=None,
        verbose=False):
        """Run all requested test cases
        """
        if isinstance(cases, str):
            raise ValidationError("cases should be an iterator over case nums, not a string")
        if cases and not test_case_type_name:
            raise ValidationError('if cases provided then test_case_type_name must be provided too')
        if test_case_type_name:
            if test_case_type_name not in ValidationTestCaseType.__members__:
                raise ValidationError("Unknown ValidationTestCaseType: '{}'".format(test_case_type_name))
            if cases is None:
                cases = os.listdir(os.path.join(self.cases_dir, TEST_CASE_TYPE_TO_DIR[test_case_type_name]))
            for case_num in cases:
                self._run_test(test_case_type_name, case_num, num_stochastic_runs=num_stochastic_runs,
                    time_step_factor=time_step_factor, verbose=verbose)
        else:
            for validation_test_case_type in ValidationTestCaseType:
                for case_num in os.listdir(os.path.join(self.cases_dir, TEST_CASE_TYPE_TO_DIR[validation_test_case_type.name])):
                    self._run_test(validation_test_case_type.name, case_num,
                        num_stochastic_runs=num_stochastic_runs, time_step_factor=time_step_factor,
                        verbose=verbose)
        return self.results


class HybridModelValidation(object):
    """
    approach
        input model or pair of equivalent models, a correct model that can be run w SSA, and a hybrid model that also uses ODE
        use the correct model to create a correct SSA validation case:
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
                validate deterministic species against their correct results
                validate hybrid and SSA-only species against their correct results
            
        if time permits, try various settings of checkpoint interval, time step factor, etc.
    """
    '''
    validation case dirs needed:
        1) correct SSA: model and settings for running the correct SSA, which will GENERATE its local *.results.csv
        2) hybrid-semantic: model and settings for running the hybrid model, and comparing its ODE predictions
        3) hybrid-stochastic: model and settings for running the hybrid model, and comparing its hybrid&SSA predictions
    '''
    def __init__(self, validation_dir, case_name, ssa_model_file, ssa_settings, hybrid_model_file, hybrid_settings):
        self.validation_dir = validation_dir
        self.case_name = case_name

        '''
        # TODO: add later
        self.continuous_typed_case_validator = self.TypedCaseValidator(self.validation_dir, case_name,
            hybrid_model_file, hybrid_settings, 'semantic')
        '''
        self.discrete_typed_case_validator = self.TypedCaseValidator(self.validation_dir, case_name,
            hybrid_model_file, hybrid_settings, 'DISCRETE_STOCHASTIC')
        self.correct_typed_case_validator = self.TypedCaseValidator(self.validation_dir, case_name,
            ssa_model_file, ssa_settings, 'DISCRETE_STOCHASTIC', correct=True)
        self.tmp_results_dir = tempfile.mkdtemp()

    class TypedCaseValidator(object):
        # represent a validation case in a HybridModelValidation, one of correct, deterministic (ODE) or discrete (SSA) 
        def __init__(self, root_dir, case_name, model_file, settings, case_type, correct=False):
            if case_type not in TEST_CASE_TYPE_TO_DIR:
                raise MultialgorithmError("bad case_type: '{}'".format(case_type))
            # create a special 'correct' sub-dir
            if correct:
                root_dir = os.path.join(root_dir, 'correct')
            self.root_dir = root_dir
            self.case_name = case_name
            # create directory
            # follow directory structure: root dir, type, case num (name)
            self.case_dir = os.path.join(root_dir, TEST_CASE_TYPE_TO_DIR[case_type], case_name)
            os.makedirs(self.case_dir, exist_ok=True)
            # copy in model file, renamed as 'case_name-wc_lang.xlsx
            self.model_file = os.path.join(self.case_dir, "{}-wc_lang.xlsx".format(case_name))
            shutil.copyfile(model_file, self.model_file)
            # read model
            self.model = Reader().run(self.model_file, strict=False)
            # create settings file
            self.write_settings_file(settings)
            # read settings
            self.validation_test_reader = ValidationTestReader(case_type, self.case_dir, case_name)
            self.validation_test_reader.settings = self.validation_test_reader.read_settings()
            # create results csv from run results
            # run validation
            if case_type == ValidationTestCaseType.DISCRETE_STOCHASTIC:
                pass

            if case_type == ValidationTestCaseType.CONTINUOUS_DETERMINISTIC:
                pass

        def get_typed_case_dir(self):
            return self.case_dir

        def get_model(self):
            return self.model

        def get_settings(self):
            return self.validation_test_reader.settings

        def get_simul_kwargs(self):
            simul_kwargs = dict(end_time=self.get_settings()['duration'],
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
        settings = self.correct_typed_case_validator.get_settings()
        pop_means = {}
        pop_sds = {}
        for species_type in settings['amount']:
            pop_means[species_type], pop_sds[species_type] = SsaEnsemble.results_mean_n_sd(run_results, species_type)

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
        results_filename = os.path.join(self.correct_typed_case_validator.get_typed_case_dir(),
            "{}-results.csv".format(self.case_name))
        correct_results_df.to_csv(results_filename)
        return results_filename

    def get_correct_results(self, num_runs, verbose=True):
        # get correct results from ssa model
        model = self.correct_typed_case_validator.get_model()
        correct_run_results = \
            SsaEnsemble.run(model,
                self.correct_typed_case_validator.get_simul_kwargs(), self.tmp_results_dir, num_runs)
        if verbose:
            print("made {} runs of {}".format(len(correct_run_results), model.id))
        ResultsComparator.strip_compartments(correct_run_results)
        return correct_run_results

    def run(self, num_runs=200, make_correct_predictions=False):
        """ Evaluate hybrid simulation of the hybrid model
        """
        if make_correct_predictions:
            correct_results = self.get_correct_results(num_runs)
            self.convert_correct_results(correct_results)

        ''' 
        set up validation
            set up SSA run (break SSA execution above into reusable method)
            execute correct SSA run to generate SSA ensemble
            Convert set of simulation results into a stochastic results
            # convert ODE only species in stochastic results into hybrid-semantic results
            convert SSA species in stochastic results into hybrid-stochastic results
            copy models and settings into 'hybrid-semantic' and 'hybrid-stochastic' dirs
        run validation
            generate ensemble of hybrid model runs
            validate
            # filter results to ODE and hybrid/SSA
            # validate each
        ''' 


class ValidationUtilities(object):

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
