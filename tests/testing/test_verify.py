"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-09-25
:Copyright: 2018, Karr Lab
:License: MIT
"""

from collections import namedtuple
from inspect import currentframe, getframeinfo
import cProfile
import datetime
import math
import numpy as np
import os
import pandas
import pstats
import shutil
import tempfile
import unittest

from wc_sim.run_results import RunResults
from wc_sim.testing.verify import (ValidationError, ValidationTestCaseType, ValidationTestReader,
                                   ResultsComparator, ResultsComparator, CaseValidator, ValidationResultType,
                                   ValidationSuite, ValidationUtilities, HybridModelValidation,
                                  ValidationRunResult, TEST_CASE_TYPE_TO_DIR, TEST_CASE_COMPARTMENT)
import obj_tables
import wc_sim.submodels.odes as odes

TEST_CASES = os.path.join(os.path.dirname(__file__), 'fixtures', 'validation', 'testing')

def make_test_case_dir(test_case_num, test_case_type='DISCRETE_STOCHASTIC'):
    return os.path.join(TEST_CASES, TEST_CASE_TYPE_TO_DIR[test_case_type], test_case_num)

def make_validation_test_reader(test_case_num, test_case_type='DISCRETE_STOCHASTIC'):
    return ValidationTestReader(test_case_type, make_test_case_dir(test_case_num, test_case_type), test_case_num)


class TestValidationTestReader(unittest.TestCase):

    def test_read_settings(self):
        settings = make_validation_test_reader('00001').read_settings()
        some_expected_settings = dict(
            start=0,
            variables=['X'],
            amount=['X'],
            sdRange=(-5, 5)
        )
        for expected_key, expected_value in some_expected_settings.items():
            self.assertEqual(settings[expected_key], expected_value)
        settings = make_validation_test_reader('00003').read_settings()
        self.assertEqual(settings['key1'], 'value1: has colon')
        self.assertEqual(
            make_validation_test_reader('00004').read_settings()['amount'], ['X', 'Y'])

        with self.assertRaisesRegexp(ValidationError,
            "duplicate key 'key1' in settings file '.*00002/00002-settings.txt"):
            make_validation_test_reader('00002').read_settings()

        with self.assertRaisesRegexp(ValidationError,
            "could not read settings file.*00005/00005-settings.txt.*No such file or directory.*"):
            make_validation_test_reader('00005').read_settings()

    def test_read_expected_predictions(self):
        for test_case_type in ['DISCRETE_STOCHASTIC', 'CONTINUOUS_DETERMINISTIC']:
            validation_test_reader = make_validation_test_reader('00001', test_case_type=test_case_type)
            validation_test_reader.settings = validation_test_reader.read_settings()
            expected_predictions_df = validation_test_reader.read_expected_predictions()
            self.assertTrue(isinstance(expected_predictions_df, pandas.core.frame.DataFrame))

        # wrong time sequence
        validation_test_reader.settings['duration'] += 1
        with self.assertRaisesRegexp(ValidationError, "times in settings .* differ from times in expected predictions"):
            validation_test_reader.read_expected_predictions()
        validation_test_reader.settings['duration'] -= 1

        # wrong columns
        missing_variable = 'MissingVariable'
        for test_case_type, expected_error in [
            ('DISCRETE_STOCHASTIC', "mean or sd of some amounts missing from expected predictions.*{}"),
            ('CONTINUOUS_DETERMINISTIC', "some amounts missing from expected predictions.*{}")]:
            validation_test_reader = make_validation_test_reader('00001', test_case_type=test_case_type)
            validation_test_reader.settings = validation_test_reader.read_settings()
            validation_test_reader.settings['amount'].append(missing_variable)
            with self.assertRaisesRegexp(ValidationError, expected_error.format(missing_variable)):
                validation_test_reader.read_expected_predictions()

    def test_read_model(self):
        validation_test_reader = make_validation_test_reader('00001')
        model = validation_test_reader.read_model()
        self.assertTrue(isinstance(model, obj_tables.Model))
        self.assertEqual(model.id, 'test_case_' + validation_test_reader.test_case_num)

    def test_validation_test_reader(self):
        test_case_num = '00001'
        validation_test_reader = make_validation_test_reader(test_case_num)
        self.assertEqual(None, validation_test_reader.run())

        # exceptions
        with self.assertRaisesRegexp(ValidationError, "Unknown ValidationTestCaseType:"):
            ValidationTestReader('no_such_test_case_type', '', test_case_num)


class TestResultsComparator(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.test_case_type = 'CONTINUOUS_DETERMINISTIC'
        self.test_case_num = '00001'
        self.simulation_run_results = self.make_run_results_from_expected_results(self.test_case_type,
            self.test_case_num)

    def make_run_results_filename(self):
        return os.path.join(tempfile.mkdtemp(dir=self.tmp_dir), 'run_results.h5')

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def make_run_results(self, pops_df, add_compartments=False):
        # make a RunResults obj with given population
        if add_compartments:
            # add compartments to species
            cols = list(pops_df.columns)
            pops_df.columns = cols[:1] + list(map(ResultsComparator.get_species, cols[1:]))

        # create an hdf
        run_results_filename = self.make_run_results_filename()
        pops_df.to_hdf(run_results_filename, 'populations')

        # add the other RunResults components as empty dfs
        empty_df = pandas.DataFrame(index=[], columns=[])
        for component in RunResults.COMPONENTS:
            if component != 'populations':
                empty_df.to_hdf(run_results_filename, component)

        # make & return a RunResults
        return RunResults(os.path.dirname(run_results_filename))

    def make_run_results_from_expected_results(self, test_case_type, test_case_num):
        """ Create a RunResults object with the same population as a test case """
        results_file = os.path.join(TEST_CASES, TEST_CASE_TYPE_TO_DIR[test_case_type], test_case_num,
            test_case_num+'-results.csv')
        results_df = pandas.read_csv(results_file)
        return self.make_run_results(results_df, add_compartments=True)

    def test_results_comparator_continuous_deterministic(self):
        validation_test_reader = make_validation_test_reader(self.test_case_num, self.test_case_type)
        validation_test_reader.run()
        results_comparator = ResultsComparator(validation_test_reader, self.simulation_run_results)
        self.assertEqual(False, results_comparator.differs())

        # modify the run results for first specie at time 0
        amount_1 = validation_test_reader.settings['amount'][0]
        self.simulation_run_results.get('populations').loc[0, amount_1] += 1
        self.assertTrue(results_comparator.differs())
        self.assertIn(amount_1, results_comparator.differs())

        '''
        # todo: test functionality of tolerances
        # allclose condition: absolute(a - b) <= (atol + rtol * absolute(b))
        # b is populations
        # assume b < a and 0 < b
        # condition becomes, a - b <= atol + rtol * b
        # or, a <= atol + (1 + rtol) * b        (1)
        # or, (a - atol)/(1 + rtol) <= b        (2)
        # use (1) or (2) to pick a given b, or b given a, respectively
        # modify b:
        # b = (a - atol)/(1 + rtol)     ==> equals
        # b += epsilon                  ==> differs
        '''

    def test_strip_compartments(self):
        pop_df = pandas.DataFrame(np.ones((2, 3)), index=[0, 10], columns='time X Y'.split())
        run_results = self.make_run_results(pop_df, add_compartments=True)
        pop_columns = list(run_results.get('populations').columns[1:])
        self.assertTrue(all(['[' in s for s in pop_columns]))
        ResultsComparator.strip_compartments(run_results)
        pop_columns = list(run_results.get('populations').columns[1:])
        self.assertFalse(all(['[' in s for s in pop_columns]))
        ResultsComparator.strip_compartments(run_results)
        pop_columns = list(run_results.get('populations').columns[1:])
        self.assertFalse(all(['[' in s for s in pop_columns]))

        rrs = []
        for i in range(2):
            rrs.append(self.make_run_results(pop_df))
        ResultsComparator.strip_compartments(rrs)
        for rr in rrs:
            pop_columns = list(rr.get('populations').columns[1:])
            self.assertFalse(all(['[' in s for s in pop_columns]))

        with self.assertRaisesRegexp(ValidationError, "wrong type for simulation_run_results.*"):
            ResultsComparator.strip_compartments(1.0)

    def stash_pd_value(self, df, loc, new_val):
        self.stashed_pd_value = df.loc[loc[0], loc[1]]
        df.loc[loc[0], loc[1]] = new_val

    def restore_pd_value(self, df, loc):
        df.loc[loc[0], loc[1]] = self.stashed_pd_value

    def test_results_comparator_discrete_stochastic(self):
        # todo: move code that's common with test_results_comparator_continuous_deterministic to function
        # todo: consolidate setup
        test_case_type = 'DISCRETE_STOCHASTIC'
        test_case_num = '00001'
        validation_test_reader = make_validation_test_reader(test_case_num, test_case_type)
        validation_test_reader.run()

        # make multiple run_results with variable populations
        # todo: use make_run_results_from_expected_results()
        # simulation_run_results = self.make_run_results_from_expected_results(test_case_type, test_case_num)
        expected_predictions_df = validation_test_reader.expected_predictions_df
        times = expected_predictions_df.loc[:,'time'].values
        n_times = len(times)
        means = expected_predictions_df.loc[:,'X-mean'].values
        correct_pop = expected_predictions_df.loc[:,['time', 'X-mean']]
        n_runs = 3
        run_results = []
        for i in range(n_runs):
            pops = np.empty((n_times, 2))
            pops[:,0] = times
            # add stochasticity to populations
            pops[:,1] = np.add(means, (np.random.rand(n_times)*2)-1)
            pop_df = pandas.DataFrame(pops, index=times, columns=['time', 'X'])
            run_results.append(self.make_run_results(pop_df, add_compartments=True))
            new_run_results_pop = run_results[-1].get('populations')

        results_comparator = ResultsComparator(validation_test_reader, run_results)
        self.assertEqual(False, results_comparator.differs())

        ### adjust data to test all Z thresholds ###
        # choose an arbitrary time
        time = 10

        # Z                 differ?
        # -                 -------
        # range[0]-epsilon  yes
        # range[0]+epsilon  no
        # range[1]-epsilon  no
        # range[1]+epsilon  yes
        epsilon = 1E-9
        lower_range, upper_range = (validation_test_reader.settings['meanRange'][0],
            validation_test_reader.settings['meanRange'][1])
        z_scores_and_expected_differs = [
            (lower_range - epsilon, ['X']),
            (lower_range + epsilon, False),
            (upper_range - epsilon, False),
            (upper_range + epsilon, ['X'])
        ]

        def get_test_pop_mean(time, n_runs, expected_df, desired_Z):
            # solve Z = math.sqrt(n_runs)*(pop_mean - e_mean)/e_sd for pop_mean:
            # pop_mean = Z*e_sd/math.sqrt(n_runs) + e_mean
            # return pop_mean for desired_Z
            return desired_Z * expected_df.loc[time, 'X-sd'] / math.sqrt(n_runs) + \
                expected_df.loc[time, 'X-mean']

        def set_all_pops(run_results_list, time, pop_val):
            # set all pops to pop_val
            for rr in run_results_list:
                rr.get('populations').loc[time, 'X'] = pop_val

        for test_z_score, expected_differ in z_scores_and_expected_differs:
            test_pop_mean = get_test_pop_mean(time, n_runs, expected_predictions_df, test_z_score)
            set_all_pops(run_results, time, test_pop_mean)
            results_comparator = ResultsComparator(validation_test_reader, run_results)
            self.assertEqual(expected_differ, results_comparator.differs())

        loc = [0, 'X-mean']
        self.stash_pd_value(expected_predictions_df, loc, -1)
        with self.assertRaisesRegexp(ValidationError, "e_mean contains negative value.*"):
            ResultsComparator(validation_test_reader, run_results).differs()
        self.restore_pd_value(expected_predictions_df, loc)

        loc = [0, 'X-sd']
        self.stash_pd_value(expected_predictions_df, loc, -1)
        with self.assertRaisesRegexp(ValidationError, "e_sd contains negative value.*"):
            ResultsComparator(validation_test_reader, run_results).differs()
        self.restore_pd_value(expected_predictions_df, loc)

    def test_prepare_tolerances(self):
        # make mock ValidationTestReader with just settings
        class Mock(object):
            def __init__(self):
                self.settings = None

        validation_test_reader = Mock()
        rel_tol, abs_tol = .1, .002
        validation_test_reader.settings = dict(relative=rel_tol, absolute=abs_tol)
        results_comparator = ResultsComparator(validation_test_reader, self.simulation_run_results)
        default_tolerances = ValidationUtilities.get_default_args(np.allclose)
        tolerances = results_comparator.prepare_tolerances()
        self.assertEqual(tolerances['rtol'], rel_tol)
        self.assertEqual(tolerances['atol'], abs_tol)
        validation_test_reader.settings['relative'] = None
        tolerances = results_comparator.prepare_tolerances()
        self.assertEqual(tolerances['rtol'], default_tolerances['rtol'])
        del validation_test_reader.settings['absolute']
        tolerances = results_comparator.prepare_tolerances()
        self.assertEqual(tolerances['atol'], default_tolerances['atol'])


class TestCaseValidator(unittest.TestCase):

    def setUp(self):
        self.test_case_num = '00001'
        self.tmp_dir = os.path.join(os.path.dirname(__file__), 'tmp')
        self.case_validators = {}
        self.model_types = ['DISCRETE_STOCHASTIC', 'CONTINUOUS_DETERMINISTIC']
        for model_type in self.model_types:
            self.case_validators[model_type] = CaseValidator(TEST_CASES, model_type, self.test_case_num,
            default_num_stochastic_runs=30, time_step_factor=0.01)

    def test_case_validator_errors(self):
        for model_type in self.model_types:
            settings = self.case_validators[model_type].validation_test_reader.settings
            del settings['duration']
            with self.assertRaisesRegexp(ValidationError, "required setting .* not provided"):
                self.case_validators[model_type].validate_model()
            settings['duration'] = 'not a float'
            with self.assertRaisesRegexp(ValidationError, "required setting .* not a float"):
                self.case_validators[model_type].validate_model()
            settings['duration'] = 10.
            settings['start'] = 3
            with self.assertRaisesRegexp(ValidationError, "non-zero start setting .* not supported"):
                self.case_validators[model_type].validate_model()

    def make_plot_file(self, model_type, case=None):
        if case is None:
            case = self.test_case_num
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')
        plot_file = os.path.join(self.tmp_dir, model_type, "{}_{}.pdf".format(case, timestamp))
        os.makedirs(os.path.dirname(plot_file), exist_ok=True)
        return plot_file

    @unittest.skip('not a test')
    def test_case_validate_ode(self):
        cases = '00001 00004'.split()
        model_type = 'CONTINUOUS_DETERMINISTIC'
        for case in cases:
            factors = [0.01, 0.1, 0.5]
            for factor in factors:
                case_validator = CaseValidator(TEST_CASES, model_type, case, time_step_factor=factor)
                plot_file = self.make_plot_file(model_type, case=case)
                try:
                    case_validator.validate_model(plot_file=plot_file)
                except Exception as e:
                    print('Exception', e)
                    pass

    # todo: fix
    @unittest.skip('broken')
    def test_case_validator(self):
        for model_type in self.model_types:
            plot_file = self.make_plot_file(model_type)
            self.assertFalse(self.case_validators[model_type].validate_model(plot_file=plot_file))
            self.assertTrue(os.path.isfile(plot_file))

    # test validation failure
    # todo: move to separate test
    '''
    expected_preds_df = self.case_validators[model_type].validation_test_reader.expected_predictions_df
    expected_preds_array = expected_preds_df.loc[:, 'X-mean'].values
    expected_preds_df.loc[:, 'X-mean'] = np.full(expected_preds_array.shape, 0)
    self.assertEqual(['X'], self.case_validators[model_type].validate_model(
        discard_run_results=False, plot_file=self.make_plot_file(model_type)))
    '''

    # todo: test deletion (and not) of run_results


# todo: fix
@unittest.skip('broken')
class TestValidationSuite(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.validation_suite = ValidationSuite(TEST_CASES, self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_init(self):
        self.assertEqual(self.validation_suite.plot_dir, self.tmp_dir)
        no_such_dir = os.path.join(self.tmp_dir, 'no_such_dir')
        with self.assertRaisesRegexp(ValidationError, "cannot open cases_dir"):
            ValidationSuite(no_such_dir)
        with self.assertRaisesRegexp(ValidationError, "cannot open plot_dir"):
            ValidationSuite(TEST_CASES, no_such_dir)

    def test_record_result(self):
        self.assertEqual(self.validation_suite.results, [])
        sub_dir = os.path.join(self.tmp_dir, 'test_case_sub_dir')

        result = ValidationRunResult(TEST_CASES, sub_dir, '00001', ValidationResultType.CASE_VALIDATED)
        self.validation_suite._record_result(*result[1:])
        self.assertEqual(len(self.validation_suite.results), 1)
        self.assertEqual(self.validation_suite.results[-1], result)

        result = ValidationRunResult(TEST_CASES, sub_dir, '00001', ValidationResultType.CASE_UNREADABLE, 'error')
        self.validation_suite._record_result(*result[1:])
        self.assertEqual(len(self.validation_suite.results), 2)
        self.assertEqual(self.validation_suite.results[-1], result)

        with self.assertRaisesRegexp(ValidationError, "result_type must be a ValidationResultType, not a"):
            self.validation_suite._record_result(TEST_CASES, sub_dir, '00001', 'not a ValidationResultType')

    def test_run_test(self):
        test_case_num = '00001'
        self.validation_suite._run_test('DISCRETE_STOCHASTIC', test_case_num)
        results = self.validation_suite.get_results()
        print(results[-1].error)
        self.assertEqual(results.pop().result_type, ValidationResultType.CASE_VALIDATED)
        plot_file_name_prefix = 'DISCRETE_STOCHASTIC' + '_' + test_case_num
        self.assertIn(plot_file_name_prefix, os.listdir(self.tmp_dir).pop())

        # test without plotting
        validation_suite = ValidationSuite(TEST_CASES)
        validation_suite._run_test('DISCRETE_STOCHASTIC', test_case_num, num_stochastic_runs=100)
        self.assertEqual(validation_suite.results.pop().result_type, ValidationResultType.CASE_VALIDATED)

        # case unreadable
        validation_suite = ValidationSuite(TEST_CASES)
        self.validation_suite._run_test('stochastic', test_case_num, num_stochastic_runs=5)
        self.assertEqual(results.pop().result_type, ValidationResultType.CASE_UNREADABLE)

        # run fails
        plot_dir = tempfile.mkdtemp()
        validation_suite = ValidationSuite(TEST_CASES, plot_dir)
        # delete plot_dir to create failure
        shutil.rmtree(plot_dir)
        validation_suite._run_test('DISCRETE_STOCHASTIC', test_case_num, num_stochastic_runs=5)
        self.assertEqual(validation_suite.results.pop().result_type, ValidationResultType.FAILED_VALIDATION_RUN)

        # run does not validate
        validation_suite = ValidationSuite(TEST_CASES)
        validation_suite._run_test('DISCRETE_STOCHASTIC', '00006', num_stochastic_runs=5)
        self.assertEqual(validation_suite.results.pop().result_type, ValidationResultType.CASE_DID_NOT_VALIDATE)

    def test_run(self):
        results = self.validation_suite.run('DISCRETE_STOCHASTIC', ['00001'], num_stochastic_runs=5)
        self.assertEqual(results.pop().result_type, ValidationResultType.CASE_VALIDATED)

        results = self.validation_suite.run('DISCRETE_STOCHASTIC', ['00001', '00006'], num_stochastic_runs=5)
        expected_types = [ValidationResultType.CASE_VALIDATED, ValidationResultType.CASE_DID_NOT_VALIDATE]
        result_types = [result_tuple.result_type for result_tuple in results[-2:]]
        self.assertEqual(expected_types, result_types)

        results = self.validation_suite.run('DISCRETE_STOCHASTIC', num_stochastic_runs=5)
        self.assertEqual(expected_types, result_types)

        results = self.validation_suite.run(num_stochastic_runs=5)
        self.assertEqual(expected_types, result_types)

        with self.assertRaisesRegexp(ValidationError, 'cases should be an iterator over case nums'):
            self.validation_suite.run('DISCRETE_STOCHASTIC', '00001')

        with self.assertRaisesRegexp(ValidationError, 'if cases provided then test_case_type_name must'):
            self.validation_suite.run(cases=['00001'])

        with self.assertRaisesRegexp(ValidationError, 'Unknown ValidationTestCaseType: '):
            self.validation_suite.run(test_case_type_name='no such ValidationTestCaseType')


SsaTestCase = namedtuple('SsaTestCase', 'case_num, dsmts_num, MA_order, num_ssa_runs')


class RunValidationSuite(unittest.TestCase):

    def setUp(self):
        NUM_SIMULATION_RUNS = 2000
        self.ssa_test_cases = [
            # see: https://github.com/sbmlteam/sbml-test-suite/blob/master/cases/stochastic/DSMTS-userguide-31v2.pdf
            SsaTestCase('00001', '001-01', (1, ), NUM_SIMULATION_RUNS),
            SsaTestCase('00003', '001-03', (1, ), NUM_SIMULATION_RUNS),
            SsaTestCase('00004', '001-04', (1, ), NUM_SIMULATION_RUNS),
            SsaTestCase('00007', '001-07', (1, ), NUM_SIMULATION_RUNS),
            SsaTestCase('00012', '001-12', (1, ), NUM_SIMULATION_RUNS),
            SsaTestCase('00020', '002-01', (0, 1), 4000),
            SsaTestCase('00021', '002-02', (0, 1), NUM_SIMULATION_RUNS),
            SsaTestCase('00030', '003-01', (1, 2), NUM_SIMULATION_RUNS),
            SsaTestCase('00037', '004-01', (0, 1), NUM_SIMULATION_RUNS)
        ]
        # todo: get rid of TIME_STEP_FACTOR
        TIME_STEP_FACTOR = 1
        self.ode_test_cases = [
            ('00001', TIME_STEP_FACTOR),
            ('00004', TIME_STEP_FACTOR),
            ('00006', TIME_STEP_FACTOR),
            ('00010', TIME_STEP_FACTOR),
            ('00014', TIME_STEP_FACTOR),
            ('00015', TIME_STEP_FACTOR),
            # ('00021', 0.2*TIME_STEP_FACTOR),  # fails, perhaps because compartment volume = 0.3 rather than 1 liter
            ('00022', TIME_STEP_FACTOR)
        ]
        root_test_dir = os.path.join(os.path.dirname(__file__), 'fixtures', 'validation', 'cases')
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.plot_dir = os.path.join(os.path.dirname(__file__), 'tmp', 'validation', timestamp)
        os.makedirs(self.plot_dir)
        self.validation_suite = ValidationSuite(root_test_dir, self.plot_dir)

    def run_validation_cases(self, case_type, validation_cases, testing=False, time_step_factor=None):
        if case_type == 'DISCRETE_STOCHASTIC':
            for ssa_test_case in validation_cases:
                self.validation_suite.run(case_type, [ssa_test_case.case_num],
                    num_stochastic_runs=ssa_test_case.num_ssa_runs, verbose=True, time_step_factor=time_step_factor)

        if case_type == 'CONTINUOUS_DETERMINISTIC':
            for test_case, time_step_factor in validation_cases:
                self.validation_suite.run('CONTINUOUS_DETERMINISTIC', [test_case],
                    time_step_factor=time_step_factor)

        failures = []
        successes = []
        # print(self.validation_suite.get_results())
        for result in self.validation_suite.get_results():
            if result.error:
                # print(result.error.replace('\\n', '\n'))
                failure_msg = "{} {}\n".format(result.case_num, result.result_type.name) + \
                    "{}".format(result.error)
                failures.append(failure_msg)
            else:
                successes.append("{} {}".format(result.case_num, result.result_type.name))
        if testing:
            self.assertTrue(failures == [], msg='\n'.join(successes + failures))
        if not failures:
            print('\n'.join(successes))
        else:
            print('failure:', '\n'.join(failures))
        return self.validation_suite.get_results(), failures, successes

    @unittest.skip('slow, because SSA tests need many simulation runs')
    def test_validation_stochastic(self):
        raise ValueError('will not work until MA reenabled')
        # todo: move to validation main program
        results, _, _ = self.run_validation_cases('DISCRETE_STOCHASTIC', self.ssa_test_cases)

        orders_validated = set()
        for result, ssa_test_case in zip(results, self.ssa_test_cases):
            if result.result_type == ValidationResultType.CASE_VALIDATED:
                orders_validated.update(ssa_test_case.MA_order)
        self.assertEqual(orders_validated, {0, 1, 2})

    def test_validation_deterministic(self):
        self.run_validation_cases('CONTINUOUS_DETERMINISTIC', self.ode_test_cases)

    def test_validation_hybrid(self):
        # transcription_translation_case = SsaTestCase('transcription_translation', 'NA', (1, ), 10)
        # translation_metabolism_case = SsaTestCase('translation_metabolism', 'NA', (1, ), 10)
        test_case = SsaTestCase('00007_hybrid', 'NA', (1, ), 500)
        profile = False
        if profile:
            print('profiling')
            tmp_dir = tempfile.mkdtemp()
            out_file = os.path.join(tmp_dir, "profile_out.out")
            locals = {'self': self,
                'test_case': test_case}
            cProfile.runctx("self.run_validation_cases('DISCRETE_STOCHASTIC', [test_case])",
                {}, locals, filename=out_file)
            profile = pstats.Stats(out_file)
            print("Profile for 'test_case' simulation objects")
            profile.strip_dirs().sort_stats('cumulative').print_stats(20)
        else:
            self.run_validation_cases('DISCRETE_STOCHASTIC', [test_case])


class TestValidationUtilities(unittest.TestCase):

    def test_get_default_args(self):

        defaults = {'a': None,
            'b': 17,
            'c': frozenset(range(3))}
        def func(y, a=defaults['a'], b=defaults['b'], c=defaults['c']):
            pass
        self.assertEqual(defaults, ValidationUtilities.get_default_args(func))


@unittest.skip('incomplete')
class TestHybridModelValidation(unittest.TestCase):

    HYBRID_DIR = os.path.join(os.path.dirname(__file__), 'fixtures', 'validation', 'testing', 'hybrid')

    def setUp(self):
        self.hybrid_model_validations = {}

        trans_trans_hybrid_settings = dict(
            start='0',
            duration='2.88E+04',
            steps='2.88E+02',
            amount='RNA, protein',
            output='RNA-mean, RNA-sd, protein-mean, protein-sd',
            meanRange='(-3, 3)',
        )
        trans_trans_correct_ssa_settings = trans_trans_hybrid_settings
        self.trans_trans_model_base = 'transcription_translation'
        self.hybrid_model_validations[self.trans_trans_model_base] = \
            self.make_hybrid_validation(self.trans_trans_model_base, trans_trans_hybrid_settings,
                trans_trans_correct_ssa_settings)

        trans_met_hybrid_settings = dict(
            start='0',
            duration='2.88E+04',
            steps='2.88E+02',
            amount='protein, Ala, H2O',
            output='protein-mean, protein-sd, Ala-mean, Ala-sd, H2O-mean, H2O-sd',
            meanRange='(-3, 3)',
        )
        trans_met_correct_ssa_settings = trans_met_hybrid_settings
        self.trans_met_model_base = 'translation_metabolism'
        self.hybrid_model_validations[self.trans_met_model_base] = \
            self.make_hybrid_validation(self.trans_met_model_base, trans_met_hybrid_settings,
                trans_met_correct_ssa_settings)

    def tearDown(self):
        pass

    def make_hybrid_validation(self, model_base_filename, correct_ssa_settings, hybrid_settings):
        validation_dir = tempfile.mkdtemp(dir=self.HYBRID_DIR)
        correct_ssa_model_file = os.path.join(self.HYBRID_DIR, model_base_filename + '_correct_ssa.xlsx')
        hybrid_model_file = os.path.join(self.HYBRID_DIR, model_base_filename + '_hybrid.xlsx')
        hybrid_model_validation = HybridModelValidation(
            validation_dir,
            model_base_filename,
            correct_ssa_model_file,
            correct_ssa_settings,
            hybrid_model_file,
            hybrid_settings
        )
        return hybrid_model_validation

    def test(self):
        # in [self.trans_trans_model_base, self.trans_met_model_base]:
        for model_base_name in [self.trans_trans_model_base, ]:
            self.hybrid_model_validations[self.trans_trans_model_base].run()
