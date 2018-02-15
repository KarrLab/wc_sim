""" Tests of sim_config.SimulationConfig

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2017-07-25
:Copyright: 2016, Karr Lab
:License: MIT
"""

from wc_sim import sim_config
import libsedml
import numpy
import os
import pytest
import tempfile
import unittest
import warnings

class TestSimulationConfig(unittest.TestCase):
    """ Test simulation configuration """

    def test_simulation_configuration(self):
        """ Test simulation configuration correctly represented """

        # create configuration
        time_max = 100
        time_step = 2
        changes = [
            sim_config.Change(target='rxn-1.vmax', value=1),
            sim_config.Change(target='species-1', value=2),
        ]
        perturbations = [
            sim_config.Perturbation(sim_config.Change('rxn-1.vmax', 3), start_time=1),
            sim_config.Perturbation(sim_config.Change('species-1', 4), start_time=0, end_time=10),
        ]
        random_seed = 3
        cfg = sim_config.SimulationConfig(time_max=time_max, time_step=time_step, changes=changes,
                                          perturbations=perturbations, random_seed=random_seed)

        # check correctly stored configuration
        self.assertEqual(time_max + 0.0, cfg.time_max)
        self.assertEqual(time_step + 0.0, cfg.time_step)
        self.assertEqual(random_seed, cfg.random_seed)

    def test_simulation_configuration_validation(self):
        # time init not number
        sim_config.SimulationConfig(time_init=1.0)
        with self.assertRaises(sim_config.SimulationConfigError):
            sim_config.SimulationConfig(time_init=None)

        # time max not number
        sim_config.SimulationConfig(time_max=1.0)
        with self.assertRaises(sim_config.SimulationConfigError):
            sim_config.SimulationConfig(time_max=None)

        # negative time max
        sim_config.SimulationConfig(time_max=1.0)
        with self.assertRaises(sim_config.SimulationConfigError):
            sim_config.SimulationConfig(time_max=-1.0)

        # time step not number
        sim_config.SimulationConfig(time_step=0.1)
        with self.assertRaises(sim_config.SimulationConfigError):
            sim_config.SimulationConfig(time_step=None)

        # negative time step
        sim_config.SimulationConfig(time_step=0.1)
        with self.assertRaises(sim_config.SimulationConfigError):
            sim_config.SimulationConfig(time_step=-0.1)

        # time max not multiple of step
        sim_config.SimulationConfig(time_max=1.0, time_step=0.1)
        with self.assertRaises(sim_config.SimulationConfigError):
            sim_config.SimulationConfig(time_max=1.0, time_step=3.0)

        # random seed not an integer
        sim_config.SimulationConfig(random_seed=1)
        with self.assertRaises(sim_config.SimulationConfigError):
            sim_config.SimulationConfig(random_seed='a')
        with self.assertRaises(sim_config.SimulationConfigError):
            sim_config.SimulationConfig(random_seed=1.5)

    @unittest.skip('Not yet implemented')
    def test_apply_changes(self):
        # .. todo :: implement
        cfg = sim_config.SimulationConfig(time_max=100, time_step=2)
        cfg.apply_changes(None)

    @unittest.skip('Not yet implemented')
    def test_apply_perturbations(self):
        # .. todo :: implement
        cfg = sim_config.SimulationConfig(time_max=100, time_step=2)
        cfg.apply_perturbations(None)

    def test_get_num_time_steps(self):
        cfg = sim_config.SimulationConfig(time_max=100, time_step=2)
        self.assertEqual(cfg.get_num_time_steps(), 50)


class TestSedMlImportExport(unittest.TestCase):

    def test_sedml_import_export(self):
        """ Test SED-ML import/export of simulation configurations """

        # create configuration
        time_max = 100.0
        time_step = 2.0
        changes = [
            sim_config.Change(target='rxn-1.vmax', value=1),
            sim_config.Change(target='species-1', value=2),
        ]
        perturbations = [
            sim_config.Perturbation(sim_config.Change('rxn-1.vmax', 3), start_time=1),
            sim_config.Perturbation(sim_config.Change('species-1', 4), start_time=0, end_time=10),
        ]
        random_seed = 3
        cfg = sim_config.SimulationConfig(time_max=time_max, time_step=time_step, changes=changes,
                                          perturbations=perturbations, random_seed=random_seed)

        # generate temporary file
        file_name = tempfile.mktemp()

        # export configuration
        sim_config.SedMl.write(cfg, file_name)

        # import configuration
        cfg2 = sim_config.SedMl.read(file_name)

        # check sim_config correctly imported/exported
        self.assertEqual(time_max, cfg2.time_max)
        self.assertEqual(time_step, cfg2.time_step)
        self.assertEqual(random_seed, cfg2.random_seed)

        self.assertEqual(len(changes), len(cfg2.changes))
        self.assertEqual(changes[0].target, cfg2.changes[0].target)
        self.assertEqual(changes[1].target, cfg2.changes[1].target)
        self.assertEqual(changes[0].value, cfg2.changes[0].value)
        self.assertEqual(changes[1].value, cfg2.changes[1].value)

        self.assertEqual(len(perturbations), len(cfg2.perturbations))
        self.assertEqual(perturbations[0].change.target, cfg2.perturbations[0].change.target)
        self.assertEqual(perturbations[1].change.target, cfg2.perturbations[1].change.target)
        self.assertEqual(perturbations[0].change.value, cfg2.perturbations[0].change.value)
        self.assertEqual(perturbations[1].change.value, cfg2.perturbations[1].change.value)

        self.assertEqual(perturbations[0].start_time, cfg2.perturbations[0].start_time)
        self.assertEqual(perturbations[1].start_time, cfg2.perturbations[1].start_time)
        numpy.testing.assert_equal(perturbations[0].end_time, cfg2.perturbations[0].end_time)
        numpy.testing.assert_equal(perturbations[1].end_time, cfg2.perturbations[1].end_time)

        # cleanup
        os.remove(file_name)

    def test_error(self):
        sim_config.SedMlError('msg')

    def test_warning(self):
        sim_config.SedMlWarning('msg')


class TestSedMlValidation(unittest.TestCase):

    def setUp(self):
        _, self.filename = tempfile.mkstemp()
        warnings.simplefilter("ignore")

    def tearDown(self):
        os.remove(self.filename)

    def test_no_model(self):
        cfg_ml = libsedml.SedDocument()
        libsedml.writeSedML(cfg_ml, self.filename)

        with self.assertRaises(sim_config.SedMlError):
            sim_config.SedMl.read(self.filename)

    def test_ignore_model_attribute(self):
        cfg_ml = libsedml.SedDocument()

        mdl = cfg_ml.createModel()
        mdl.setName('test')

        sim = cfg_ml.createUniformTimeCourse()
        sim.setInitialTime(0.0)
        sim.setOutputStartTime(0.0)
        sim.setOutputEndTime(10.)
        sim.setNumberOfPoints(10)

        alg = sim.createAlgorithm()
        alg.setKisaoID("KISAO_0000352")
        param = alg.createAlgorithmParameter()
        param.setKisaoID("KISAO_0000488")
        param.setValue('1')

        libsedml.writeSedML(cfg_ml, self.filename)

        pytest.warns(sim_config.SedMlWarning, sim_config.SedMl.read, self.filename)

    def test_no_non_change_attributes(self):
        cfg_ml = libsedml.SedDocument()

        mdl = cfg_ml.createModel()
        mdl.setName('test')

        change_ml = mdl.createComputeChange()

        sim = cfg_ml.createUniformTimeCourse()
        sim.setInitialTime(0.0)
        sim.setOutputStartTime(0.0)
        sim.setOutputEndTime(10.)
        sim.setNumberOfPoints(10)

        alg = sim.createAlgorithm()
        alg.setKisaoID("KISAO_0000352")
        param = alg.createAlgorithmParameter()
        param.setKisaoID("KISAO_0000488")
        param.setValue('1')

        libsedml.writeSedML(cfg_ml, self.filename)

        with self.assertRaises(sim_config.SedMlError):
            sim_config.SedMl.read(self.filename)

    def test_no_simulations(self):
        cfg_ml = libsedml.SedDocument()

        mdl = cfg_ml.createModel()
        mdl.setName('test')

        libsedml.writeSedML(cfg_ml, self.filename)

        with self.assertRaises(sim_config.SedMlError):
            sim_config.SedMl.read(self.filename)

    def test_no_initial_time(self):
        cfg_ml = libsedml.SedDocument()

        mdl = cfg_ml.createModel()
        mdl.setName('test')

        sim = cfg_ml.createUniformTimeCourse()

        libsedml.writeSedML(cfg_ml, self.filename)

        with self.assertRaises(sim_config.SedMlError):
            sim_config.SedMl.read(self.filename)

    def test_no_output_start_time(self):
        cfg_ml = libsedml.SedDocument()

        mdl = cfg_ml.createModel()
        mdl.setName('test')

        sim = cfg_ml.createUniformTimeCourse()
        sim.setInitialTime(0.0)

        libsedml.writeSedML(cfg_ml, self.filename)

        with self.assertRaises(sim_config.SedMlError):
            sim_config.SedMl.read(self.filename)

    def test_invalid_kisao_id(self):
        cfg_ml = libsedml.SedDocument()

        mdl = cfg_ml.createModel()
        mdl.setName('test')

        sim = cfg_ml.createUniformTimeCourse()
        sim.setInitialTime(0.0)
        sim.setOutputStartTime(0.0)

        alg = sim.createAlgorithm()
        alg.setKisaoID("__x__")

        libsedml.writeSedML(cfg_ml, self.filename)

        with self.assertRaises(sim_config.SedMlError):
            sim_config.SedMl.read(self.filename)

    def test_invalid_random_seed_string(self):
        cfg_ml = libsedml.SedDocument()

        mdl = cfg_ml.createModel()
        mdl.setName('test')

        sim = cfg_ml.createUniformTimeCourse()
        sim.setInitialTime(0.0)
        sim.setOutputStartTime(0.0)

        alg = sim.createAlgorithm()
        alg.setKisaoID("KISAO_0000352")

        param = alg.createAlgorithmParameter()
        param.setKisaoID("KISAO_0000488")
        param.setValue('x')

        libsedml.writeSedML(cfg_ml, self.filename)

        with self.assertRaisesRegexp(sim_config.SedMlError, 'random_seed must be an integer'):
            sim_config.SedMl.read(self.filename)

    def test_invalid_random_seed_float(self):
        cfg_ml = libsedml.SedDocument()

        mdl = cfg_ml.createModel()
        mdl.setName('test')

        sim = cfg_ml.createUniformTimeCourse()
        sim.setInitialTime(0.0)
        sim.setOutputStartTime(0.0)

        alg = sim.createAlgorithm()
        alg.setKisaoID("KISAO_0000352")

        param = alg.createAlgorithmParameter()
        param.setKisaoID("KISAO_0000488")
        param.setValue('1.5')

        libsedml.writeSedML(cfg_ml, self.filename)

        with self.assertRaisesRegexp(sim_config.SedMlError, 'random_seed must be an integer'):
            sim_config.SedMl.read(self.filename)

    def test_not_unique_algorithm_parameters(self):
        cfg_ml = libsedml.SedDocument()

        mdl = cfg_ml.createModel()
        mdl.setName('test')

        sim = cfg_ml.createUniformTimeCourse()
        sim.setInitialTime(0.0)
        sim.setOutputStartTime(0.0)

        alg = sim.createAlgorithm()
        alg.setKisaoID("KISAO_0000352")

        param = alg.createAlgorithmParameter()
        param.setKisaoID("x")
        param.setValue('x')

        param = alg.createAlgorithmParameter()
        param.setKisaoID("x")
        param.setValue('x')

        libsedml.writeSedML(cfg_ml, self.filename)

        with self.assertRaisesRegexp(sim_config.SedMlError, 'Algorithm parameter KISAO ids must be unique'):
            sim_config.SedMl.read(self.filename)

    def test_invalid_algorithm_parameter(self):
        cfg_ml = libsedml.SedDocument()

        mdl = cfg_ml.createModel()
        mdl.setName('test')

        sim = cfg_ml.createUniformTimeCourse()
        sim.setInitialTime(0.0)
        sim.setOutputStartTime(0.0)

        alg = sim.createAlgorithm()
        alg.setKisaoID("KISAO_0000352")

        param = alg.createAlgorithmParameter()
        param.setKisaoID("x")
        param.setValue('x')

        libsedml.writeSedML(cfg_ml, self.filename)

        with self.assertRaises(sim_config.SedMlError):
            sim_config.SedMl.read(self.filename)

    def test_no_task_change(self):
        cfg_ml = libsedml.SedDocument()

        mdl = cfg_ml.createModel()
        mdl.setName('test')

        sim = cfg_ml.createUniformTimeCourse()
        sim.setInitialTime(0.0)
        sim.setOutputStartTime(0.0)

        alg = sim.createAlgorithm()
        alg.setKisaoID("KISAO_0000352")

        param = alg.createAlgorithmParameter()
        param.setKisaoID("KISAO_0000488")
        param.setValue('1')

        task_ml = cfg_ml.createRepeatedTask()

        libsedml.writeSedML(cfg_ml, self.filename)

        with self.assertRaises(sim_config.SedMlError):
            sim_config.SedMl.read(self.filename)

    def test_invalid_set_value(self):
        cfg_ml = libsedml.SedDocument()

        mdl = cfg_ml.createModel()
        mdl.setName('test')

        sim = cfg_ml.createUniformTimeCourse()
        sim.setInitialTime(0.0)
        sim.setOutputStartTime(0.0)

        alg = sim.createAlgorithm()
        alg.setKisaoID("KISAO_0000352")

        param = alg.createAlgorithmParameter()
        param.setKisaoID("KISAO_0000488")
        param.setValue('1')

        task_ml = cfg_ml.createRepeatedTask()
        change_ml = task_ml.createTaskChange()
        change_ml.setTarget('x')
        change_ml.setMath(libsedml.parseFormula('x'))

        libsedml.writeSedML(cfg_ml, self.filename)

        with self.assertRaises(sim_config.SedMlError):
            sim_config.SedMl.read(self.filename)

    def test_no_range(self):
        cfg_ml = libsedml.SedDocument()

        mdl = cfg_ml.createModel()
        mdl.setName('test')

        sim = cfg_ml.createUniformTimeCourse()
        sim.setInitialTime(0.0)
        sim.setOutputStartTime(0.0)

        alg = sim.createAlgorithm()
        alg.setKisaoID("KISAO_0000352")

        param = alg.createAlgorithmParameter()
        param.setKisaoID("KISAO_0000488")
        param.setValue('1')

        task_ml = cfg_ml.createRepeatedTask()
        change_ml = task_ml.createTaskChange()
        change_ml.setTarget('x')
        change_ml.setMath(libsedml.parseFormula('1'))

        libsedml.writeSedML(cfg_ml, self.filename)

        with self.assertRaises(sim_config.SedMlError):
            sim_config.SedMl.read(self.filename)

    def test_no_functional_range(self):
        cfg_ml = libsedml.SedDocument()

        mdl = cfg_ml.createModel()
        mdl.setName('test')

        sim = cfg_ml.createUniformTimeCourse()
        sim.setInitialTime(0.0)
        sim.setOutputStartTime(0.0)

        alg = sim.createAlgorithm()
        alg.setKisaoID("KISAO_0000352")

        param = alg.createAlgorithmParameter()
        param.setKisaoID("KISAO_0000488")
        param.setValue('1')

        task_ml = cfg_ml.createRepeatedTask()
        change_ml = task_ml.createTaskChange()
        change_ml.setTarget('x')
        change_ml.setMath(libsedml.parseFormula('1'))

        rangle_ml = task_ml.createFunctionalRange()

        libsedml.writeSedML(cfg_ml, self.filename)

        with self.assertRaises(sim_config.SedMlError):
            sim_config.SedMl.read(self.filename)

    def test_no_non_repeated_task(self):
        cfg_ml = libsedml.SedDocument()

        mdl = cfg_ml.createModel()
        mdl.setName('test')

        sim = cfg_ml.createUniformTimeCourse()
        sim.setInitialTime(0.0)
        sim.setOutputStartTime(0.0)

        alg = sim.createAlgorithm()
        alg.setKisaoID("KISAO_0000352")

        param = alg.createAlgorithmParameter()
        param.setKisaoID("KISAO_0000488")
        param.setValue('1')

        task_ml = cfg_ml.createTask()

        libsedml.writeSedML(cfg_ml, self.filename)

        with self.assertRaises(sim_config.SedMlError):
            sim_config.SedMl.read(self.filename)

    def test_no_subtasks(self):
        cfg_ml = libsedml.SedDocument()

        mdl = cfg_ml.createModel()
        mdl.setName('test')

        sim = cfg_ml.createUniformTimeCourse()
        sim.setInitialTime(0.0)
        sim.setOutputStartTime(0.0)
        sim.setOutputEndTime(10.)
        sim.setNumberOfPoints(10)

        alg = sim.createAlgorithm()
        alg.setKisaoID("KISAO_0000352")

        param = alg.createAlgorithmParameter()
        param.setKisaoID("KISAO_0000488")
        param.setValue('1')

        task_ml = cfg_ml.createRepeatedTask()
        change_ml = task_ml.createTaskChange()
        change_ml.setTarget('x')
        change_ml.setMath(libsedml.parseFormula('1'))

        rangle_ml = task_ml.createVectorRange()
        rangle_ml.setValues([1])

        task_ml.createSubTask()

        libsedml.writeSedML(cfg_ml, self.filename)

        with self.assertRaises(sim_config.SedMlError):
            sim_config.SedMl.read(self.filename)

    def test_no_data_generators(self):
        cfg_ml = libsedml.SedDocument()

        mdl = cfg_ml.createModel()
        mdl.setName('test')

        sim = cfg_ml.createUniformTimeCourse()
        sim.setInitialTime(0.0)
        sim.setOutputStartTime(0.0)
        sim.setOutputEndTime(10.)
        sim.setNumberOfPoints(10)

        alg = sim.createAlgorithm()
        alg.setKisaoID("KISAO_0000352")

        param = alg.createAlgorithmParameter()
        param.setKisaoID("KISAO_0000488")
        param.setValue('1')

        task_ml = cfg_ml.createRepeatedTask()
        change_ml = task_ml.createTaskChange()
        change_ml.setTarget('x')
        change_ml.setMath(libsedml.parseFormula('1'))

        rangle_ml = task_ml.createVectorRange()
        rangle_ml.setValues([1])

        task_ml.createSubTask()

        libsedml.writeSedML(cfg_ml, self.filename)

        with self.assertRaises(sim_config.SedMlError):
            sim_config.SedMl.read(self.filename)

    def test_no_data_generators(self):
        cfg_ml = libsedml.SedDocument()

        mdl = cfg_ml.createModel()
        mdl.setName('test')

        sim = cfg_ml.createUniformTimeCourse()
        sim.setInitialTime(0.0)
        sim.setOutputStartTime(0.0)
        sim.setOutputEndTime(10.)
        sim.setNumberOfPoints(10)

        alg = sim.createAlgorithm()
        alg.setKisaoID("KISAO_0000352")

        param = alg.createAlgorithmParameter()
        param.setKisaoID("KISAO_0000488")
        param.setValue('1')

        cfg_ml.createDataGenerator()

        libsedml.writeSedML(cfg_ml, self.filename)

        with self.assertRaises(sim_config.SedMlError):
            sim_config.SedMl.read(self.filename)

    def test_no_data_descriptions(self):
        cfg_ml = libsedml.SedDocument()

        mdl = cfg_ml.createModel()
        mdl.setName('test')

        sim = cfg_ml.createUniformTimeCourse()
        sim.setInitialTime(0.0)
        sim.setOutputStartTime(0.0)
        sim.setOutputEndTime(10.)
        sim.setNumberOfPoints(10)

        alg = sim.createAlgorithm()
        alg.setKisaoID("KISAO_0000352")

        param = alg.createAlgorithmParameter()
        param.setKisaoID("KISAO_0000488")
        param.setValue('1')

        cfg_ml.createDataDescription()

        libsedml.writeSedML(cfg_ml, self.filename)

        with self.assertRaises(sim_config.SedMlError):
            sim_config.SedMl.read(self.filename)

    def test_no_outputs(self):
        cfg_ml = libsedml.SedDocument()

        mdl = cfg_ml.createModel()
        mdl.setName('test')

        sim = cfg_ml.createUniformTimeCourse()
        sim.setInitialTime(0.0)
        sim.setOutputStartTime(0.0)
        sim.setOutputEndTime(10.)
        sim.setNumberOfPoints(10)

        alg = sim.createAlgorithm()
        alg.setKisaoID("KISAO_0000352")

        param = alg.createAlgorithmParameter()
        param.setKisaoID("KISAO_0000488")
        param.setValue('1')

        cfg_ml.createReport()

        libsedml.writeSedML(cfg_ml, self.filename)

        with self.assertRaises(sim_config.SedMlError):
            sim_config.SedMl.read(self.filename)

    def test_no_number_of_points(self):
        cfg_ml = libsedml.SedDocument()

        mdl = cfg_ml.createModel()
        mdl.setName('test')

        sim = cfg_ml.createUniformTimeCourse()
        sim.setInitialTime(0.0)
        sim.setOutputStartTime(0.0)
        sim.setOutputEndTime(10.)
        sim.setNumberOfPoints(10)

        alg = sim.createAlgorithm()
        alg.setKisaoID("KISAO_0000352")
        param = alg.createAlgorithmParameter()
        param.setKisaoID("KISAO_0000488")
        param.setValue('1')

        task_ml = cfg_ml.createRepeatedTask()
        change_ml = task_ml.createTaskChange()
        change_ml.setTarget('x')
        change_ml.setMath(libsedml.parseFormula('1'))

        rangle_ml = task_ml.createUniformRange()
        rangle_ml.setStart(0)
        rangle_ml.setEnd(1)
        rangle_ml.setNumberOfPoints(2)

        libsedml.writeSedML(cfg_ml, self.filename)

        with self.assertRaises(sim_config.SedMlError):
            sim_config.SedMl.read(self.filename)

    def test_no_multiple_vector_ranges(self):
        cfg_ml = libsedml.SedDocument()

        mdl = cfg_ml.createModel()
        mdl.setName('test')

        sim = cfg_ml.createUniformTimeCourse()
        sim.setInitialTime(0.0)
        sim.setOutputStartTime(0.0)
        sim.setOutputEndTime(10.)
        sim.setNumberOfPoints(10)

        alg = sim.createAlgorithm()
        alg.setKisaoID("KISAO_0000352")
        param = alg.createAlgorithmParameter()
        param.setKisaoID("KISAO_0000488")
        param.setValue('1')

        task_ml = cfg_ml.createRepeatedTask()
        change_ml = task_ml.createTaskChange()
        change_ml.setTarget('x')
        change_ml.setMath(libsedml.parseFormula('1'))

        rangle_ml = task_ml.createVectorRange()
        rangle_ml.setValues([1, 2])

        libsedml.writeSedML(cfg_ml, self.filename)

        with self.assertRaises(sim_config.SedMlError):
            sim_config.SedMl.read(self.filename)

    def test_no_non_linear_uniform_range(self):
        cfg_ml = libsedml.SedDocument()

        mdl = cfg_ml.createModel()
        mdl.setName('test')

        sim = cfg_ml.createUniformTimeCourse()
        sim.setInitialTime(0.0)
        sim.setOutputStartTime(0.0)
        sim.setOutputEndTime(10.)
        sim.setNumberOfPoints(10)

        alg = sim.createAlgorithm()
        alg.setKisaoID("KISAO_0000352")
        param = alg.createAlgorithmParameter()
        param.setKisaoID("KISAO_0000488")
        param.setValue('1')

        task_ml = cfg_ml.createRepeatedTask()
        change_ml = task_ml.createTaskChange()
        change_ml.setTarget('x')
        change_ml.setMath(libsedml.parseFormula('1'))

        rangle_ml = task_ml.createUniformRange()
        rangle_ml.setStart(0)
        rangle_ml.setEnd(1)
        rangle_ml.setType('log')

        libsedml.writeSedML(cfg_ml, self.filename)

        with self.assertRaises(sim_config.SedMlError):
            sim_config.SedMl.read(self.filename)
