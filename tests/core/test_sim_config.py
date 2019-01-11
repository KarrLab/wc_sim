""" Tests of sim_config.SimulationConfig

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2017-07-25
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

import libsedml
import numpy
import os
import pytest
import scipy.constants
import tempfile
import unittest
import warnings
import wc_lang

from wc_sim.core import sim_config


class TestSimulationConfig(unittest.TestCase):
    """ Test simulation configuration """

    def test_simulation_configuration(self):
        """ Test simulation configuration correctly represented """

        # create configuration
        time_max = 100
        time_step = 2
        changes = [
            sim_config.Change([
                ['reactions', {'id': 'rxn-1'}],
                ['rate_laws', {'direction': wc_lang.RateLawDirection.forward}],
                'expression',
                ['parameters', {'id': 'k_cat'}],
                'mean',
            ], 1),
            sim_config.Change([
                ['species', {'id': 'species-type-1[compartment-1]'}],
                'distribution_init_concentration',
                'mean',
            ], 2),
        ]
        perturbations = [
            sim_config.Perturbation(sim_config.Change([
                ['reactions', {'id': 'rxn-1'}],
                ['rate_laws', {}],
                'expression',
                ['parameters', {'id': 'k_cat'}],
                'mean',
            ], 3), start_time=1),
            sim_config.Perturbation(sim_config.Change([
                ['species', {'id': 'species-1[compartment-1]'}],
                'distribution_init_concentration',
                'mean',
            ], 4), start_time=0, end_time=10),
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

    def test_apply_changes(self):
        cfg = sim_config.SimulationConfig(time_max=100, time_step=2)
        cfg.changes.append(sim_config.Change([
            ['reactions', {'id': 'rxn_1'}],
            ['rate_laws', {'direction': wc_lang.RateLawDirection.forward}],
            'expression',
            ['parameters', {'id': 'k_cat'}],
            'value',
        ], 2.5))
        model = wc_lang.Model()
        comp_1 = model.compartments.create(id='comp_1')

        comp_1.init_density = model.parameters.create(id='density_comp_1', value=1100, units='g l^-1')
        volume = model.functions.create(id='volume_comp_1', units='l')
        volume.expression, error = wc_lang.FunctionExpression.deserialize('{} / {}'.format(comp_1.id, comp_1.init_density.id), {
            wc_lang.Compartment: {comp_1.id: comp_1},
            wc_lang.Parameter: {comp_1.init_density.id: comp_1.init_density},
            })
        assert error is None, str(error)

        species_type_1 = model.species_types.create(id='species_type_1')
        species_1_comp_1 = model.species.create(species_type=species_type_1, compartment=comp_1)
        species_1_comp_1.id = species_1_comp_1.gen_id()
        submodel = model.submodels.create(id='submodel_1')
        rxn_1 = model.reactions.create(id='rxn_1', submodel=submodel)
        rl_1 = model.rate_laws.create(reaction=rxn_1, direction=wc_lang.RateLawDirection.forward, units=wc_lang.ReactionRateUnit['s^-1'])
        k_cat = model.parameters.create(id='k_cat', value=1., units='s^-1')
        K_m = model.parameters.create(id='K_m', value=1., units='M')
        Avogadro = model.parameters.create(id='Avogadro', value=scipy.constants.Avogadro, units='molecule mol^-1')

        objects = {
            wc_lang.Compartment: {
                comp_1.id: comp_1,
            },
            wc_lang.Species: {
                species_1_comp_1.id: species_1_comp_1,
            },
            wc_lang.Function: {
                volume.id: volume,
            },
            wc_lang.Parameter: {
                k_cat.id: k_cat,
                K_m.id: K_m,
                Avogadro.id: Avogadro,                
            },            
        }
        rl_1.expression, errors = wc_lang.RateLawExpression.deserialize(
            '{} * {} / ({} * {} * {} + {})'.format(k_cat.id, species_1_comp_1.id,
                                                   K_m.id, Avogadro.id, volume.id, species_1_comp_1.id), objects)
        self.assertEqual(errors, None, str(errors))

        cfg.apply_changes(model)

        self.assertEqual(k_cat.value, 2.5)

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
            sim_config.Change([
                ['reactions', {'id': 'rxn-1'}],
                ['rate_laws', {'direction': wc_lang.RateLawDirection.forward}],
                'expression',
                ['parameters', {'id': 'k_cat'}],
                'mean',
            ], 1),
            sim_config.Change([
                ['species', {'id': 'species-1[compartment-1]'}],
                'distribution_init_concentration',
                'mean',
            ], 2),
        ]
        perturbations = [
            sim_config.Perturbation(sim_config.Change([
                ['reactions', {'id': 'rxn-1'}],
                ['rate_laws', {'direction': wc_lang.RateLawDirection.forward}],
                'expression',
                ['parameters', {'id': 'k_cat'}],
                'mean',
            ], 3
            ), start_time=1),
            sim_config.Perturbation(sim_config.Change([
                ['species', {'id': 'species-1[compartment-1]'}],
                'distribution_init_concentration',
                'mean',
            ], 4,
            ), start_time=0, end_time=10),
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
        self.assertEqual(changes[0], cfg2.changes[0])
        self.assertEqual(changes[1], cfg2.changes[1])

        self.assertEqual(len(perturbations), len(cfg2.perturbations))
        self.assertEqual(perturbations[0].change, cfg2.perturbations[0].change)
        self.assertEqual(perturbations[1].change, cfg2.perturbations[1].change)
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

        with self.assertRaisesRegex(sim_config.SedMlError, 'random_seed must be an integer'):
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

        with self.assertRaisesRegex(sim_config.SedMlError, 'random_seed must be an integer'):
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

        with self.assertRaisesRegex(sim_config.SedMlError, 'Algorithm parameter KISAO ids must be unique'):
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
