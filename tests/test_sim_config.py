""" Tests of WCSimulationConfig

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
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

from de_sim.simulation_config import SimulationConfig
from wc_sim.multialgorithm_errors import MultialgorithmError
from wc_sim.sim_config import WCSimulationConfig, Change, Perturbation, SedMl, SedMlError, SedMlWarning
from wc_utils.util.units import unit_registry
import wc_lang


class TestWCSimulationConfig(unittest.TestCase):

    def setUp(self):
        self.de_simulation_config = SimulationConfig(max_time=10)

    def test_simulation_configuration(self):
        """ Test simulation configuration correctly represented """

        # create configuration
        random_seed = 3
        ode_time_step = 2
        dfba_time_step = 5
        changes = [
            Change([
                ['reactions', {'id': 'rxn-1'}],
                ['rate_laws', {'direction': wc_lang.RateLawDirection.forward}],
                'expression',
                ['parameters', {'id': 'k_cat'}],
                'value',
            ], 1),
            Change([
                ['species', {'id': 'species-type-1[compartment-1]'}],
                'distribution_init_concentration',
                'mean',
            ], 2),
        ]
        perturbations = [
            Perturbation(Change([
                ['reactions', {'id': 'rxn-1'}],
                ['rate_laws', {}],
                'expression',
                ['parameters', {'id': 'k_cat'}],
                'value',
            ], 3), start_time=1),
            Perturbation(Change([
                ['species', {'id': 'species-1[compartment-1]'}],
                'distribution_init_concentration',
                'mean',
            ], 4), start_time=0, end_time=10),
        ]
        cfg = WCSimulationConfig(de_simulation_config=self.de_simulation_config, random_seed=random_seed,
                                 ode_time_step=ode_time_step, dfba_time_step=dfba_time_step,
                                 changes=changes, perturbations=perturbations)
        self.assertEqual(None, cfg.validate())
        # test __setattr__
        self.assertEqual(None, setattr(cfg, 'random_seed', 3))

        # check correctly stored configuration
        self.assertEqual(random_seed, cfg.random_seed)
        self.assertEqual(ode_time_step + 0.0, cfg.ode_time_step)
        self.assertEqual(dfba_time_step + 0.0, cfg.dfba_time_step)

        # no random_seed
        cfg = WCSimulationConfig(de_simulation_config=self.de_simulation_config)
        self.assertEqual(None, cfg.random_seed)

        # no lists
        cfg = WCSimulationConfig(self.de_simulation_config, 1)
        self.assertEqual([], cfg.changes)
        self.assertEqual([], cfg.perturbations)

    def test_simulation_configuration_validation(self):

        # random seed not an integer
        WCSimulationConfig(self.de_simulation_config, random_seed=1)
        with self.assertRaises(MultialgorithmError):
            WCSimulationConfig(self.de_simulation_config, random_seed='a')
        with self.assertRaises(MultialgorithmError):
            WCSimulationConfig(self.de_simulation_config, random_seed=1.5)

        # time steps not numbers
        with self.assertRaises(MultialgorithmError):
            WCSimulationConfig(self.de_simulation_config, random_seed=1, ode_time_step=set())

        with self.assertRaises(MultialgorithmError):
            WCSimulationConfig(self.de_simulation_config, random_seed=1, dfba_time_step={})

        # negative time steps
        cfg = WCSimulationConfig(self.de_simulation_config, random_seed=1, ode_time_step=0.1)
        cfg.validate_individual_fields()
        with self.assertRaises(MultialgorithmError):
            cfg = WCSimulationConfig(self.de_simulation_config, random_seed=1, ode_time_step=-0.1)
            cfg.validate_individual_fields()

        cfg = WCSimulationConfig(self.de_simulation_config, random_seed=1, dfba_time_step=0.1)
        cfg.validate_individual_fields()
        with self.assertRaises(MultialgorithmError):
            cfg = WCSimulationConfig(self.de_simulation_config, random_seed=1, dfba_time_step=-0.1)
            cfg.validate_individual_fields()

        cfg = WCSimulationConfig(self.de_simulation_config, random_seed=1, checkpoint_period=0.1)
        cfg.validate_individual_fields()
        with self.assertRaises(MultialgorithmError):
            cfg = WCSimulationConfig(self.de_simulation_config, random_seed=1, checkpoint_period=-10)
            cfg.validate_individual_fields()

        # simulation duration not multiple of steps
        with self.assertRaises(MultialgorithmError):
            cfg = WCSimulationConfig(self.de_simulation_config, random_seed=1, ode_time_step=3.0)
            cfg.validate()

        with self.assertRaises(MultialgorithmError):
            cfg = WCSimulationConfig(self.de_simulation_config, random_seed=1, dfba_time_step=3.0)
            cfg.validate()

    def test_semantically_equal(self):
        random_seed = 3
        ode_time_step = 2
        dfba_time_step = 5
        checkpoint_period = 1
        submodels_to_skip = ['ode_mdl', 'ssa_mdl']
        cfg = WCSimulationConfig(de_simulation_config=self.de_simulation_config,
                                 random_seed=random_seed,
                                 ode_time_step=ode_time_step,
                                 dfba_time_step=dfba_time_step,
                                 checkpoint_period=checkpoint_period,
                                 submodels_to_skip=submodels_to_skip)
        cfg_equal = WCSimulationConfig(de_simulation_config=self.de_simulation_config,
                                       random_seed=random_seed,
                                       ode_time_step=ode_time_step,
                                       dfba_time_step=dfba_time_step,
                                       checkpoint_period=checkpoint_period,
                                       submodels_to_skip=submodels_to_skip)
        self.assertTrue(cfg.semantically_equal(cfg_equal))
        self.assertTrue(cfg_equal.semantically_equal(cfg))

        # vary de_simulation_config
        cfg.de_simulation_config = SimulationConfig(max_time=100)
        self.assertFalse(cfg.semantically_equal(cfg_equal))
        cfg.de_simulation_config = self.de_simulation_config

        # vary submodels_to_skip
        cfg.submodels_to_skip = submodels_to_skip[1:]
        self.assertFalse(cfg.semantically_equal(cfg_equal))
        cfg.submodels_to_skip = submodels_to_skip

        # vary all numerical attributes
        for attr in ['random_seed', 'ode_time_step', 'dfba_time_step', 'checkpoint_period']:
            save = getattr(cfg, attr)
            setattr(cfg, attr, 2 * save)
            self.assertFalse(cfg.semantically_equal(cfg_equal))
            setattr(cfg, attr, save)

        # with values reset, WCSimulationConfig still semantically_equal
        self.assertTrue(cfg.semantically_equal(cfg_equal))

    def test_apply_changes(self):
        cfg = WCSimulationConfig(self.de_simulation_config, random_seed=1, ode_time_step=2)
        cfg.changes.append(Change([
            ['reactions', {'id': 'rxn_1'}],
            ['rate_laws', {'direction': wc_lang.RateLawDirection.forward}],
            'expression',
            ['parameters', {'id': 'k_cat'}],
            'value',
        ], 2.5))
        model = wc_lang.Model()
        comp_1 = model.compartments.create(id='comp_1')

        comp_1.init_density = model.parameters.create(id='density_comp_1', value=1100, units=unit_registry.parse_units('g l^-1'))
        volume = model.functions.create(id='volume_comp_1', units=unit_registry.parse_units('l'))
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
        rl_1 = model.rate_laws.create(reaction=rxn_1, direction=wc_lang.RateLawDirection.forward, units=unit_registry.parse_units('s^-1'))
        k_cat = model.parameters.create(id='k_cat', value=1., units=unit_registry.parse_units('s^-1'))
        K_m = model.parameters.create(id='K_m', value=1., units=unit_registry.parse_units('M'))
        Avogadro = model.parameters.create(id='Avogadro', value=scipy.constants.Avogadro, units=unit_registry.parse_units('molecule mol^-1'))

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

    def test_apply_perturbations(self):
        # todo: implement
        cfg = WCSimulationConfig(self.de_simulation_config, random_seed=1, ode_time_step=2)
        cfg.apply_perturbations(None)

    def test_get_num_time_steps(self):
        cfg = WCSimulationConfig(self.de_simulation_config, random_seed=1, ode_time_step=2)
        self.assertEqual(cfg.get_num_time_steps(), 5)


class TestPerturbation(unittest.TestCase):

    def test_perturbation(self):
        change = Change([['species', {'id': 'species_1[compartment_1]'}],
                          'distribution_init_concentration',
                          'mean',
                         ], 4)
        perturbation_0 = Perturbation(change, start_time=2, end_time=10)
        perturbation_1 = Perturbation(change, start_time=0, end_time=10)
        perturbation_2 = Perturbation(change, start_time=0, end_time=float('nan'))
        perturbation_3 = Perturbation(change, start_time=0, end_time=11)
        self.assertNotEqual(perturbation_1, 7)
        self.assertNotEqual(perturbation_0, perturbation_1)
        self.assertNotEqual(perturbation_1, perturbation_2)
        self.assertNotEqual(perturbation_2, perturbation_1)
        self.assertNotEqual(perturbation_1, perturbation_3)
        self.assertEqual(perturbation_1, perturbation_1)
        self.assertEqual(perturbation_2, perturbation_2)


class TestSedMlImportExport(unittest.TestCase):

    def setUp(self):
        self.de_simulation_config = SimulationConfig(max_time=10)

    def test_sedml_import_export(self):
        """ Test SED-ML import/export of simulation configurations """

        # create configuration
        random_seed = 3
        ode_time_step = 2.0
        changes = [
            Change([
                ['reactions', {'id': 'rxn-1'}],
                ['rate_laws', {'direction': wc_lang.RateLawDirection.forward}],
                'expression',
                ['parameters', {'id': 'k_cat'}],
                'value',
            ], 1),
            Change([
                ['species', {'id': 'species-1[compartment-1]'}],
                'distribution_init_concentration',
                'mean',
            ], 2),
        ]
        perturbations = [
            Perturbation(Change([
                ['reactions', {'id': 'rxn-1'}],
                ['rate_laws', {'direction': wc_lang.RateLawDirection.forward}],
                'expression',
                ['parameters', {'id': 'k_cat'}],
                'value',
            ], 3
            ), start_time=1),
            Perturbation(Change([
                ['species', {'id': 'species-1[compartment-1]'}],
                'distribution_init_concentration',
                'mean',
            ], 4,
            ), start_time=0, end_time=10),
        ]
        cfg = WCSimulationConfig(self.de_simulation_config, random_seed=random_seed,
                                 ode_time_step=ode_time_step, changes=changes,
                                 perturbations=perturbations)

        # generate temporary file
        file_name = tempfile.mktemp()

        # export configuration
        SedMl.write(cfg, file_name)

        # import configuration
        cfg2 = SedMl.read(file_name)

        # check sim_config correctly imported/exported
        # self.assertEqual(max_time, cfg2.max_time)
        self.assertEqual(ode_time_step, cfg2.ode_time_step)
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
        SedMlError('msg')

    def test_warning(self):
        SedMlWarning('msg')


class TestSedMlValidation(unittest.TestCase):

    def setUp(self):
        _, self.filename = tempfile.mkstemp()
        warnings.simplefilter("ignore")

    def tearDown(self):
        os.remove(self.filename)

    def test_no_model(self):
        cfg_ml = libsedml.SedDocument()
        libsedml.writeSedML(cfg_ml, self.filename)

        with self.assertRaises(SedMlError):
            SedMl.read(self.filename)

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

        pytest.warns(SedMlWarning, SedMl.read, self.filename)

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

        with self.assertRaises(SedMlError):
            SedMl.read(self.filename)

    def test_no_simulations(self):
        cfg_ml = libsedml.SedDocument()

        mdl = cfg_ml.createModel()
        mdl.setName('test')

        libsedml.writeSedML(cfg_ml, self.filename)

        with self.assertRaises(SedMlError):
            SedMl.read(self.filename)

    def test_no_initial_time(self):
        cfg_ml = libsedml.SedDocument()

        mdl = cfg_ml.createModel()
        mdl.setName('test')

        sim = cfg_ml.createUniformTimeCourse()

        libsedml.writeSedML(cfg_ml, self.filename)

        with self.assertRaises(SedMlError):
            SedMl.read(self.filename)

    def test_no_output_start_time(self):
        cfg_ml = libsedml.SedDocument()

        mdl = cfg_ml.createModel()
        mdl.setName('test')

        sim = cfg_ml.createUniformTimeCourse()
        sim.setInitialTime(0.0)

        libsedml.writeSedML(cfg_ml, self.filename)

        with self.assertRaises(SedMlError):
            SedMl.read(self.filename)

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

        with self.assertRaises(SedMlError):
            SedMl.read(self.filename)

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

        with self.assertRaisesRegex(SedMlError, 'random_seed must be an integer'):
            SedMl.read(self.filename)

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

        with self.assertRaisesRegex(SedMlError, 'random_seed must be an integer'):
            SedMl.read(self.filename)

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

        with self.assertRaisesRegex(SedMlError, 'Algorithm parameter KISAO ids must be unique'):
            SedMl.read(self.filename)

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

        with self.assertRaises(SedMlError):
            SedMl.read(self.filename)

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

        with self.assertRaises(SedMlError):
            SedMl.read(self.filename)

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

        with self.assertRaises(SedMlError):
            SedMl.read(self.filename)

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

        with self.assertRaises(SedMlError):
            SedMl.read(self.filename)

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

        with self.assertRaises(SedMlError):
            SedMl.read(self.filename)

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

        with self.assertRaises(SedMlError):
            SedMl.read(self.filename)

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

        with self.assertRaises(SedMlError):
            SedMl.read(self.filename)

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

        with self.assertRaises(SedMlError):
            SedMl.read(self.filename)

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

        with self.assertRaises(SedMlError):
            SedMl.read(self.filename)

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

        with self.assertRaises(SedMlError):
            SedMl.read(self.filename)

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

        with self.assertRaises(SedMlError):
            SedMl.read(self.filename)

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

        with self.assertRaises(SedMlError):
            SedMl.read(self.filename)

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

        with self.assertRaises(SedMlError):
            SedMl.read(self.filename)

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

        with self.assertRaises(SedMlError):
            SedMl.read(self.filename)
