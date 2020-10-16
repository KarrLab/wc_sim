"""
:Author: Yin Hoon Chew <yinhoon.chew@mssm.edu>
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2020-08-05
:Copyright: 2016-2020, Karr Lab
:License: MIT
"""

import conv_opt
import copy
import math
import numpy
import os
import re
import scipy.constants
import unittest
import wc_lang

from de_sim.simulation_config import SimulationConfig
from wc_onto import onto as wc_ontology
from wc_sim.message_types import RunFba
from wc_sim.multialgorithm_errors import DynamicMultialgorithmError, MultialgorithmError
from wc_sim.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.sim_config import WCSimulationConfig
from wc_sim.submodels.dfba import DfbaSubmodel
from wc_utils.util.environ import EnvironUtils, ConfigEnvDict
from wc_utils.util.units import unit_registry


class TestDfbaSubmodel(unittest.TestCase):

    def setUp(self):
        # APG: YH, I recommend that this model be described in a WC Lang workbook, which would be more easily
        # read and edited by some people
        # Also, using it would ensure that dFBA works end-to-end from a workbook, and one could add extra
        # spreadsheet(s) that contain the calculations that predict the FBA solution(s)
        # the WC Lang workbook can be easily created by writing this model
        Av = scipy.constants.Avogadro
        model = self.model = wc_lang.Model()

        # Create compartment
        init_volume_c = wc_lang.core.InitVolume(distribution=wc_ontology['WC:normal_distribution'],
                    mean=5e-13, std=0)
        c = model.compartments.create(id='c', init_volume=init_volume_c)
        c.init_density = model.parameters.create(id='density_c', value=1.,
                units=unit_registry.parse_units('g l^-1'))
        volume_c = model.functions.create(id='volume_c', units=unit_registry.parse_units('l'))
        volume_c.expression, error = wc_lang.FunctionExpression.deserialize(f'{c.id} / {c.init_density.id}', {
                wc_lang.Compartment: {c.id: c},
                wc_lang.Parameter: {c.init_density.id: c.init_density},
                })
        assert error is None, str(error)

        init_volume_n = wc_lang.core.InitVolume(distribution=wc_ontology['WC:normal_distribution'],
                    mean=5e-13, std=0)
        # APG: role does compartment 'n' play? it doesn't have a name, or a nesting (parent)
        # relationship with compartment 'c' and none of the reactions transform species in 'n'.
        n = model.compartments.create(id='n', init_volume=init_volume_n)
        n.init_density = model.parameters.create(id='density_n', value=1.,
                units=unit_registry.parse_units('g l^-1'))
        volume_n = model.functions.create(id='volume_n', units=unit_registry.parse_units('l'))
        volume_n.expression, error = wc_lang.FunctionExpression.deserialize(f'{n.id} / {n.init_density.id}', {
                wc_lang.Compartment: {n.id: n},
                wc_lang.Parameter: {n.init_density.id: n.init_density},
                })
        assert error is None, str(error)

        self.cell_volume = init_volume_c.mean + init_volume_n.mean

        # Create metabolites
        for m in [1,2,3]:
            metabolite_st = model.species_types.create(id='m{}'.format(m), type=wc_ontology['WC:metabolite'],
                structure=wc_lang.ChemicalStructure(molecular_weight=1.))
            for comp in [c, n]:
                metabolite_species = model.species.create(species_type=metabolite_st, compartment=comp)
                metabolite_species.id = metabolite_species.gen_id()
                conc_model = model.distribution_init_concentrations.create(species=metabolite_species, mean=100., std=0.,
                    units=unit_registry.parse_units('molecule'))
                conc_model.id = conc_model.gen_id()

        # Create enzymes
        for i, conc in {'enzyme1': 1, 'enzyme2': 1, 'enzyme3': 1}.items():
            enzyme_st = model.species_types.create(id=i, type=wc_ontology['WC:pseudo_species'],
                structure=wc_lang.ChemicalStructure(molecular_weight=10.))
            enzyme_species = model.species.create(species_type=enzyme_st, compartment=c)
            enzyme_species.id = enzyme_species.gen_id()
            conc_model = model.distribution_init_concentrations.create(species=enzyme_species, mean=conc, std=0.,
                units=unit_registry.parse_units('molecule'))
            conc_model.id = conc_model.gen_id()

        # Create reactions in metabolism submodel
        self.submodel = submodel = model.submodels.create(
            id='metabolism', framework=wc_ontology['WC:dynamic_flux_balance_analysis'])

        ex1 = submodel.reactions.create(id='ex_m1', reversible=False, model=model)
        ex1.flux_bounds = wc_lang.FluxBounds(min=100./Av, max=120./Av)
        ex1.participants.append(model.species.get_one(id='m1[c]').species_coefficients.get_or_create(coefficient=1))

        ex2 = submodel.reactions.create(id='ex_m2', reversible=False, model=model)
        ex2.flux_bounds = wc_lang.FluxBounds(min=100./Av, max=120./Av)
        ex2.participants.append(model.species.get_one(id='m2[c]').species_coefficients.get_or_create(coefficient=1))

        ex3 = submodel.reactions.create(id='ex_m3', reversible=False, model=model)
        ex3.flux_bounds = wc_lang.FluxBounds(min=0., max=0.)
        ex3.participants.append(model.species.get_one(id='m3[c]').species_coefficients.get_or_create(coefficient=1))

        r1 = submodel.reactions.create(id='r1', reversible=True, model=model)
        r1.participants.append(model.species.get_one(id='m1[c]').species_coefficients.get_or_create(coefficient=-1))
        r1.participants.append(model.species.get_one(id='m2[c]').species_coefficients.get_or_create(coefficient=-1))
        r1.participants.append(model.species.get_one(id='m3[c]').species_coefficients.get_or_create(coefficient=1))
        r1_rate_law_expression1, error = wc_lang.RateLawExpression.deserialize('k_cat_r1_forward_enzyme3 * enzyme3[c]', {
            wc_lang.Parameter: {'k_cat_r1_forward_enzyme3': model.parameters.create(id='k_cat_r1_forward_enzyme3', value=1.)},
            wc_lang.Species: {'enzyme3[c]': model.species.get_one(id='enzyme3[c]')},
            })
        assert error is None, str(error)
        r1_model_rate_law1 = model.rate_laws.create(
            expression=r1_rate_law_expression1,
            reaction=r1,
            direction=wc_lang.RateLawDirection['forward'])
        r1_model_rate_law1.id = r1_model_rate_law1.gen_id()
        r1_rate_law_expression2, error = wc_lang.RateLawExpression.deserialize('k_cat_r1_backward_enzyme3 * enzyme3[c]', {
            wc_lang.Parameter: {'k_cat_r1_backward_enzyme3': model.parameters.create(id='k_cat_r1_backward_enzyme3', value=1.)},
            wc_lang.Species: {'enzyme3[c]': model.species.get_one(id='enzyme3[c]')},
            })
        assert error is None, str(error)
        r1_model_rate_law2 = model.rate_laws.create(
            expression=r1_rate_law_expression2,
            reaction=r1,
            direction=wc_lang.RateLawDirection['backward'])
        r1_model_rate_law2.id = r1_model_rate_law2.gen_id()

        r2 = submodel.reactions.create(id='r2', reversible=True, model=model)
        r2.participants.append(model.species.get_one(id='m1[c]').species_coefficients.get_or_create(coefficient=-1))
        r2.participants.append(model.species.get_one(id='m2[c]').species_coefficients.get_or_create(coefficient=1))
        r2_rate_law_expression1, error = wc_lang.RateLawExpression.deserialize('k_cat_r2_forward_enzyme1 * enzyme1[c] + k_cat_r2_forward_enzyme2 * enzyme2[c]', {
            wc_lang.Parameter: {'k_cat_r2_forward_enzyme1': model.parameters.create(id='k_cat_r2_forward_enzyme1', value=1.),
                                'k_cat_r2_forward_enzyme2': model.parameters.create(id='k_cat_r2_forward_enzyme2', value=2.)},
            wc_lang.Species: {'enzyme1[c]': model.species.get_one(id='enzyme1[c]'),
                              'enzyme2[c]': model.species.get_one(id='enzyme2[c]')},
            })
        assert error is None, str(error)
        r2_model_rate_law1 = model.rate_laws.create(
            expression=r2_rate_law_expression1,
            reaction=r2,
            direction=wc_lang.RateLawDirection['forward'])
        r2_model_rate_law1.id = r2_model_rate_law1.gen_id()
        r2_rate_law_expression2, error = wc_lang.RateLawExpression.deserialize('k_cat_r2_backward_enzyme1 * enzyme1[c] + k_cat_r2_backward_enzyme2 * enzyme2[c]', {
            wc_lang.Parameter: {'k_cat_r2_backward_enzyme1': model.parameters.create(id='k_cat_r2_backward_enzyme1', value=1.),
                                'k_cat_r2_backward_enzyme2': model.parameters.create(id='k_cat_r2_backward_enzyme2', value=1.)},
            wc_lang.Species: {'enzyme1[c]': model.species.get_one(id='enzyme1[c]'),
                              'enzyme2[c]': model.species.get_one(id='enzyme2[c]')},
            })
        assert error is None, str(error)
        r2_model_rate_law2 = model.rate_laws.create(
            expression=r2_rate_law_expression2,
            reaction=r2,
            direction=wc_lang.RateLawDirection['backward'])
        r2_model_rate_law2.id = r2_model_rate_law2.gen_id()

        r3 = submodel.reactions.create(id='r3', reversible=False, model=model)
        r3.participants.append(model.species.get_one(id='m1[c]').species_coefficients.get_or_create(coefficient=-2))
        r3.participants.append(model.species.get_one(id='m3[c]').species_coefficients.get_or_create(coefficient=1))
        r3_rate_law_expression, error = wc_lang.RateLawExpression.deserialize('k_cat_r3_forward_enzyme2 * enzyme2[c]', {
            wc_lang.Parameter: {'k_cat_r3_forward_enzyme2': model.parameters.create(id='k_cat_r3_forward_enzyme2', value=5.)},
            wc_lang.Species: {'enzyme2[c]': model.species.get_one(id='enzyme2[c]')},
            })
        assert error is None, str(error)
        r3_model_rate_law = model.rate_laws.create(
            expression=r3_rate_law_expression,
            reaction=r3,
            direction=wc_lang.RateLawDirection['forward'])
        r3_model_rate_law.id = r3_model_rate_law.gen_id()

        r4 = submodel.reactions.create(id='r4', reversible=False, model=model)
        r4.participants.append(model.species.get_one(id='m2[c]').species_coefficients.get_or_create(coefficient=-2))
        r4.participants.append(model.species.get_one(id='m3[c]').species_coefficients.get_or_create(coefficient=1))
        r4 = model.reactions.get_one(id='r4')
        r4_rate_law_expression, error = wc_lang.RateLawExpression.deserialize('k_cat_r4_forward_enzyme1 * enzyme1[c]', {
            wc_lang.Parameter: {'k_cat_r4_forward_enzyme1': model.parameters.create(id='k_cat_r4_forward_enzyme1', value=6.)},
            wc_lang.Species: {'enzyme1[c]': model.species.get_one(id='enzyme1[c]')},
            })
        assert error is None, str(error)
        r4_model_rate_law = model.rate_laws.create(
            expression=r4_rate_law_expression,
            reaction=r4,
            direction=wc_lang.RateLawDirection['forward'])
        r4_model_rate_law.id = r4_model_rate_law.gen_id()

        biomass_rxn = submodel.dfba_obj_reactions.create(id='biomass_reaction', model=model)
        biomass_rxn.dfba_obj_species.append(model.species.get_one(id='m3[c]').dfba_obj_species.get_or_create(model=model, value=-1))
        submodel.dfba_obj = wc_lang.DfbaObjective(model=model)
        submodel.dfba_obj.id = submodel.dfba_obj.gen_id()
        obj_expression = biomass_rxn.id
        dfba_obj_expression, error = wc_lang.DfbaObjectiveExpression.deserialize(
            obj_expression, {wc_lang.DfbaObjReaction: {biomass_rxn.id: biomass_rxn}})
        assert error is None, str(error)
        submodel.dfba_obj.expression = dfba_obj_expression

        self.dfba_submodel_options = {
            'dfba_bound_scale_factor': 1e2,
            'dfba_coef_scale_factor': 10,
            'solver': 'cplex',
            'presolve': 'on',
            'optimization_type': 'maximize',
            'flux_bounds_volumetric_compartment_id': 'wc',
            'solver_options': {
                'cplex': {
                    'parameters': {
                        'emphasis': {
                            'numerical': 1,
                        },
                        'read': {
                            'scale': 1,
                        },
                    },
                },
            }
        }

        self.dfba_submodel_1 = self.make_dfba_submodel(self.model,
            submodel_options=self.dfba_submodel_options)

        scale_factor = self.dfba_submodel_options['dfba_bound_scale_factor']
        self.expected_rxn_flux_bounds = {
            # map: rxn id: (lower bound, upper bound)
            'ex_m1': (100. * self.cell_volume * scale_factor, 120. * self.cell_volume * scale_factor),
            'ex_m2': (100. * self.cell_volume * scale_factor, 120. * self.cell_volume * scale_factor),
            'ex_m3': (0., 0.),
            'r1': (-1. * 1 * scale_factor, 1. * 1 * scale_factor),
            'r2': (-(1. * 1 * scale_factor + 1. * 1 * scale_factor),
                   1. * 1 * scale_factor + 2. * 1 * scale_factor),
            'r3': (0., 5. * 1 * scale_factor),
            'r4': (0., 6. * 1 * scale_factor),
        }
        # APG: try to write WC Lang model workbook -- worked, but encountered multiple warnings
        # from wc_lang.io import Reader, Writer
        # lang_filename = 'dfba_test_model.xlsx'
        # print(f'writing {lang_filename}')
        # Writer().run(lang_filename, model, protected=False)

    def make_dfba_submodel(self, model, dfba_time_step=1.0, submodel_id='metabolism',
                           submodel_options=None, dfba_obj_with_regular_rxn=False):
        """ Make a MultialgorithmSimulation from a wc lang model, and return its dFBA submodel

        Args:
            model (:obj:`wc_lang.Model`): the model
            dfba_time_step (:obj:`float`, optional): the dFBA submodel's timestep
            submodel_id (:obj:`wc_lang.Model`, optional): the dFBA submodel's id
            submodel_options (:obj:`type`, optional): the dynamic dFBA submodel's options
            dfba_obj_with_regular_rxn (:obj:`bool`, optional): whether to give the submodel
                a dfba_obj that has a regular reaction

        Returns:
            :obj:`DfbaSubmodel`: a dynamic dFBA submodel
        """
        if dfba_obj_with_regular_rxn:
            obj_expression = f"biomass_reaction + 2 * r2"
            dfba_obj_expression, error = wc_lang.DfbaObjectiveExpression.deserialize(
                obj_expression, {
                    wc_lang.DfbaObjReaction: {
                        'biomass_reaction': model.dfba_obj_reactions.get_one(id='biomass_reaction')
                        },
                    wc_lang.Reaction:{
                        'r2': model.reactions.get_one(id='r2')},
                    })
            assert error is None, str(error)
            self.submodel.dfba_obj.expression = dfba_obj_expression
            self.dfba_submodel_options['optimization_type'] = 'minimize'

        self.dfba_time_step = dfba_time_step

        de_simulation_config = SimulationConfig(max_time=10)
        wc_sim_config = WCSimulationConfig(de_simulation_config, dfba_time_step=dfba_time_step)
        multialgorithm_simulation = MultialgorithmSimulation(model,
                                                             wc_sim_config,
                                                             options={'DfbaSubmodel':
                                                                      dict(options=submodel_options)})
        simulator, dynamic_model = multialgorithm_simulation.build_simulation()
        simulator.initialize()

        return dynamic_model.dynamic_submodels[submodel_id]

    ### test low level methods ###
    def test_dfba_submodel_init(self):
        self.assertEqual(self.dfba_submodel_1.time_step, self.dfba_time_step)
        # test options
        self.assertEqual(self.dfba_submodel_1.dfba_solver_options, self.dfba_submodel_options)

        # cover other branches in DfbaSubmodel.__init__()
        value = 3
        dfba_submodel = self.make_dfba_submodel(self.model,
                                                submodel_options=dict(dfba_bound_scale_factor=value))
        self.assertEqual(dfba_submodel.dfba_solver_options['dfba_bound_scale_factor'], value)
        dfba_submodel = self.make_dfba_submodel(self.model,
                                                submodel_options=dict(dfba_coef_scale_factor=value))
        self.assertEqual(dfba_submodel.dfba_solver_options['dfba_coef_scale_factor'], value)

        # test exceptions
        bad_dfba_time_step = -2
        with self.assertRaisesRegexp(MultialgorithmError,
            'DfbaSubmodel metabolism: time_step must be positive, but is {}'.format(bad_dfba_time_step)):
            self.make_dfba_submodel(self.model, dfba_time_step=bad_dfba_time_step)

        with self.assertRaisesRegexp(MultialgorithmError,
                                     "DfbaSubmodel metabolism: time_step must be a number but is"):
            self.make_dfba_submodel(self.model, dfba_time_step=None)

        bad_dfba_bound_scale_factor = copy.deepcopy(self.dfba_submodel_options)
        bad_dfba_bound_scale_factor['dfba_bound_scale_factor'] = -2.
        with self.assertRaisesRegexp(MultialgorithmError,
                                     "DfbaSubmodel metabolism: dfba_bound_scale_factor must"
                                     " be larger than zero but is -2."):
            self.make_dfba_submodel(self.model, submodel_options=bad_dfba_bound_scale_factor)

        bad_dfba_coef_scale_factor = copy.deepcopy(self.dfba_submodel_options)
        bad_dfba_coef_scale_factor['dfba_coef_scale_factor'] = -2.
        with self.assertRaisesRegexp(MultialgorithmError,
                                     "DfbaSubmodel metabolism: dfba_coef_scale_factor must"
                                     " be larger than zero but is -2."):
            self.make_dfba_submodel(self.model, submodel_options=bad_dfba_coef_scale_factor)

        bad_solver = copy.deepcopy(self.dfba_submodel_options)
        bad_solver['solver'] = 'cp'
        with self.assertRaisesRegexp(MultialgorithmError,
                                     f"DfbaSubmodel metabolism: {bad_solver['solver']} "
                                     f"is not a valid Solver"):
            self.make_dfba_submodel(self.model, submodel_options=bad_solver)

        bad_presolve = copy.deepcopy(self.dfba_submodel_options)
        bad_presolve['presolve'] = 'of'
        with self.assertRaisesRegexp(MultialgorithmError,
                                     "DfbaSubmodel metabolism: {} is not a valid Presolve option".format(
                                         bad_presolve['presolve'])):
            self.make_dfba_submodel(self.model, submodel_options=bad_presolve)

        bad_solver_options = copy.deepcopy(self.dfba_submodel_options)
        bad_solver_options['solver_options'] = {'gurobi': {'parameters': {}}}
        with self.assertRaisesRegexp(MultialgorithmError,
                                     "DfbaSubmodel metabolism: the solver key in"
                                        f" solver_options is not the same as the selected solver 'cplex'"):
            self.make_dfba_submodel(self.model, submodel_options=bad_solver_options)

        bad_optimization_type = copy.deepcopy(self.dfba_submodel_options)
        bad_optimization_type['optimization_type'] = 'bad'
        with self.assertRaisesRegexp(MultialgorithmError,
                                     "DfbaSubmodel metabolism: the optimization_type in"
                                     f" options can only take 'maximize', 'max', 'minimize' or 'min' "
                                     f"as value but is 'bad'"):
            self.make_dfba_submodel(self.model, submodel_options=bad_optimization_type)

        bad_flux_comp_id = copy.deepcopy(self.dfba_submodel_options)
        bad_flux_comp_id['flux_bounds_volumetric_compartment_id'] = 'bad'
        with self.assertRaisesRegexp(MultialgorithmError,
                                     "DfbaSubmodel metabolism: the user-provided"
                                        f" flux_bounds_volumetric_compartment_id"
                                        f" '{bad_flux_comp_id['flux_bounds_volumetric_compartment_id']}'"
                                        f" is not the ID of a compartment in the model"):
            self.make_dfba_submodel(self.model, submodel_options=bad_flux_comp_id)

    def test_set_up_optimizations(self):
        dfba_submodel = self.dfba_submodel_1
        self.assertTrue(set(dfba_submodel.species_ids) == dfba_submodel.species_ids_set \
            == set(dfba_submodel.adjustments.keys()))
        self.assertEqual(dfba_submodel.populations.shape, ((len(dfba_submodel.species_ids), )))
        self.assertEqual(dfba_submodel.reaction_fluxes,
            {'ex_m1': None, 'ex_m2': None, 'ex_m3': None, 'r1': None, 'r2': None, 'r3': None, 'r4': None})

    def test_set_up_dfba_submodel(self):
        self.dfba_submodel_1.set_up_dfba_submodel()

        self.assertEqual(len(self.dfba_submodel_1._conv_variables), len(self.submodel.reactions) + 1)
        self.assertEqual('biomass_reaction' in self.dfba_submodel_1._conv_variables, True)
        for k,v in self.dfba_submodel_1._conv_variables.items():
            self.assertEqual(k, v.name)

        expected_stoch_consts = {
            'm1[c]': {'ex_m1': 1, 'r1': -1, 'r2': -1, 'r3': -2},
            'm2[c]': {'ex_m2': 1, 'r1': -1, 'r2': 1, 'r4': -2},
            'm3[c]': {'ex_m3': 1, 'r1': 1, 'r3': 1, 'r4': 1,
                      'biomass_reaction': -1 * self.dfba_submodel_options['dfba_coef_scale_factor']},
        }
        test_results = {species_id: {lt.variable.name: lt.coefficient for lt in linear_terms}
                        for species_id, linear_terms
                            in self.dfba_submodel_1._conv_metabolite_matrices.items()}
        self.assertEqual(test_results, expected_stoch_consts)

        self.assertEqual(len(self.dfba_submodel_1.get_conv_model().variables),
            len(self.dfba_submodel_1._conv_variables.values()))
        for conv_opt_var in self.dfba_submodel_1._conv_variables.values():
            self.assertEqual(conv_opt_var in self.dfba_submodel_1.get_conv_model().variables, True)

        for constr in self.dfba_submodel_1.get_conv_model().constraints:
            if constr.name in expected_stoch_consts:
                self.assertEqual({linear_term.variable.name:linear_term.coefficient
                                  for linear_term in constr.terms},
                                 expected_stoch_consts[constr.name])
                self.assertEqual(constr.upper_bound, 0.)
                self.assertEqual(constr.lower_bound, 0.)

        self.assertEqual(self.dfba_submodel_1._dfba_obj_rxn_ids, ['biomass_reaction'])
        self.assertEqual({i.variable.name:i.coefficient
                          for i in self.dfba_submodel_1.get_conv_model().objective_terms},
                         {'biomass_reaction': 1})
        self.assertEqual(self.dfba_submodel_1.get_conv_model().objective_direction,
                         conv_opt.ObjectiveDirection.maximize)

        # Test model where the objective function is made of dfba objective reactions and
        # network reactions and is minimized
        dfba_submodel_2 = self.make_dfba_submodel(self.model,
                                                  submodel_options=self.dfba_submodel_options,
                                                  dfba_obj_with_regular_rxn=True)

        self.assertEqual(len(dfba_submodel_2._conv_variables), len(self.submodel.reactions) + 1)
        self.assertEqual('biomass_reaction' in dfba_submodel_2._conv_variables, True)
        for rxn_id, conv_opt_var in dfba_submodel_2._conv_variables.items():
            self.assertEqual(rxn_id, conv_opt_var.name)

        test_results = {species_id: {lt.variable.name: lt.coefficient for lt in linear_terms}
                        for species_id, linear_terms
                            in dfba_submodel_2._conv_metabolite_matrices.items()}
        self.assertEqual(test_results, expected_stoch_consts)

        self.assertEqual(len(dfba_submodel_2.get_conv_model().variables),
                         len(dfba_submodel_2._conv_variables.values()))

        for conv_opt_var in dfba_submodel_2._conv_variables.values():
            self.assertEqual(conv_opt_var in dfba_submodel_2.get_conv_model().variables, True)

        for constr in dfba_submodel_2.get_conv_model().constraints:
            if constr.name in expected_stoch_consts:
                self.assertEqual({linear_term.variable.name:linear_term.coefficient
                                  for linear_term in constr.terms},
                                 expected_stoch_consts[constr.name])
                self.assertEqual(constr.upper_bound, 0.)
                self.assertEqual(constr.lower_bound, 0.)

        self.assertEqual(dfba_submodel_2._dfba_obj_rxn_ids, ['biomass_reaction'])
        self.assertEqual({objective_term.variable.name: objective_term.coefficient
                          for objective_term in dfba_submodel_2.get_conv_model().objective_terms},
            {'biomass_reaction': 1, 'r2': 2})
        self.assertEqual(dfba_submodel_2.get_conv_model().objective_direction, conv_opt.ObjectiveDirection.minimize)

    def test_get_species_and_stoichiometry(self):
        def get_species(id):
            return self.model.species.get_one(id=id)

        # test with a dFBA obj that contains a regular reaction
        dfba_submodel_2 = self.make_dfba_submodel(self.model,
                                                  submodel_options=self.dfba_submodel_options,
                                                  dfba_obj_with_regular_rxn=True)
        expected = dict(r2={get_species('m1[c]'): -1.,
                            get_species('m2[c]'): 1.},
                        biomass_reaction={get_species('m3[c]'): -1.})
        for rxn_class in dfba_submodel_2.dfba_obj_expr.related_objects:
            for rxn_id, rxn in dfba_submodel_2.dfba_obj_expr.related_objects[rxn_class].items():
                self.assertEqual(DfbaSubmodel._get_species_and_stoichiometry(rxn),
                                 expected[rxn_id])

        # also test reactions with the same species on both sides
        r2 = self.model.reactions.get_one(id='r2')
        r2.participants.append(get_species(id='m1[c]').species_coefficients.get_or_create(coefficient=1))
        dfba_submodel_2 = self.make_dfba_submodel(self.model,
                                                  submodel_options=self.dfba_submodel_options,
                                                  dfba_obj_with_regular_rxn=True)
        expected = dict(r2={get_species('m2[c]'): 1.},
                        biomass_reaction={get_species('m3[c]'): -1.})
        for rxn_class in dfba_submodel_2.dfba_obj_expr.related_objects:
            for rxn_id, rxn in dfba_submodel_2.dfba_obj_expr.related_objects[rxn_class].items():
                self.assertEqual(DfbaSubmodel._get_species_and_stoichiometry(rxn),
                                 expected[rxn_id])

    def test_species_id_conversions(self):
        species_id = 's[c]'
        round_trip_id = DfbaSubmodel.species_id_with_brkts(DfbaSubmodel.species_id_without_brkts(species_id))
        self.assertEqual(species_id, round_trip_id)

    def test_initialize_neg_species_pop_constraints(self):
        def get_const_name(species_id):
            """ Get the name of the negative species constraint for species `species_id` """
            return DfbaSubmodel.gen_neg_species_pop_constraint_id(species_id)

        def get_rxn_set(rxn_ids):
            """ Get the set of reactions specified by their ids in `rxn_ids` """
            return {self.model.reactions.get_one(id=rxn_id) for rxn_id in rxn_ids}
            rxn_set = set()
            for rxn_id in rxn_ids:
                rxn_set.add(self.model.reactions.get_one(id=rxn_id))
            return rxn_set

        def check_neg_species_pop_constraints(test_case, constraints, expected_constrs):
            test_case.assertEqual(len(constraints), len(expected_constrs))
            for id, constraint in constraints.items():
                set_of_terms = set()
                for linear_term in constraint.terms:
                    set_of_terms.add((linear_term.variable.name, linear_term.coefficient))
                test_case.assertEqual(set_of_terms, expected_constrs[id])

        # test with species that don't need constraints; dfba_obj only contains 'biomass_reaction'
        dfba_submodel_2 = self.make_dfba_submodel(self.model,
                                                  submodel_options=self.dfba_submodel_options,
                                                  dfba_obj_with_regular_rxn=False)
        dfba_submodel_2.dfba_solver_options['dfba_coef_scale_factor'] = 1
        constraints = dfba_submodel_2.initialize_neg_species_pop_constraints()
        # for each species, set of expected (rxn, coef) pairs contributing to species' consumption
        expected_constrs = {get_const_name('m3[c]'): {('ex_m3', -1.0), ('biomass_reaction', 1.0)}}
        check_neg_species_pop_constraints(self, constraints, expected_constrs)
        self.assertEqual(get_rxn_set(['ex_m3']), dfba_submodel_2._constrained_exchange_rxns)

        # TODO: test with a dFBA obj that contains multiple species
        dfba_submodel_2 = self.make_dfba_submodel(self.model,
                                                  submodel_options=self.dfba_submodel_options,
                                                  dfba_obj_with_regular_rxn=True)
        dfba_submodel_2.dfba_solver_options['dfba_coef_scale_factor'] = 1
        # no constraint on 'm2[c]' is made because 'm1[c]' isn't being consumed
        constraints = dfba_submodel_2.initialize_neg_species_pop_constraints()
        check_neg_species_pop_constraints(self, constraints, expected_constrs)
        self.assertEqual(get_rxn_set(['ex_m3']), dfba_submodel_2._constrained_exchange_rxns)

        # test with dfba_coef_scale_factor != 1
        dfba_submodel_2.dfba_solver_options['dfba_coef_scale_factor'] = 10
        constraints = dfba_submodel_2.initialize_neg_species_pop_constraints()
        expected_constrs = {get_const_name('m3[c]'): {('ex_m3', -1.0), ('biomass_reaction', 10.0)}}
        check_neg_species_pop_constraints(self, constraints, expected_constrs)
        self.assertEqual(get_rxn_set(['ex_m3']), dfba_submodel_2._constrained_exchange_rxns)

        # test with a species that is not consumed in any reaction that might be used in a constraint
        ex_m3 = self.model.reactions.get_one(id='ex_m3')
        # TODO: APG: fix
        for part in ex_m3.participants:
            print('part.coefficient', part.coefficient)
            # part.coefficient = 0
        dfba_submodel_2 = self.make_dfba_submodel(self.model,
                                                  submodel_options=self.dfba_submodel_options)
        dfba_submodel_2.dfba_solver_options['dfba_coef_scale_factor'] = 1
        constraints = dfba_submodel_2.initialize_neg_species_pop_constraints()
        print('constraints', constraints)
        # check_neg_species_pop_constraints(self, constraints, expected_constrs)

    def test_bound_neg_species_pop_constraints(self):
        scale_factor = self.dfba_submodel_options['dfba_bound_scale_factor']
        dfba_submodel_2 = self.make_dfba_submodel(self.model,
                                                  submodel_options=self.dfba_submodel_options,
                                                  dfba_obj_with_regular_rxn=True)
        dfba_submodel_2.bound_neg_species_pop_constraints()
        expected_constraint_bounds = {'neg_pop_constr__m1[c]': (0, 100),
                                      'neg_pop_constr__m3[c]': (0, 100)}
        conf_opt_model = dfba_submodel_2.get_conv_model()
        for constraint in conf_opt_model.constraints:
            if constraint.name in expected_constraint_bounds:
                ex_lower_bound, ex_upper_bound = expected_constraint_bounds[constraint.name]
                self.assertEqual(constraint.lower_bound, ex_lower_bound)
                self.assertEqual(constraint.upper_bound, ex_upper_bound)

    def bounds_test(test_case, dfba_submodel, expected_bounds):
        """ Test whether the reaction bounds specified by `expected_bounds` are set in `dfba_submodel`
        """
        for rxn_id, bounds in dfba_submodel._reaction_bounds.items():
            for ind, bound in enumerate(bounds):
                test_case.assertAlmostEqual(bound, expected_bounds[rxn_id][ind], delta=1e-09)

    def test_determine_bounds(self):
        scale_factor = self.dfba_submodel_options['dfba_bound_scale_factor']

        self.dfba_submodel_1.determine_bounds()
        self.bounds_test(self.dfba_submodel_1, self.expected_rxn_flux_bounds)

        # Test changing flux_bounds_volumetric_compartment_id
        new_options = copy.deepcopy(self.dfba_submodel_options)
        new_options['flux_bounds_volumetric_compartment_id'] = 'c'
        new_submodel = self.make_dfba_submodel(self.model, submodel_options=new_options)
        self.expected_rxn_flux_bounds['ex_m1'] = (100. * 5e-13 * scale_factor, 120. * 5e-13 * scale_factor)
        self.expected_rxn_flux_bounds['ex_m2'] = (100. * 5e-13 * scale_factor, 120. * 5e-13 * scale_factor)
        new_submodel.determine_bounds()
        self.bounds_test(new_submodel, self.expected_rxn_flux_bounds)

        del self.model.reactions.get_one(id='r1').rate_laws[0]
        del self.model.reactions.get_one(id='r2').rate_laws[1]
        self.model.reactions.get_one(id='ex_m1').flux_bounds = None
        self.model.reactions.get_one(id='ex_m1').reversible = True
        self.model.reactions.get_one(id='ex_m2').flux_bounds = None
        self.model.reactions.get_one(id='ex_m3').flux_bounds.max = numpy.nan
        self.model.reactions.get_one(id='ex_m3').flux_bounds.min = numpy.nan
        dfba_submodel_2 = self.make_dfba_submodel(self.model,
                                                  submodel_options=self.dfba_submodel_options)
        dfba_submodel_2.determine_bounds()
        expected_results_2 = {
            'ex_m1': (None, None),
            'ex_m2': (0., None),
            'ex_m3': (None, None),
            'r1': (-1. * 1 * scale_factor, None),
            'r2': (None, 1. * 1 * scale_factor + 2. * 1 * scale_factor),
            'r3': (0., 5. * 1 * scale_factor),
            'r4': (0., 6. * 1 * scale_factor),
        }
        self.bounds_test(dfba_submodel_2, expected_results_2)

    def test_determine_exchange_bounds(self):
        # test bounds on exchange reactions that avoid negative species populations
        scale_factor = self.dfba_submodel_options['dfba_bound_scale_factor']

        # test exchange rxn with reactant and no max bound
        # add a metabolite
        met_st_new = self.model.species_types.create(id='m_new',
                                                     type=wc_ontology['WC:metabolite'],
                                                     structure=wc_lang.ChemicalStructure(molecular_weight=1.))
        comp_c = self.model.compartments.get_one(id='c')
        met_species_new = self.model.species.create(species_type=met_st_new, compartment=comp_c)
        met_species_new.id = met_species_new.gen_id()
        conc_model = self.model.distribution_init_concentrations.create(species=met_species_new,
                                                                        mean=100.,
                                                                        std=0.,
                                                                        units=unit_registry.parse_units('molecule'))
        conc_model.id = conc_model.gen_id()

        # add an exchange reaction that consumes new species
        ex4 = self.submodel.reactions.create(id='ex_m4', reversible=False, model=self.model)
        ex4.flux_bounds = wc_lang.FluxBounds(min=0., max=None)
        ex4.participants.append(met_species_new.species_coefficients.get_or_create(coefficient=-2))

        # test exchange rxn with reactant used by another reaction and large max bound
        ex5 = self.submodel.reactions.create(id='ex_m5', reversible=False, model=self.model)
        ex5.flux_bounds = wc_lang.FluxBounds(min=0., max=None)
        m3 = self.model.species.get_one(id='m3[c]')
        ex5.participants.append(m3.species_coefficients.get_or_create(coefficient=-1))
        self.expected_rxn_flux_bounds['ex_m5'] = (0, None)

        dfba_submodel_3 = self.make_dfba_submodel(self.model,
                                                  submodel_options=self.dfba_submodel_options)
        rxn_id, species_id, coefficient = ('ex_m4', 'm_new[c]', -2)
        species_pop = dfba_submodel_3.local_species_population.read_one(0, species_id)
        expected_max_constr = -scale_factor * species_pop / (coefficient * dfba_submodel_3.time_step)
        self.expected_rxn_flux_bounds[rxn_id] = (0, expected_max_constr)

        dfba_submodel_3.determine_bounds()
        self.bounds_test(dfba_submodel_3, self.expected_rxn_flux_bounds)

    def test_update_bounds(self):
        new_bounds = {
            'ex_m1': (12., 20.),
            'ex_m2': (27.4, 33.8),
            'ex_m3': (0., 0.),
            'r1': (-2.5, 1.8),
            'r2': (-3, None),
            'r3': (0., 30.),
            'r4': (None, None),
            'biomass_reaction': (0., None),
        }
        self.dfba_submodel_1._reaction_bounds = new_bounds
        self.dfba_submodel_1.update_bounds()
        for var_id, variable in self.dfba_submodel_1._conv_variables.items():
            self.assertEqual(variable.lower_bound, new_bounds[var_id][0])
            self.assertEqual(variable.upper_bound, new_bounds[var_id][1])

    def do_test_compute_population_change_rates_control_caching(self, caching_settings):
        ### test with caching specified by caching_settings ###
        config_env_dict = ConfigEnvDict()
        for caching_attr, caching_setting in caching_settings:
            config_var_path = ['wc_sim', 'multialgorithm']
            config_var_path.append(caching_attr)
            config_env_dict.add_config_value(config_var_path, caching_setting)
        with EnvironUtils.make_temp_environ(**config_env_dict.get_env_dict()):

            new_fluxes = {
                'ex_m1': 13.2,
                'ex_m2': 30.,
                'ex_m3': 0.,
                'r1': -1.3,
                'r2': 0.5,
                'r3': 5.6,
                'r4': 2.5,
                'biomass_reaction': 6.8,
            }
            self.dfba_submodel_1.reaction_fluxes = new_fluxes
            self.dfba_submodel_1.compute_population_change_rates()

            expected_rates = {
                'm1[c]': -1 * 13.2,
                'm2[c]': -1 * 30.,
                'm3[c]': -1 * 0. +1 * 6.8,
            }
            self.assertEqual(self.dfba_submodel_1.adjustments, expected_rates)

    def test_compute_population_change_rates_control_caching(self):
        ### test all 3 caching combinations ###
        # NO CACHING
        # EVENT_BASED invalidation
        # REACTION_DEPENDENCY_BASED invalidation
        for caching_settings in ([('expression_caching', 'False')],
                                 [('expression_caching', 'True'),
                                  ('cache_invalidation', 'event_based')],
                                 [('expression_caching', 'True'),
                                  ('cache_invalidation', 'reaction_dependency_based')]):
            self.do_test_compute_population_change_rates_control_caching(caching_settings)

    def test_current_species_populations(self):
        self.dfba_submodel_1.current_species_populations()
        for idx, species_id in enumerate(self.dfba_submodel_1.species_ids):
            self.assertEqual(self.dfba_submodel_1.populations[idx],
                self.model.distribution_init_concentrations.get_one(
                    species=self.model.species.get_one(id=species_id)).mean)

    def test_scale_conv_opt_model(self):
        dfba_submodel_1 = self.dfba_submodel_1
        dfba_submodel_1.determine_bounds()
        dfba_submodel_1.update_bounds()
        scaled_conv_opt_model = dfba_submodel_1.scale_conv_opt_model(dfba_submodel_1.get_conv_model())
        # a new conv opt model has been made
        self.assertNotEqual(id(scaled_conv_opt_model), id(dfba_submodel_1.get_conv_model()))

        # supply scaling factors
        bound_scale_factor = dfba_submodel_1.dfba_solver_options['dfba_bound_scale_factor']
        coef_scale_factor = dfba_submodel_1.dfba_solver_options['dfba_coef_scale_factor']
        scaled_conv_opt_model_2 = dfba_submodel_1.scale_conv_opt_model(dfba_submodel_1.get_conv_model(),
                                                                       bound_scale_factor=bound_scale_factor,
                                                                       coef_scale_factor=coef_scale_factor)

        ## invert scale factors and test round-trip ##
        dfba_submodel_1.dfba_solver_options['dfba_bound_scale_factor'] = 1.0/bound_scale_factor
        dfba_submodel_1.dfba_solver_options['dfba_coef_scale_factor'] = 1.0/coef_scale_factor
        unscaled_conv_opt_model = dfba_submodel_1.scale_conv_opt_model(scaled_conv_opt_model)
        old_conv_model = dfba_submodel_1.get_conv_model()

        # check variables
        for old_var, new_var in zip(old_conv_model.variables, unscaled_conv_opt_model.variables):
            self.assertAlmostEqual(old_var.lower_bound, new_var.lower_bound)
            self.assertAlmostEqual(old_var.upper_bound, new_var.upper_bound)

        # check constraints
        for old_const, new_const in zip(old_conv_model.constraints, unscaled_conv_opt_model.constraints):
            self.assertAlmostEqual(old_const.lower_bound, new_const.lower_bound)
            self.assertAlmostEqual(old_const.upper_bound, new_const.upper_bound)

        # check coefficient terms in conv_opt model objective
        for old_term, new_term in zip(old_conv_model.objective_terms, unscaled_conv_opt_model.objective_terms):
            self.assertAlmostEqual(old_term.coefficient, new_term.coefficient)

        # alter the conv opt model in place
        modified_conv_opt_model = dfba_submodel_1.scale_conv_opt_model(dfba_submodel_1.get_conv_model(),
                                                                       copy_model=False)
        self.assertEqual(id(modified_conv_opt_model), id(dfba_submodel_1.get_conv_model()))

    def test_run_fba_solver(self):

        n = 1
        print(f'model #{n} in test_run_fba_solver')
        n += 1
        self.model.reactions.get_one(id='ex_m1').flux_bounds.max *= 1e11
        self.model.reactions.get_one(id='ex_m2').flux_bounds.max *= 1e11
        dfba_submodel_2 = self.make_dfba_submodel(self.model,
                                                  submodel_options=self.dfba_submodel_options)
        dfba_submodel_2.time_step = 1.

        dfba_submodel_2.run_fba_solver()

        bound_scale_factor = dfba_submodel_2.dfba_solver_options['dfba_bound_scale_factor']
        coef_scale_factor = dfba_submodel_2.dfba_solver_options['dfba_coef_scale_factor']
        '''
        self.assertEqual(dfba_submodel_2._optimal_obj_func_value,
                         (120. / bound_scale_factor) * coef_scale_factor)
        '''
        biomass_reaction = 120
        expected_adjustments = {
            'm1[c]': -120. * self.cell_volume * dfba_submodel_2.time_step * 1e11,
            'm2[c]': -120. * self.cell_volume * dfba_submodel_2.time_step * 1e11,
            'm3[c]': 120. * self.cell_volume * dfba_submodel_2.time_step * 1e11
            }
        for species_id, expected_adjustment in expected_adjustments.items():
            self.assertAlmostEqual(dfba_submodel_2.adjustments[species_id], expected_adjustment,
                                   delta=1e-09)

        species = ['m1[c]', 'm2[c]', 'm3[c]']
        expected_population = dict(zip(species, [
            100 - 120. * self.cell_volume * dfba_submodel_2.time_step * 1e11,
            100 - 120. * self.cell_volume * dfba_submodel_2.time_step * 1e11,
            100 + 120 * self.cell_volume * dfba_submodel_2.time_step * 1e11]))
        population = dfba_submodel_2.local_species_population.read(1., set(species))
        self.assertEqual(population, expected_population)

        # TODO: OPTIMIZE DFBA CACHING: test that expressions that depend on exchange and biomass reactions
        # have been flushed, and that expressions which don't depend on them have not been flushed
        # Test flush expression
        # TODO: fix
        # self.assertTrue(dfba_submodel_2.dynamic_model.cache_manager.empty())

        # Test using a different solver
        print(f'model #{n} in test_run_fba_solver')
        n += 1
        self.dfba_submodel_options['solver'] = 'glpk'
        del self.dfba_submodel_options['solver_options']
        dfba_submodel_2 = self.make_dfba_submodel(self.model,
            submodel_options=self.dfba_submodel_options)
        dfba_submodel_2.time_step = 1.
        dfba_submodel_2.run_fba_solver()

        self.assertEqual(dfba_submodel_2._optimal_obj_func_value, 120. * self.cell_volume * 1e11)
        for species_id, expected_adjustment in expected_adjustments.items():
            self.assertAlmostEqual(dfba_submodel_2.adjustments[species_id], expected_adjustment,
                                   delta=1e-09)

        species = ['m1[c]', 'm2[c]', 'm3[c]']
        expected_population = dict(zip(species, [
            100 - 120. * self.cell_volume * dfba_submodel_2.time_step * 1e11,
            100 - 120. * self.cell_volume * dfba_submodel_2.time_step * 1e11,
            100 + 120 * self.cell_volume * dfba_submodel_2.time_step * 1e11]))
        population = dfba_submodel_2.local_species_population.read(1., set(species))
        self.assertEqual(population, expected_population)

        # Test raise DynamicMultialgorithmError
        print(f'model #{n} in test_run_fba_solver')
        n += 1
        self.model.reactions.get_one(id='ex_m1').flux_bounds.min = -1000.
        self.model.reactions.get_one(id='ex_m1').flux_bounds.max = -1000.
        dfba_submodel_3 = self.make_dfba_submodel(self.model,
            submodel_options=self.dfba_submodel_options)
        dfba_submodel_3.time = 0.1
        dfba_submodel_3.time_step = 1.
        with self.assertRaisesRegexp(DynamicMultialgorithmError,
                                    re.escape("0.1: "
                                    "DfbaSubmodel metabolism: No optimal solution found: "
                                    "'infeasible' for time step [0.1, 1.1]")):
            dfba_submodel_3.run_fba_solver()

    def test_schedule_next_fba_event(self):
        # check that the next event is a RunFba message at time expected_time
        def check_next_event(expected_time):
            next_event = custom_fba_submodel.simulator.event_queue.next_events()[0]
            self.assertEqual(next_event.creation_time, 0)
            self.assertEqual(next_event.event_time, expected_time)
            self.assertEqual(next_event.sending_object, custom_fba_submodel)
            self.assertEqual(next_event.receiving_object, custom_fba_submodel)
            self.assertEqual(type(next_event.message), RunFba)
            self.assertTrue(custom_fba_submodel.simulator.event_queue.empty())

        custom_fba_time_step = 4
        custom_fba_submodel = self.make_dfba_submodel(self.model, dfba_time_step=custom_fba_time_step)
        # 1 initial event is scheduled
        self.assertEqual(custom_fba_submodel.simulator.event_queue.len(), 1)

        # initial event should be at time 0
        check_next_event(0)

        # next RunFba event should be at custom_fba_time_step
        custom_fba_submodel.schedule_next_periodic_analysis()
        check_next_event(custom_fba_time_step)