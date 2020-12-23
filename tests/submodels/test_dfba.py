"""
:Author: Yin Hoon Chew <yinhoon.chew@mssm.edu>
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2020-08-05
:Copyright: 2016-2020, Karr Lab
:License: MIT
"""

import copy
import itertools
import math
import numpy
import os
import re
import scipy.constants
import unittest

from de_sim.simulation_config import SimulationConfig
from wc_lang.io import Reader
from wc_onto import onto as wc_ontology
from wc_sim.message_types import RunFba
from wc_sim.multialgorithm_errors import DynamicMultialgorithmError, MultialgorithmError
from wc_sim.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.sim_config import WCSimulationConfig
from wc_sim.submodels.dfba import DfbaSubmodel, ShowConvOptElements
from wc_utils.util.environ import EnvironUtils, ConfigEnvDict
from wc_utils.util.units import unit_registry
import conv_opt
import wc_lang


class TestDfbaSubmodel(unittest.TestCase):

    def get_model(self):
        test_model = os.path.join(os.path.dirname(__file__), 'fixtures', 'dfba_test_model.xlsx')
        # TODO (APG): test with model that uses extracellular compartment w id == EXTRACELLULAR_COMPARTMENT_ID in config
        return Reader().run(test_model, validate=False)[wc_lang.Model][0]

    def setUp(self):
        self.model = model = self.get_model()
        self.cell_volume = model.compartments.get_one(id='c').init_volume.mean + \
                           model.compartments.get_one(id='n').init_volume.mean
        self.submodel = model.submodels.get_one(id='metabolism')
        self.dfba_submodel_options = {
            'dfba_bound_scale_factor': 1e2,
            'dfba_coef_scale_factor': 10,
            'solver': 'cplex',
            'presolve': 'on',
            'optimization_type': 'maximize',
            'flux_bounds_volumetric_compartment_id': 'wc',
            'verbosity': 'off',
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
            },
            'negative_pop_constraints': True
        }

        self.dfba_submodel_1 = self.make_dfba_submodel(self.model,
                                                       submodel_options=self.dfba_submodel_options)

        # reactions in model were split into irreversible reactions
        # by SplitReversibleReactionsTransform in PrepForWcSimTransform
        #
        # map: rxn id: (lower bound, upper bound)
        self.expected_rxn_flux_bounds = {
            'ex_m1_backward': (0., -(-120. * self.cell_volume)),
            'ex_m1_forward': (0., -100. * self.cell_volume),
            'ex_m2_backward': (0., -(-120. * self.cell_volume)),
            'ex_m2_forward': (0., -100. * self.cell_volume),
            'ex_m3': (0., 0.),
            'r1_backward': (0., -(-1. * 1)),
            'r1_forward': (0., 1. * 1),
            'r2_backward': (0., (1. * 1 + 1. * 1)),
            'r2_forward': (0., 1. * 1 + 2. * 1),
            'r3': (0., 5. * 1),
            'r4': (0., 6. * 1)
        }

    def make_dfba_submodel(self, model, dfba_time_step=1.0, submodel_id='metabolism',
                           submodel_options=None, dfba_obj_with_regular_rxn=False, obj_expression=None):
        """ Make a MultialgorithmSimulation from a wc lang model, and return its dFBA submodel

        Args:
            model (:obj:`wc_lang.Model`): the model
            dfba_time_step (:obj:`float`, optional): the dFBA submodel's timestep
            submodel_id (:obj:`wc_lang.Model`, optional): the dFBA submodel's id
            submodel_options (:obj:`type`, optional): the dynamic dFBA submodel's options
            obj_expression (:obj:`str`, optional): a custom dfba objective expression

        Returns:
            :obj:`DfbaSubmodel`: a dynamic dFBA submodel
        """
        if dfba_obj_with_regular_rxn or obj_expression:
            # build a dfba objective expression
            if obj_expression is None:
                obj_expression = 'biomass_reaction + r2_forward'

            reactions_dict = {rxn.id: rxn for rxn in model.get_reactions()}
            dfba_obj_reactions_dict = {dfba_obj_rxn.id: dfba_obj_rxn for dfba_obj_rxn in model.get_dfba_obj_reactions()}
            dfba_obj_expression, error = wc_lang.DfbaObjectiveExpression.deserialize(
                obj_expression, {wc_lang.DfbaObjReaction: dfba_obj_reactions_dict,
                                 wc_lang.Reaction: reactions_dict})
            assert error is None, str(error)
            submodel = model.submodels.get_one(id='metabolism')
            submodel.dfba_obj.expression = dfba_obj_expression

        de_simulation_config = SimulationConfig(max_time=10)
        wc_sim_config = WCSimulationConfig(de_simulation_config, dfba_time_step=dfba_time_step)
        multialgorithm_simulation = MultialgorithmSimulation(model,
                                                             wc_sim_config,
                                                             options={'DfbaSubmodel':
                                                                      dict(options=submodel_options)})
        simulator, dynamic_model = multialgorithm_simulation.build_simulation()
        simulator.initialize()

        return dynamic_model.dynamic_submodels[submodel_id]

    def test_dfba_submodel_init(self):
        # test options
        self.assertEqual(self.dfba_submodel_1.dfba_solver_options, self.dfba_submodel_options)

        # test _dfba_obj_reactions
        expected = {'biomass_reaction': self.model.dfba_obj_reactions.get_one(id='biomass_reaction')}
        self.assertEqual(self.dfba_submodel_1._dfba_obj_reactions, expected)

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

        bad_verbosity_option = copy.deepcopy(self.dfba_submodel_options)
        bad_verbosity_option['verbosity'] = 'bad'
        with self.assertRaisesRegexp(MultialgorithmError, 'the verbosity in options must be one of'):
            self.make_dfba_submodel(self.model, submodel_options=bad_verbosity_option)

        # test ensure that all reactions are irreversible
        model = self.get_model()
        de_simulation_config = SimulationConfig(max_time=10)
        wc_sim_config = WCSimulationConfig(de_simulation_config, dfba_time_step=1.)
        multialgorithm_simulation = MultialgorithmSimulation(model,
                                                             wc_sim_config,
                                                             options=self.dfba_submodel_options)
        with self.assertRaisesRegexp(MultialgorithmError, 'DfbaSubmodel .*: reactions are reversible'):
            multialgorithm_simulation.build_simulation(prepare_model=False)

        # test exchange_rxns
        dfba_submodel = self.make_dfba_submodel(self.model)
        expected_exchange_rxns = set([self.model.reactions.get_one(id=rxn_id)
                                      for rxn_id in ('ex_m3', 'ex_m1_forward', 'ex_m1_backward',
                                      'ex_m2_forward', 'ex_m2_backward')])
        self.assertEqual(dfba_submodel.exchange_rxns, expected_exchange_rxns)

        # test dfba_obj_expr
        model = self.get_model()
        dfba_submodel = self.make_dfba_submodel(model)
        self.assertEqual(dfba_submodel.dfba_obj_expr.expression, 'biomass_reaction')

        # test exception in dfba_obj.expression that contains an exchange reaction
        with self.assertRaisesRegexp(MultialgorithmError,
            f"the dfba objective 'dfba-obj-{dfba_submodel.id}' uses exchange reactions:"):
            self.make_dfba_submodel(model, obj_expression='biomass_reaction + ex_m1_forward + ex_m3')

    def test_set_up_optimizations(self):
        dfba_submodel = self.dfba_submodel_1
        self.assertTrue(set(dfba_submodel.species_ids) == dfba_submodel.species_ids_set \
            == set(dfba_submodel.adjustments.keys()))
        self.assertEqual(dfba_submodel.populations.shape, ((len(dfba_submodel.species_ids), )))
        self.assertEqual(dfba_submodel.reaction_fluxes, {'ex_m1_backward': None,
                                                         'ex_m1_forward': None,
                                                         'ex_m2_backward': None,
                                                         'ex_m2_forward': None,
                                                         'ex_m3': None,
                                                         'r1_backward': None,
                                                         'r1_forward': None,
                                                         'r2_backward': None,
                                                         'r2_forward': None,
                                                         'r3': None,
                                                         'r4': None})

    def test_set_up_dfba_submodel(self):
        # test exception when the ids in DfbaObjReactions and Reactions intersect
        model = self.get_model()
        # add 'biomass_reaction' to Reactions to create intersection
        dfba_submodel = model.submodels.get_one(id='metabolism')
        ex1 = dfba_submodel.reactions.create(id='biomass_reaction', reversible=False, model=model)
        with self.assertRaisesRegexp(MultialgorithmError,
            "in model .* the ids in DfbaObjReactions and Reactions intersect:"):
            self.make_dfba_submodel(model)

        self.dfba_submodel_1.set_up_dfba_submodel()
        self.assertEqual(len(self.dfba_submodel_1._conv_variables), len(self.submodel.reactions) + 1)
        self.assertEqual('biomass_reaction' in self.dfba_submodel_1._conv_variables, True)
        for k,v in self.dfba_submodel_1._conv_variables.items():
            self.assertEqual(k, v.name)

        # test the stoichiometric matrix
        expected_stoch_coefficients = {
            'm1[c]': {'ex_m1_backward': 1,
                      'ex_m1_forward': -1,
                      'r1_backward': 1,
                      'r1_forward': -1,
                      'r2_backward': 1,
                      'r2_forward': -1,
                      'r3': -2},
            'm2[c]': {'ex_m2_backward': 1,
                      'ex_m2_forward': -1,
                      'r1_backward': 1,
                      'r1_forward': -1,
                      'r2_backward': -1,
                      'r2_forward': 1,
                      'r4': -2},
            'm3[c]': {'biomass_reaction': -1,
                      'ex_m3': -1,
                      'r1_backward': -1,
                      'r1_forward': 1,
                      'r3': 1,
                      'r4': 1}}
        test_results = {species_id: {lt.variable.name: lt.coefficient for lt in linear_terms}
                        for species_id, linear_terms
                            in self.dfba_submodel_1._conv_metabolite_matrices.items()}
        self.assertEqual(test_results, expected_stoch_coefficients)

        self.assertEqual(len(self.dfba_submodel_1.get_conv_model().variables),
            len(self.dfba_submodel_1._conv_variables.values()))
        for conv_opt_var in self.dfba_submodel_1._conv_variables.values():
            self.assertEqual(conv_opt_var in self.dfba_submodel_1.get_conv_model().variables, True)

        for constr in self.dfba_submodel_1.get_conv_model().constraints:
            if constr.name in expected_stoch_coefficients:
                self.assertEqual({linear_term.variable.name:linear_term.coefficient
                                  for linear_term in constr.terms},
                                 expected_stoch_coefficients[constr.name])
                self.assertEqual(constr.upper_bound, 0.)
                self.assertEqual(constr.lower_bound, 0.)

        self.assertEqual({i.variable.name:i.coefficient
                          for i in self.dfba_submodel_1.get_conv_model().objective_terms},
                         {'biomass_reaction': 1})
        self.assertEqual(self.dfba_submodel_1.get_conv_model().objective_direction,
                         conv_opt.ObjectiveDirection.maximize)

        # test model where the objective function is made of dfba objective reactions and
        # network reactions and is minimized
        self.dfba_submodel_options['optimization_type'] = 'minimize'
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
        self.assertEqual(test_results, expected_stoch_coefficients)

        self.assertEqual(len(dfba_submodel_2.get_conv_model().variables),
                         len(dfba_submodel_2._conv_variables.values()))

        for conv_opt_var in dfba_submodel_2._conv_variables.values():
            self.assertEqual(conv_opt_var in dfba_submodel_2.get_conv_model().variables, True)

        for constr in dfba_submodel_2.get_conv_model().constraints:
            if constr.name in expected_stoch_coefficients:
                self.assertEqual({linear_term.variable.name:linear_term.coefficient
                                  for linear_term in constr.terms},
                                 expected_stoch_coefficients[constr.name])
                self.assertEqual(constr.upper_bound, 0.)
                self.assertEqual(constr.lower_bound, 0.)

        self.assertEqual({objective_term.variable.name: objective_term.coefficient
                          for objective_term in dfba_submodel_2.get_conv_model().objective_terms},
                         {'biomass_reaction': 1, 'r2_forward': 1})
        self.assertEqual(dfba_submodel_2.get_conv_model().objective_direction,
                         conv_opt.ObjectiveDirection.minimize)

        with self.assertRaisesRegexp(MultialgorithmError, 'objective function not linear: reaction.* have NaN coeff'):
            self.make_dfba_submodel(self.model, obj_expression='biomass_reaction * r2_forward')

    def test_get_species_and_stoichiometry(self):
        def get_species(id):
            return self.model.species.get_one(id=id)

        # test with a dFBA obj that contains a regular reaction
        dfba_submodel_2 = self.make_dfba_submodel(self.model,
                                                  submodel_options=self.dfba_submodel_options,
                                                  dfba_obj_with_regular_rxn=True)
        expected = dict(r2_forward={get_species('m1[c]'): -1.,
                                    get_species('m2[c]'): 1.},
                        biomass_reaction={get_species('m3[c]'): -1.})
        for rxn_class in dfba_submodel_2.dfba_obj_expr.related_objects:
            for rxn_id, rxn in dfba_submodel_2.dfba_obj_expr.related_objects[rxn_class].items():
                self.assertEqual(DfbaSubmodel._get_species_and_stoichiometry(rxn),
                                 expected[rxn_id])

        # also test reactions with the same species on both sides
        r2_forward = self.model.reactions.get_one(id='r2_forward')
        r2_forward.participants.append(get_species(id='m1[c]').species_coefficients.get_or_create(coefficient=1))
        dfba_submodel_2 = self.make_dfba_submodel(self.model,
                                                  submodel_options=self.dfba_submodel_options,
                                                  dfba_obj_with_regular_rxn=True)
        expected = dict(r2_forward={get_species('m2[c]'): 1.},
                        biomass_reaction={get_species('m3[c]'): -1.})
        for rxn_class in dfba_submodel_2.dfba_obj_expr.related_objects:
            for rxn_id, rxn in dfba_submodel_2.dfba_obj_expr.related_objects[rxn_class].items():
                self.assertEqual(DfbaSubmodel._get_species_and_stoichiometry(rxn),
                                 expected[rxn_id])

    def test_species_id_conversions(self):
        species_id = 's[c]'
        round_trip_id = DfbaSubmodel.species_id_with_brkts(DfbaSubmodel.species_id_without_brkts(species_id))
        self.assertEqual(species_id, round_trip_id)

        # test species_id_without_brkts exceptions
        invalid_species_id = 'x[c'
        with self.assertRaisesRegexp(MultialgorithmError, 'invalid species_id'):
            DfbaSubmodel.species_id_without_brkts(invalid_species_id)
        invalid_species_id = f'x{DfbaSubmodel.LB}[c]'
        with self.assertRaisesRegexp(MultialgorithmError, "species_id .* already has bracket code"):
            DfbaSubmodel.species_id_without_brkts(invalid_species_id)

        # test species_id_with_brkts exceptions
        invalid_species_id = 'x[c]'
        with self.assertRaisesRegexp(MultialgorithmError,
                                     "invalid species_id with bracket codes .* it should be"):
            DfbaSubmodel.species_id_with_brkts(invalid_species_id)
        invalid_species_id = 'xyz'
        with self.assertRaisesRegexp(MultialgorithmError,
                                     "invalid species_id with bracket codes .* it should be"):
            DfbaSubmodel.species_id_with_brkts(invalid_species_id)

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

        # generate constraint involving 2 pseudo-reactions that use m3[c], ex_m3 and biomass_reaction
        dfba_submodel_2 = self.make_dfba_submodel(self.get_model(),
                                                  submodel_options=self.dfba_submodel_options)
        constraints = dfba_submodel_2.initialize_neg_species_pop_constraints()
        print(ShowConvOptElements.show_conv_opt_constraints(constraints))
        # for each species, set of expected (rxn, coef) pairs contributing to the species' consumption
        raw_exp_constrs = dict(m1={('ex_m1_backward', 1), ('ex_m1_forward', -1)},
                               m2={('ex_m2_backward', 1), ('ex_m2_forward', -1)},
                               m3={('ex_m3', -1), ('biomass_reaction', -1)})

        expected_constrs = {get_const_name(f'{s_id}[c]'): constrs for s_id, constrs in raw_exp_constrs.items()}
        check_neg_species_pop_constraints(self, constraints, expected_constrs)

        # add another dfba obj. species to biomass_reaction to cover the remaining branches
        model = self.get_model()
        biomass_reaction = model.dfba_obj_reactions.get_one(id='biomass_reaction')
        # m1[n] won't need a constraint because it is not consumed
        dfba_obj_species = wc_lang.DfbaObjSpecies(value=2, species=model.species.get_one(id='m1[n]'))
        biomass_reaction.dfba_obj_species.add(dfba_obj_species)
        dfba_submodel_2 = self.make_dfba_submodel(model,
                                                  submodel_options=self.dfba_submodel_options)
        constraints = dfba_submodel_2.initialize_neg_species_pop_constraints()
        check_neg_species_pop_constraints(self, constraints, expected_constrs)

    def test_bound_neg_species_pop_constraints(self):
        dfba_submodel_2 = self.make_dfba_submodel(self.model,
                                                  submodel_options=self.dfba_submodel_options,
                                                  dfba_obj_with_regular_rxn=True)
        dfba_submodel_2.bound_neg_species_pop_constraints()
        expected_constraint_bounds = {'neg_pop_constr__m1[c]': (-100, None),
                                      'neg_pop_constr__m3[c]': (-100, None)}
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
            if rxn_id in expected_bounds:
                for ind, bound in enumerate(bounds):
                    test_case.assertAlmostEqual(bound, expected_bounds[rxn_id][ind], delta=1e-09)

    def test_determine_bounds(self):
        self.dfba_submodel_1.determine_bounds()
        self.bounds_test(self.dfba_submodel_1, self.expected_rxn_flux_bounds)

        ### test changing flux_bounds_volumetric_compartment_id
        new_options = copy.deepcopy(self.dfba_submodel_options)
        new_options['flux_bounds_volumetric_compartment_id'] = 'c'
        new_submodel = self.make_dfba_submodel(self.model, submodel_options=new_options)
        self.expected_rxn_flux_bounds['ex_m1_backward'] = (0., -(-120. * 5e-13))
        self.expected_rxn_flux_bounds['ex_m1_forward'] = (0., -100. * 5e-13)
        self.expected_rxn_flux_bounds['ex_m2_backward'] = (0., -(-120. * 5e-13))
        self.expected_rxn_flux_bounds['ex_m2_forward'] = (0., -100. * 5e-13)
        new_submodel.determine_bounds()
        self.bounds_test(new_submodel, self.expected_rxn_flux_bounds)
        # TODO (APG): to cover other branches in determine_bounds(), test without using PrepForWcSimTransform

    def test_update_bounds(self):
        new_bounds = {
            'ex_m1_backward': (0., 0.),
            'ex_m1_forward': (12., 20.),
            'ex_m2_backward': (0., 0.),
            'ex_m2_forward': (27.4, 33.8),
            'ex_m3': (0., 0.),
            'r1_backward': (0., 2.5),
            'r1_forward': (0., 1.8),
            'r2_backward': (0., 3.),
            'r2_forward': (0., None),
            'r3': (0., 30.),
            'r4': (None, None),
            'biomass_reaction': (0., None)}
        self.dfba_submodel_1._reaction_bounds = new_bounds
        self.dfba_submodel_1.update_bounds()
        for var_id, variable in self.dfba_submodel_1._conv_variables.items():
            self.assertEqual(variable.lower_bound, new_bounds[var_id][0])
            self.assertEqual(variable.upper_bound, new_bounds[var_id][1])

    def do_test_compute_population_change_rates_control_caching(self, caching_settings):
        # test with caching specified by caching_settings
        config_env_dict = ConfigEnvDict()
        for caching_attr, caching_setting in caching_settings:
            config_var_path = ['wc_sim', 'multialgorithm']
            config_var_path.append(caching_attr)
            config_env_dict.add_config_value(config_var_path, caching_setting)
        with EnvironUtils.make_temp_environ(**config_env_dict.get_env_dict()):

            # use fluxes and expected population change rates from
            # test I in "Solutions to test dFBA models, by hand.txt"
            fluxes = dict(ex_m1_backward=12.,
                          ex_m1_forward=0.,
                          ex_m2_backward=12.,
                          ex_m2_forward=0.,
                          ex_m3=0.,
                          r1_backward=0.,
                          r1_forward=1.,
                          r2_backward=0.,
                          r2_forward=1.,
                          r3=5.,
                          r4=6.,
                          biomass_reaction=12.)

            self.dfba_submodel_1.reaction_fluxes = fluxes
            self.dfba_submodel_1.compute_population_change_rates()
            expected_rates = {'m1[c]': -12.,
                              'm2[c]': -12.,
                              'm3[c]': 12.}
            self.assertEqual(self.dfba_submodel_1.adjustments, expected_rates)
            # test repeated calculations
            self.dfba_submodel_1.compute_population_change_rates()
            self.assertEqual(self.dfba_submodel_1.adjustments, expected_rates)

    def test_compute_population_change_rates(self):
        ### test all 3 caching combinations:
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

    # TODO (APG): later: why didn't these 2 unit tests find problems with scaling/unscaling?
    def test_scale_conv_opt_model(self):
        dfba_submodel_1 = self.dfba_submodel_1
        dfba_submodel_1.determine_bounds()

        # rxn and constraint bounds have not been set in the conv opt model
        scaled_co_model = dfba_submodel_1.scale_conv_opt_model(dfba_submodel_1.get_conv_model())
        old_co_model = dfba_submodel_1.get_conv_model()
        # check that they have not changed
        for old_var, new_var in zip(old_co_model.variables, scaled_co_model.variables):
            self.assertEqual(old_var.lower_bound, new_var.lower_bound)
            self.assertEqual(old_var.upper_bound, new_var.upper_bound)
        for old_const, new_const in zip(old_co_model.constraints, scaled_co_model.constraints):
            self.assertEqual(old_const.lower_bound, new_const.lower_bound)
            self.assertEqual(old_const.upper_bound, new_const.upper_bound)

        dfba_submodel_1.update_bounds()
        scaled_co_model = dfba_submodel_1.scale_conv_opt_model(dfba_submodel_1.get_conv_model())
        # a new conv opt model has been made
        self.assertNotEqual(id(scaled_co_model), id(dfba_submodel_1.get_conv_model()))

        ## invert scale factors and test round-trip ##
        # supply scaling factors
        bound_scale_factor = dfba_submodel_1.dfba_solver_options['dfba_bound_scale_factor']
        coef_scale_factor = dfba_submodel_1.dfba_solver_options['dfba_coef_scale_factor']
        unscaled_co_model = dfba_submodel_1.scale_conv_opt_model(scaled_co_model,
                                                                 dfba_bound_scale_factor=1.0/bound_scale_factor,
                                                                 dfba_coef_scale_factor=1.0/coef_scale_factor)
        old_co_model = dfba_submodel_1.get_conv_model()
        # check variables
        for old_var, new_var in zip(old_co_model.variables, unscaled_co_model.variables):
            self.assertAlmostEqual(old_var.lower_bound, new_var.lower_bound)
            self.assertAlmostEqual(old_var.upper_bound, new_var.upper_bound)
        # check constraints
        for old_const, new_const in zip(old_co_model.constraints, unscaled_co_model.constraints):
            self.assertAlmostEqual(old_const.lower_bound, new_const.lower_bound)
            self.assertAlmostEqual(old_const.upper_bound, new_const.upper_bound)
        # check coefficient terms in conv_opt model objective
        for old_term, new_term in zip(old_co_model.objective_terms, unscaled_co_model.objective_terms):
            self.assertAlmostEqual(old_term.coefficient, new_term.coefficient)

        # alter the conv opt model in place
        modified_co_model = dfba_submodel_1.scale_conv_opt_model(dfba_submodel_1.get_conv_model(),
                                                                 copy_model=False)
        self.assertEqual(id(modified_co_model), id(dfba_submodel_1.get_conv_model()))

    def test_unscale_conv_opt_solution(self):
        dfba_submodel_1 = self.dfba_submodel_1
        dfba_submodel_1.determine_bounds()
        dfba_submodel_1.update_bounds()

        ## invert scale factors and test round-trip ##
        conv_opt_solution = dfba_submodel_1.get_conv_model().solve()
        dfba_submodel_1.save_fba_solution(dfba_submodel_1.get_conv_model(), conv_opt_solution)
        original_dfba_submodel_1__optimal_obj_func_value = dfba_submodel_1._optimal_obj_func_value
        original_dfba_submodel_1__reaction_fluxes = copy.deepcopy(dfba_submodel_1.reaction_fluxes)

        dfba_submodel_1.unscale_conv_opt_solution()
        bound_scale_factor = dfba_submodel_1.dfba_solver_options['dfba_bound_scale_factor']
        coef_scale_factor = dfba_submodel_1.dfba_solver_options['dfba_coef_scale_factor']
        # reverse the removal of scaling factors
        dfba_submodel_1.unscale_conv_opt_solution(dfba_bound_scale_factor=1.0/bound_scale_factor,
                                                  dfba_coef_scale_factor=1.0/coef_scale_factor)
        # check _optimal_obj_func_value
        self.assertAlmostEqual(original_dfba_submodel_1__optimal_obj_func_value,
                               dfba_submodel_1._optimal_obj_func_value)
        # check reaction_fluxes
        for rxn_variable in dfba_submodel_1.get_conv_model().variables:
            self.assertAlmostEqual(original_dfba_submodel_1__reaction_fluxes[rxn_variable.name],
                                   dfba_submodel_1.reaction_fluxes[rxn_variable.name])

    def check_expected_solution(test_case, dfba_submodel, obj_func_value, expected_adjustments):
        test_case.assertEqual(dfba_submodel._optimal_obj_func_value, obj_func_value)
        for species_id, expected_adjustment in expected_adjustments.items():
            test_case.assertAlmostEqual(dfba_submodel.adjustments[species_id], expected_adjustment,
                                   delta=1e-09)

    def test_run_fba_solver(self):
        # Algebraic solutions to these tests are documented in the
        # file "tests/submodels/fixtures/Solutions to test dFBA models, by hand.txt"
        # TODO (APG): update this file for split, irreversible reactions
        TEST_NAME = 'I: No scaling (scaling factors equal 1) and no negative species population checks'
        '''
        print('flux_bounds before')
        for r_id in ['ex_m1_backward', 'ex_m1_forward', 'ex_m2_backward', 'ex_m2_forward']:
            r = self.model.reactions.get_one(id=r_id)
            print(f"{r_id}: ({r.flux_bounds.min}, {r.flux_bounds.max})")
        '''
        self.model.reactions.get_one(id='ex_m1_backward').flux_bounds.max *= 1e11
        self.model.reactions.get_one(id='ex_m2_backward').flux_bounds.max *= 1e11

        self.dfba_submodel_options['dfba_bound_scale_factor'] = 1.
        self.dfba_submodel_options['dfba_coef_scale_factor'] = 1.
        self.dfba_submodel_options['negative_pop_constraints'] = False
        dfba_submodel_2 = self.make_dfba_submodel(self.model,
                                                  submodel_options=self.dfba_submodel_options)
        dfba_submodel_2.get_conv_model().name = TEST_NAME
        dfba_submodel_2.time_step = 1.
        dfba_submodel_2.run_fba_solver()

        self.assertEqual(dfba_submodel_2._optimal_obj_func_value, 12)
        expected_adjustments_I = {'m1[c]': -12,
                                  'm2[c]': -12,
                                  'm3[c]': 12}
        self.check_expected_solution(dfba_submodel_2, 12, expected_adjustments_I)

        TEST_NAME = 'II: Add negative species population constraints to I'
        self.dfba_submodel_options['negative_pop_constraints'] = True
        dfba_submodel_2 = self.make_dfba_submodel(self.model,
                                                  submodel_options=self.dfba_submodel_options)
        dfba_submodel_2.get_conv_model().name = TEST_NAME
        dfba_submodel_2.run_fba_solver()
        self.check_expected_solution(dfba_submodel_2, 12, expected_adjustments_I)

        TEST_NAME = 'III: Modify II by scaling bounds by 10'
        self.dfba_submodel_options['dfba_bound_scale_factor'] = 10.
        dfba_submodel_2 = self.make_dfba_submodel(self.model,
                                                  submodel_options=self.dfba_submodel_options)
        dfba_submodel_2.get_conv_model().name = TEST_NAME
        dfba_submodel_2.run_fba_solver()
        self.check_expected_solution(dfba_submodel_2, 12, expected_adjustments_I)
        self.dfba_submodel_options['dfba_bound_scale_factor'] = 1.

        TEST_NAME = 'IV: Alter II so that a negative species population constraints change the solution'
        dfba_submodel_2 = self.make_dfba_submodel(self.model,
                                                  submodel_options=self.dfba_submodel_options)
        dfba_submodel_2.get_conv_model().name = TEST_NAME
        dfba_submodel_2.time_step = 10
        dfba_submodel_2.run_fba_solver()
        expected_adjustments_IV = {'m1[c]': -8,
                                   'm2[c]': -12,
                                   'm3[c]': 10}
        self.check_expected_solution(dfba_submodel_2, 10, expected_adjustments_IV)
        dfba_submodel_2.time_step = 1

        TEST_NAME = "V: Modify II by scaling the coefficients of objective function's terms by 10"
        self.dfba_submodel_options['dfba_coef_scale_factor'] = 10.
        dfba_submodel_2 = self.make_dfba_submodel(self.model,
                                                  submodel_options=self.dfba_submodel_options)
        dfba_submodel_2.get_conv_model().name = TEST_NAME
        dfba_submodel_2.run_fba_solver()
        self.check_expected_solution(dfba_submodel_2, 12, expected_adjustments_I)
        self.dfba_submodel_options['dfba_coef_scale_factor'] = 1.

        TEST_NAME = 'VI: Modify V by scaling bounds by 5'
        self.dfba_submodel_options['dfba_bound_scale_factor'] = 5.
        self.dfba_submodel_options['dfba_coef_scale_factor'] = 10.
        dfba_submodel_2 = self.make_dfba_submodel(self.model,
                                                  submodel_options=self.dfba_submodel_options)
        dfba_submodel_2.get_conv_model().name = TEST_NAME
        dfba_submodel_2.run_fba_solver()
        self.check_expected_solution(dfba_submodel_2, 12, expected_adjustments_I)
        self.dfba_submodel_options['dfba_bound_scale_factor'] = 1.
        self.dfba_submodel_options['dfba_coef_scale_factor'] = 1.

        # test raise DynamicMultialgorithmError
        self.model.reactions.get_one(id='ex_m1_forward').flux_bounds.min = 1000.
        self.model.reactions.get_one(id='ex_m1_forward').flux_bounds.max = 1000.
        dfba_submodel_3 = self.make_dfba_submodel(self.model,
                                                  submodel_options=self.dfba_submodel_options)
        dfba_submodel_3.time = 0.1
        dfba_submodel_3.time_step = 1.
        with self.assertRaisesRegexp(DynamicMultialgorithmError,
                                    re.escape("0.1: "
                                    "DfbaSubmodel metabolism: No optimal solution found: "
                                    "'infeasible' for time step [0.1, 1.1]")):
            dfba_submodel_3.run_fba_solver()

        species = ['m1[c]', 'm2[c]', 'm3[c]']
        expected_population = dict(zip(species, [
            100 - 120. * self.cell_volume * dfba_submodel_2.time_step * 1E11,
            100 - 120. * self.cell_volume * dfba_submodel_2.time_step * 1E11,
            100 + 120. * self.cell_volume * dfba_submodel_2.time_step * 1E11]))
        population = dfba_submodel_2.local_species_population.read(1., set(species))
        self.assertEqual(population, expected_population)

        # TODO (APG): OPTIMIZE DFBA CACHING: test that expressions that depend on exchange and biomass reactions
        # have been flushed, and that expressions which don't depend on them have not been flushed
        # test flush expression
        # self.assertTrue(dfba_submodel_2.dynamic_model.cache_manager.empty())

        # TODO (APG): later: test using a different solver
        '''
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
        '''

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