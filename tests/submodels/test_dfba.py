"""
:Author: Yin Hoon Chew <yinhoon.chew@mssm.edu>
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2020-08-05
:Copyright: 2016-2020, Karr Lab
:License: MIT
"""

import copy
import numpy
import os
import re
import scipy.constants
import unittest
import wc_lang

from de_sim.simulation_config import SimulationConfig
from wc_lang.core import ReactionParticipantAttribute
from wc_lang.io import Reader
from wc_onto import onto as wc_ontology
from wc_utils.util.units import unit_registry
from wc_sim.dynamic_components import DynamicRateLaw
from wc_sim.message_types import RunFba
from wc_sim.multialgorithm_errors import DynamicMultialgorithmError, MultialgorithmError
from wc_sim.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.sim_config import WCSimulationConfig
from wc_sim.submodels.dfba import DfbaSubmodel
from wc_sim.testing.make_models import MakeModel
from wc_sim.testing.utils import read_model_for_test, TempConfigFileModifier


class TestDfbaSubmodel(unittest.TestCase):

    def setUp(self):
        Av = scipy.constants.Avogadro
        model = self.model = wc_lang.Model()
        
        # Create compartment
        init_volume = wc_lang.core.InitVolume(distribution=wc_ontology['WC:normal_distribution'], 
                    mean=1e-12, std=0)
        c = model.compartments.create(id='c', init_volume=init_volume)
        c.init_density = model.parameters.create(id='density_c', value=1., 
                units=unit_registry.parse_units('g l^-1'))
        volume = model.functions.create(id='volume_c', units=unit_registry.parse_units('l'))
        volume.expression, error = wc_lang.FunctionExpression.deserialize(f'{c.id} / {c.init_density.id}', {
                wc_lang.Compartment: {c.id: c},
                wc_lang.Parameter: {c.init_density.id: c.init_density},
                })
        assert error is None, str(error)

        self.cell_volume = init_volume.mean

        # Create metabolites
        for m in [1,2,3]:
            metabolite_st = model.species_types.create(id='m{}'.format(m), type=wc_ontology['WC:metabolite'],
            	structure=wc_lang.ChemicalStructure(molecular_weight=1.))
            metabolite_species = model.species.create(species_type=metabolite_st, compartment=c)
            metabolite_species.id = metabolite_species.gen_id()
            conc_model = model.distribution_init_concentrations.create(species=metabolite_species, mean=10.,
            	units=unit_registry.parse_units('molecule'))
            conc_model.id = conc_model.gen_id()

        # Create enzymes
        for i, conc in {'enzyme1': 1, 'enzyme2': 1, 'enzyme3': 1}.items():
            enzyme_st = model.species_types.create(id=i, type=wc_ontology['WC:pseudo_species'],
            	structure=wc_lang.ChemicalStructure(molecular_weight=10.))
            enzyme_species = model.species.create(species_type=enzyme_st, compartment=c)
            enzyme_species.id = enzyme_species.gen_id()
            conc_model = model.distribution_init_concentrations.create(species=enzyme_species, mean=conc,
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
            'solver': 1,
            'presolve': 1,
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
     
        self.config_file_modifier = TempConfigFileModifier()

    def tearDown(self):
        self.config_file_modifier.clean_up()
        
    def make_dfba_submodel(self, model, dfba_time_step=1.0, submodel_name='metabolism', 
    	    submodel_options=None):
        """ Make a MultialgorithmSimulation from a wc lang model """
        # assume a single submodel
        self.dfba_time_step = dfba_time_step
        
        de_simulation_config = SimulationConfig(time_max=10)
        wc_sim_config = WCSimulationConfig(de_simulation_config, dfba_time_step=dfba_time_step)
        multialgorithm_simulation = MultialgorithmSimulation(model, wc_sim_config, 
        	options={'DfbaSubmodel': dict(options=submodel_options)})
        simulator, dynamic_model = multialgorithm_simulation.build_simulation()
        simulator.initialize()
        return dynamic_model.dynamic_submodels[submodel_name]

    ### test low level methods ###
    def test_dfba_submodel_init(self):
        self.assertEqual(self.dfba_submodel_1.time_step, self.dfba_time_step)
        # test options
        self.assertEqual(self.dfba_submodel_1.dfba_solver_options, self.dfba_submodel_options)    

        # test exceptions
        bad_dfba_time_step = -2
        with self.assertRaisesRegexp(MultialgorithmError,
            'DfbaSubmodel metabolism: time_step must be positive, but is {}'.format(bad_dfba_time_step)):
            self.make_dfba_submodel(self.model, dfba_time_step=bad_dfba_time_step)

        with self.assertRaisesRegexp(MultialgorithmError,
                                     "DfbaSubmodel metabolism: time_step must be a number but is"):
            self.make_dfba_submodel(self.model, dfba_time_step=None) 
        
        bad_solver = copy.deepcopy(self.dfba_submodel_options)
        bad_solver['solver'] = 20
        with self.assertRaisesRegexp(MultialgorithmError,
                                     "DfbaSubmodel metabolism: {} is not a valid Solver".format(
                                     	bad_solver['solver'])):
            self.make_dfba_submodel(self.model, submodel_options=bad_solver)

        bad_presolve = copy.deepcopy(self.dfba_submodel_options)
        bad_presolve['presolve'] = 20
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
        
        expected_results = {
            'm1[c]': {'ex_m1': 1, 'r1': -1, 'r2': -1, 'r3': -2},
            'm2[c]': {'ex_m2': 1, 'r1': -1, 'r2': 1, 'r4': -2},
            'm3[c]': {'ex_m3': 1, 'r1':1, 'r3': 1, 'r4': 1, 
                      'biomass_reaction': -1*self.dfba_submodel_options['dfba_coef_scale_factor']},
        }
        test_results = {k:{i.variable.name:i.coefficient for i in v} for k,v in 
            self.dfba_submodel_1._conv_metabolite_matrices.items()}
        self.assertEqual(test_results, expected_results)

        self.assertEqual(len(self.dfba_submodel_1._conv_model.variables), 
        	len(self.dfba_submodel_1._conv_variables.values()))
        for i in self.dfba_submodel_1._conv_variables.values():
        	self.assertEqual(i in self.dfba_submodel_1._conv_model.variables, True)

        self.assertEqual(len(self.dfba_submodel_1._conv_model.constraints), len(expected_results))
        for i in self.dfba_submodel_1._conv_model.constraints:
            self.assertEqual({j.variable.name:j.coefficient for j in i.terms}, expected_results[i.name])    	
            self.assertEqual(i.upper_bound, 0.)
            self.assertEqual(i.lower_bound, 0.)

        self.assertEqual(self.dfba_submodel_1._dfba_obj_rxn_ids, ['biomass_reaction'])
        self.assertEqual({i.variable.name:i.coefficient for i in self.dfba_submodel_1._conv_model.objective_terms},
        	{'biomass_reaction': 1})

        # Test model where the objective function is made of dfba objective reactions and network reactions
        obj_expression = f"biomass_reaction + 2 * r2"
        dfba_obj_expression, error = wc_lang.DfbaObjectiveExpression.deserialize(
            obj_expression, {
                wc_lang.DfbaObjReaction: {
                    'biomass_reaction': self.model.dfba_obj_reactions.get_one(id='biomass_reaction')
                    },
                wc_lang.Reaction:{    
                    'r2': self.model.reactions.get_one(id='r2')},
                })
        assert error is None, str(error)
        self.submodel.dfba_obj.expression = dfba_obj_expression
        dfba_submodel_2 = self.make_dfba_submodel(self.model, 
        	submodel_options=self.dfba_submodel_options)

        self.assertEqual(len(dfba_submodel_2._conv_variables), len(self.submodel.reactions) + 1)
        self.assertEqual('biomass_reaction' in dfba_submodel_2._conv_variables, True)
        for k,v in dfba_submodel_2._conv_variables.items():
            self.assertEqual(k, v.name)
        
        expected_results = {
            'm1[c]': {'ex_m1': 1, 'r1': -1, 'r2': -1, 'r3': -2},
            'm2[c]': {'ex_m2': 1, 'r1': -1, 'r2': 1, 'r4': -2},
            'm3[c]': {'ex_m3': 1, 'r1':1, 'r3': 1, 'r4': 1, 
                      'biomass_reaction': -1*self.dfba_submodel_options['dfba_coef_scale_factor']},
        }
        test_results = {k:{i.variable.name:i.coefficient for i in v} for k,v in 
            dfba_submodel_2._conv_metabolite_matrices.items()}
        self.assertEqual(test_results, expected_results)

        self.assertEqual(len(dfba_submodel_2._conv_model.variables), 
        	len(dfba_submodel_2._conv_variables.values()))
        for i in dfba_submodel_2._conv_variables.values():
        	self.assertEqual(i in dfba_submodel_2._conv_model.variables, True)

        self.assertEqual(len(dfba_submodel_2._conv_model.constraints), len(expected_results))
        for i in dfba_submodel_2._conv_model.constraints:
            self.assertEqual({j.variable.name:j.coefficient for j in i.terms}, expected_results[i.name])    	
            self.assertEqual(i.upper_bound, 0.)
            self.assertEqual(i.lower_bound, 0.)

        self.assertEqual(dfba_submodel_2._dfba_obj_rxn_ids, ['biomass_reaction'])
        self.assertEqual({i.variable.name:i.coefficient for i in dfba_submodel_2._conv_model.objective_terms},
        	{'biomass_reaction': 1, 'r2': 2})


    def test_determine_bounds(self):
        scale_factor = self.dfba_submodel_options['dfba_bound_scale_factor']
        
        self.dfba_submodel_1.determine_bounds()
        expected_results = {
            'ex_m1': (100.*self.cell_volume*scale_factor, 120.*self.cell_volume*scale_factor), 
            'ex_m2': (100.*self.cell_volume*scale_factor, 120.*self.cell_volume*scale_factor), 
            'ex_m3': (0., 0.), 
            'r1': (-1.*1*scale_factor, 1.*1*scale_factor), 
            'r2': (-(1.*1*scale_factor + 1.*1*scale_factor), 1.*1*scale_factor + 2.*1*scale_factor), 
            'r3': (0., 5.*1*scale_factor),
            'r4': (0., 6.*1*scale_factor),
        }        
        for k,v in self.dfba_submodel_1._reaction_bounds.items():
            for ind,val in enumerate(v):
                self.assertAlmostEqual(val, expected_results[k][ind], delta=1e-09)

        self.model.reactions.get_one(id='ex_m3').flux_bounds.max = numpy.nan
        self.model.reactions.get_one(id='ex_m3').flux_bounds.min = numpy.nan
        dfba_submodel_2 = self.make_dfba_submodel(self.model, 
        	submodel_options=self.dfba_submodel_options)
        dfba_submodel_2.determine_bounds()
        expected_results['ex_m3'] = (None, None)
        for k,v in dfba_submodel_2._reaction_bounds.items():
            for ind,val in enumerate(v):
                self.assertAlmostEqual(val, expected_results[k][ind], delta=1e-09)

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
            