""" Verification of FBA test cases from the SBML Test Suite

:Author: Yin Hoon Chew <yinhoon.chew@mssm.edu>
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2020-08-12
:Copyright: 2016-2020, Karr Lab
:License: MIT
"""

import libsbml
import numpy as np
import os
import pandas as pd
import shutil
import tempfile
import time
import wc_lang

from wc_onto import onto as wc_onto
from wc_sim.run_results import RunResults
from wc_sim.simulation import Simulation
from wc_sim.testing.verify import (VerificationError, VerificationTestCaseType, 
	                               VerificationTestReader, ResultsComparator, 
	                               VerificationResultType, VerificationSuite)
from wc_utils.util import chem
from wc_utils.util.units import unit_registry


class FbaVerificationTestReader(VerificationTestReader):
    """ Read an FBA model verification test case
    
    Read and access settings and expected results of an SBML test suite FBA test case
    
    Attributes:
        expected_predictions_df (:obj:`pandas.DataFrame`): the test case's expected predictions
        objective_direction (:obj:`str`): direction of optimization, i.e. 'maximize' or 'minimize'
    """

    def __init__(self, test_case_type_name, test_case_dir, test_case_num):
        """ Initialize an FBA model verification test case reader
        
        Args:
            test_case_type_name (:obj:`str`): a member of `VerificationTestCaseType`
            test_case_dir (:obj:`str`): pathname of the directory storing the test case
            test_case_num (:obj:`str`): the test case's unique ID number

        Raises:
            :obj:`VerificationError`: if test_case_type is not DYNAMIC_FLUX_BALANCE_ANALYSIS   
        """        
        super().__init__(test_case_type_name, test_case_dir, test_case_num)
        if self.test_case_type != VerificationTestCaseType.DYNAMIC_FLUX_BALANCE_ANALYSIS:
            raise VerificationError("VerificationTestCaseType is '{}' and not "
                                    "'DYNAMIC_FLUX_BALANCE_ANALYSIS'".format(test_case_type_name))

    def read_expected_predictions(self):
        """ Get the test case's expected predictions
        
        Returns:
            :obj:`pandas.DataFrame`: the test case's expected predictions

        Raises:
            :obj:`VerificationError`: if the variables in settings differ from the variables
            	in expected predictions
        """
        self.expected_predictions_file = expected_predictions_file = os.path.join(
            self.test_case_dir, self.test_case_num+'-results.csv')
        expected_predictions_df = pd.read_csv(expected_predictions_file)
        # expected predictions should contain data for all variables
        expected_columns = set(self.settings['variables'])
        actual_columns = set(expected_predictions_df.columns.values)
        if expected_columns - actual_columns:
            raise VerificationError("some variables missing from expected predictions '{}': {}".format(
                expected_predictions_file, expected_columns - actual_columns))
        
        return expected_predictions_df

    def species_columns(self):
        """ Get the names of the species columns """
        pass

    def slope_of_predictions(self):
        """ Determine the expected derivatives of species from the expected populations """
        pass

    def read_model(self, sbml_version='l3v2'):
        """  Read a model into a `wc_lang` representation

        Args:
            sbml_version (:obj:`str`, optional): SBML version, default is Level 3 Version 2 (l3v2)
        
        Returns:
            :obj:`wc_lang.Model`: the root of the test case's `wc_lang` model

        Raises:
            :obj:`VerificationError`: if model xml file does not exist, or if test case has more than
                one compartment    
        """
        self.model_filename = os.path.join(self.test_case_dir, 
        	self.test_case_num + '-sbml-' + sbml_version + self.SBML_FILE_SUFFIX)
        if not os.path.exists(self.model_filename):
        	raise VerificationError("Test case model file '{}' does not exists".format(self.model_filename))
        
        # Create `wc_lang` model and FBA submodel
        model = wc_lang.Model(id='test_case_' + self.test_case_num)
        dfba_submodel = wc_lang.Submodel(id=self.test_case_num + '-dfba', model=model, 
        	framework=wc_onto['WC:dynamic_flux_balance_analysis'])

        sbml_doc = libsbml.SBMLReader().readSBML(self.model_filename)        
        sbml_model = sbml_doc.getModel()
        fbc_sbml_model = sbml_model.getPlugin('fbc')

        compartment_list = sbml_model.getListOfCompartments()
        if len(compartment_list) > 1:
            raise VerificationError("Test case {} has more than one compartment: {}".format(
        							self.test_case_num, [i.getId() for i in compartment_list]))
        # Add compartment volume and density to fulfill wc_sim validation
        init_volume = wc_lang.InitVolume(distribution=wc_onto['WC:normal_distribution'], 
                    mean=1., std=0)
        model_compartment = model.compartments.create(id=compartment_list[0].getId(), 
        	init_volume=init_volume)
        model_compartment.init_density = model.parameters.create(id='density_' + model_compartment.id, 
        	value=1., units=unit_registry.parse_units('g l^-1'))
        volume = model.functions.create(id='volume_' + model_compartment.id, 
        	units=unit_registry.parse_units('l'))
        volume.expression, error = wc_lang.FunctionExpression.deserialize(
        	f'{model_compartment.id} / {model_compartment.init_density.id}', 
        	{
                wc_lang.Compartment: {model_compartment.id: model_compartment},
                wc_lang.Parameter: {model_compartment.init_density.id: model_compartment.init_density},
            })
        assert error is None, str(error)
        
        # Create model species
        for species in sbml_model.getListOfSpecies():
            fbc_species = species.getPlugin('fbc')
            charge = fbc_species.getCharge()
            formula = chem.EmpiricalFormula(fbc_species.getChemicalFormula())
            
            model_species_type = model.species_types.create(id=species.getId())
            model_species_type.structure = wc_lang.ChemicalStructure(
                charge=charge, empirical_formula=formula, molecular_weight=formula.get_molecular_weight())
            model_species = model.species.create(species_type=model_species_type, compartment=model_compartment)
            model_species.id = model_species.gen_id()	
            # Add concentration to fulfill wc_sim validation            
            conc_model = model.distribution_init_concentrations.create(species=model_species, mean=1000., std=0.,
            	units=unit_registry.parse_units('molecule'))
            conc_model.id = conc_model.gen_id()
            if not np.isnan(species.getInitialAmount()):
                conc_model.mean = species.getInitialAmount()
            # Create unbounded exchange reactions for boundary species
            if species.getBoundaryCondition():
                model_rxn = model.reactions.create(
            	    id='EX_' + species.getId(), 
            	    submodel=dfba_submodel,
            	    reversible=True)            
                model_rxn.participants.add(model_species.species_coefficients.get_or_create(
          	        coefficient=-1))
                model_rxn.flux_bounds = wc_lang.FluxBounds(units=unit_registry.parse_units('M s^-1'))
                model_rxn.flux_bounds.min = np.nan
                model_rxn.flux_bounds.max = np.nan    

        # Extract flux bounds
        flux_bounds = {}
        for bound in fbc_sbml_model.getListOfFluxBounds():
            rxn_id = bound.getReaction()
            if rxn_id not in flux_bounds:
                flux_bounds[rxn_id] = {'lower_bound': None, 'upper_bound': None}
            if bound.getOperation()=='greaterEqual':
                flux_bounds[rxn_id]['lower_bound'] = bound.getValue()
            elif bound.getOperation()=='lessEqual':
                flux_bounds[rxn_id]['upper_bound'] = bound.getValue()          

        # Create model reactions
        for rxn in sbml_model.getListOfReactions():        	
            fbc_rxn = rxn.getPlugin('fbc')  
            
            model_rxn = model.reactions.create(
            	id=rxn.getId(), 
            	submodel=dfba_submodel,
            	reversible=rxn.getReversible())
            # Add reaction participants
            for reactant in rxn.getListOfReactants():
                model_species_type = model.species_types.get_one(id=reactant.getSpecies())
                model_species = model.species.get_one(
                    species_type=model_species_type, compartment=model_compartment)
                model_rxn.participants.add(model_species.species_coefficients.get_or_create(
          	        coefficient=reactant.getStoichiometry()))
            for product in rxn.getListOfProducts():
                model_species_type = model.species_types.get_one(id=reactant.getSpecies())
                model_species = model.species.get_one(
                    species_type=model_species_type, compartment=model_compartment)
                model_rxn.participants.add(model_species.species_coefficients.get_or_create(
          	        coefficient=reactant.getStoichiometry()))
            # Add flux bounds
            model_rxn.flux_bounds = wc_lang.FluxBounds(units=unit_registry.parse_units('M s^-1'))
            if flux_bounds:                
                lower_bound = flux_bounds[model_rxn.id]['lower_bound']                
                upper_bound = flux_bounds[model_rxn.id]['upper_bound']                
            else:
                lower_bound_id = fbc_rxn.getLowerFluxBound()
                lower_bound = sbml_model.getParameter(lower_bound_id).value
                upper_bound_id = fbc_rxn.getUpperFluxBound()
                upper_bound = sbml_model.getParameter(upper_bound_id).value
            model_rxn.flux_bounds.min = np.nan if np.isinf(lower_bound) else lower_bound
            model_rxn.flux_bounds.max = np.nan if np.isinf(upper_bound) else upper_bound       
               
        # Add objective function
        dfba_submodel.dfba_obj = wc_lang.DfbaObjective(model=model)
        dfba_submodel.dfba_obj.id = dfba_submodel.dfba_obj.gen_id()

        sbml_objective_id = fbc_sbml_model.getListOfObjectives().getActiveObjective()
        sbml_objective_function = fbc_sbml_model.getObjective(sbml_objective_id)
        self.objective_direction = sbml_objective_function.getType()
        
        objective_terms = []
        rxn_objects = {}
        for rxn in sbml_objective_function.getListOfFluxObjectives():
            rxn_id = rxn.getReaction()
            objective_terms.append('{} * {}'.format(
            	str(rxn.getCoefficient()), rxn_id))  
            rxn_objects[rxn_id] = model.reactions.get_one(id=rxn_id)

        obj_expression = ' + '.join(objective_terms)
        dfba_obj_expression, error = wc_lang.DfbaObjectiveExpression.deserialize(
        obj_expression, {wc_lang.Reaction: rxn_objects})
        assert error is None, str(error)
        dfba_submodel.dfba_obj.expression = dfba_obj_expression 

        return model  


class FbaResultsComparator(ResultsComparator):
    """ Compare simulated flux predictions against expected fluxes

    Attributes:
        dfba_submodel (:obj:`wc_sim.submodels.dfba.DfbaSubmodel`): dynamic FBA submodel
    """
    def __init__(self, verification_test_reader, simulation_run_results, dfba_submodel):
    	super().__init__(verification_test_reader, simulation_run_results, dfba_submodel)
    	self.dfba_submodel = dfba_submodel

    def quantify_stoch_diff(self, evaluate=False):
        pass

    def differs(self):
        """ Evaluate whether the reaction fluxes predicted by simulation run differ from the correct fluxes
        
        Returns:
            :obj:`obj`: `False` if fluxes in the simulation run and the correct fluxes are equal
                within tolerances, otherwise :obj:`list`: of reaction IDs whose fluxes differ
        """       	
        differing_values = []        
        kwargs = self.prepare_tolerances()
        predicted_fluxes = self.dfba_submodel.reaction_fluxes
        predicted_fluxes['OBJF'] = self.dfba_submodel._optimal_obj_func_value
        # for each prediction, determine if its value is close enough to the expected predictions
        for reaction_id in self.verification_test_reader.settings['variables']:            
            if not np.allclose(self.verification_test_reader.expected_predictions_df[species_type].values,
                predicted_fluxes[reaction_id], **kwargs):
                differing_values.append(reaction_id)
        return differing_values or False


class FbaCaseVerifier(object):
    """ Verify or evaluate a test case
    
    Attributes:
        results_comparator (:obj:`ResultsComparator`): object that compares expected and actual predictions
        simulation_run_results (:obj:`RunResults`): results for a simulation run
        test_case_dir (:obj:`str`): directory containing the test case
        tmp_results_dir (:obj:`str`): temporary directory for simulation results
        verification_test_reader (:obj:`VerificationTestReader`): the test case's reader
        comparison_result (:obj:`obj`): `False` if fluxes in the expected result and simulation run
            are equal within tolerances, otherwise :obj:`list`: of reaction IDs whose fluxes differ
    """
    def __init__(self, test_cases_root_dir, test_case_type, test_case_num):
        """ Read model, config and expected predictions
        
        Args:
            test_cases_root_dir (:obj:`str`): pathname of directory containing test case files
            test_case_type (:obj:`str`): the type of case, `DYNAMIC_FLUX_BALANCE_ANALYSIS`
            test_case_num (:obj:`str`): unique id of a verification case
        """
        self.test_case_dir = os.path.join(test_cases_root_dir,
                                          VerificationTestCaseType[test_case_type].value, test_case_num)
        self.verification_test_reader = FbaVerificationTestReader(test_case_type, self.test_case_dir,
                                                                  test_case_num)
        self.verification_test_reader.run()
        
    def verify_model(self, discard_run_results=True, evaluate=False):
        """ Verify a model
        
        Args:
            discard_run_results (:obj:`bool`, optional): whether to discard run results
            evaluate (:obj:`bool`, optional): control the return value
                    
        Returns:
            :obj:`obj`: if `evaluate` is `False`, then return `False` if fluxes in the expected
                result and simulation run are equal within tolerances, otherwise :obj:`list`: of reaction
                IDs whose fluxes differ;        
        """
        # prepare for simulation
        self.tmp_results_dir = tmp_results_dir = tempfile.mkdtemp()
        simul_kwargs = dict(time_max=1.,
                            checkpoint_period=1.,
                            results_dir=tmp_results_dir,
                            verbose=False,
                            dfba_time_step=1.)

        model_compartment = self.verification_test_reader.model.compartments
        if len(model_compartment) == 1:       	
            simul_kwargs['options'] = dict(DfbaSubmodel=dict(options=dict(
        	    flux_bounds_volumetric_compartment_id=model_compartment[0].id)))

        # run simulation
        simulation = Simulation(self.verification_test_reader.model)
        results_dir = simulation.run(**simul_kwargs).results_dir
        self.simulation_run_results = RunResults(results_dir)

        # compare results
        self.results_comparator = FbaResultsComparator(self.verification_test_reader, self.simulation_run_results)
        self.comparison_result = self.results_comparator.differs()

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
        summary.append("model compartment: {}".format(cmpt.id))
        reaction_participant_attribute = ReactionParticipantAttribute()
        for sm in mdl.submodels:
            summary.append("submodel {}:".format(sm.id))
            summary.append("framework {}:".format(sm.framework.name.title()))
            for rxn in sm.reactions:
                summary.append("rxn & rl & bounds '{}': {}, ({}, {}), {}".format(rxn.id,
                    reaction_participant_attribute.serialize(rxn.participants),
                    str(rxn.flux_bounds.min) if rxn.flux_bounds else 'na',
                    str(rxn.flux_bounds.max) if rxn.flux_bounds else 'na',
                    rxn.rate_laws[0].expression.serialize()))
        for param in mdl.get_parameters():
            summary.append("param: {}={} ({})".format(param.id, param.value, param.units))
        return summary

    def get_test_case_summary(self):
        """ Summarize the test case
        """
        summary = ['Test Case Summary']
        if self.comparison_result:
            summary.append("Failing reactions: {}".format(', '.join(self.comparison_result)))
        else:
            summary.append('All reaction fluxes verify')
        summary.append("Test case type: {}".format(self.verification_test_reader.test_case_type.name))
        summary.append("Test case number: {}".format(self.verification_test_reader.test_case_num))
        return summary


class FbaVerificationSuite(VerificationSuite):
    """ Manage a suite of verification tests of `wc_sim`'s flux balance steady state behavioor
    
    Attributes:
        cases_dir (:obj:`str`): path to cases directory
        results (:obj:`list` of :obj:`VerificationRunResult`): a result for each test
    """
    def __init__(self, cases_dir):
        super().__init__(cases_dir)

    def _run_test(self, case_type_name, case_num, verbose=False, evaluate=False):
        """ Run one test case and record the result

        The results from running a test case are recorded by calling `self._record_result()`.
        It records results in the list `self.results`. All types of results, including exceptions,
        are recorded.
        
        Args:
            case_type_name (:obj:`str`): the type of case, a name in `VerificationTestCaseType`
            case_num (:obj:`str`): unique id of a verification case
            verbose (:obj:`bool`, optional): whether to produce verbose output
            evaluate (:obj:`bool`, optional): whether to quantitatively evaluate the test case;
                if `False`, then indicate whether the test passed in the saved result's `result_type`;
                otherwise, provide the flux divergence in the result's `quant_diff`
        
        Returns:
            :obj:`None`: results are recorded in `self.results`
        """
        try:
            case_verifier = FbaCaseVerifier(self.cases_dir, case_type_name, case_num)
        except:
            tb = traceback.format_exc()
            self._record_result(case_type_name, case_num, VerificationResultType.CASE_UNREADABLE,
                                float('nan'), error=tb)
            return

        # assemble kwargs
        kwargs = {}

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

    def _run_tests(self):
        """ Run one or more tests"""
        pass
    
    def run(self, test_case_type_name=None, cases=None, verbose=False, empty_results=True,
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
            verbose (:obj:`bool`, optional): whether to produce verbose output
            empty_results (:obj:`bool`, optional): whether to empty the list of verification run results
            evaluate (:obj:`bool`, optional): whether to quantitatively evaluate the test case(s)
        
        Returns:
            :obj:`list`: of :obj:`VerificationRunResult`: the results for this :obj:`FbaVerificationSuite`
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
            if test_case_type_name != VerificationTestCaseType.DYNAMIC_FLUX_BALANCE_ANALYSID:
                raise VerificationError("VerificationTestCaseType is '{}' and not "
            	                    "'DYNAMIC_FLUX_BALANCE_ANALYSIS'".format(test_case_type_name))
            if cases is None:
                cases = os.listdir(os.path.join(self.cases_dir,
                                                VerificationTestCaseType[test_case_type_name].value))
            for case_num in cases:
                self._run_test(test_case_type_name, case_num, verbose=verbose, evaluate=evaluate)
        else:
            for verification_test_case_type in VerificationTestCaseType:
                for case_num in os.listdir(os.path.join(self.cases_dir, verification_test_case_type.value)):
                    self._run_test(verification_test_case_type.name, case_num,
                                    verbose=verbose, evaluate=evaluate)
        return self.results

    
    def run_multialg(self):
        """ Test a suite of multialgorithmic models """
        pass
