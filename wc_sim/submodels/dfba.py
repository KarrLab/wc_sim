""" A submodel that uses Dynamic Flux Balance Analysis (dFBA) to model a set of reactions

:Author: Yin Hoon Chew <yinhoon.chew@mssm.edu>
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2020-07-29
:Copyright: 2016-2020, Karr Lab
:License: MIT
"""

import collections
import conv_opt
import math
import scipy.constants
import wc_lang

from de_sim.simulation_object import SimulationObject
from wc_sim import message_types
from wc_sim.config import core as config_core_multialgorithm
from wc_sim.multialgorithm_errors import DynamicMultialgorithmError, MultialgorithmError
from wc_sim.species_populations import TempPopulationsLSP
from wc_sim.submodels.dynamic_submodel import ContinuousTimeSubmodel


class DfbaSubmodel(ContinuousTimeSubmodel):
    """ Use dynamic Flux Balance Analysis to predict the dynamics of chemical species in a container

    Attributes:
        reaction_fluxes (:obj:`dict`): pre-allocated reaction fluxes
        dfba_solver_options (:obj:`dict`): options for solving DFBA submodel
        _conv_model (:obj:`conv_opt.Model`): linear programming model in conv_opt format
        _conv_variables (:obj:`dict`): a dictionary mapping reaction IDs to the associated 
            `conv_opt.Variable` objects
        _conv_metabolite_matrices (:obj:`dict`): a dictionary mapping metabolite species IDs to lists
            of `conv_opt.LinearTerm` objects defining the reactions and stoichiometries that the 
            species participate in
        _reaction_bounds (:obj:`dict`): a dictionary with reaction IDs as keys and tuples of scaled minimum
            and maximum bounds as values
        _optimal_obj_func_value (:obj:`float`): the value of objective function returned by the solver            
    """
    DFA_BOUND_SCALE_FACTOR = 1
    DFBA_COEF_SCALE_FACTOR = 1
    SOLVER = 1
    PRESOLVE = 1
    SOLVER_OPTIONS = {
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
    FLUX_BOUNDS_VOLUMETRIC_COMPARTMENT_ID = 'wc'

    # register the message types sent by DfbaSubmodel
    messages_sent = [message_types.RunFba]

    # register 'handle_RunFba_msg' to handle RunFba events
    event_handlers = [(message_types.RunFba, 'handle_RunFba_msg')]

    time_step_message = message_types.RunFba

    def __init__(self, id, dynamic_model, reactions, species, dynamic_compartments,
                 local_species_population, dfba_time_step, options=None):
        """ Initialize a dFBA submodel instance
        Args:
            id (:obj:`str`): unique id of this dFBA submodel
            dynamic_model (:obj: `DynamicModel`): the simulation's central coordinator
            reactions (:obj:`list` of `wc_lang.Reaction`): the reactions modeled by this dFBA submodel
            species (:obj:`list` of `wc_lang.Species`): the species that participate in the reactions
                modeled by this dFBA submodel
            dynamic_compartments (:obj: `dict`): `DynamicCompartment`s, keyed by id, that contain
                species which participate in reactions that this dFBA submodel models, including
                adjacent compartments used by its transfer reactions
            local_species_population (:obj:`LocalSpeciesPopulation`): the store that maintains this
                dFBA submodel's species population               
            dfba_time_step (:obj:`float`): time interval between FBA optimization
            options (:obj:`dict`, optional): dFBA submodel options
        """
        super().__init__(id, dynamic_model, reactions, species, dynamic_compartments,
                         local_species_population, dfba_time_step, options)
        
        dfba_objective_id = 'dfba-obj-{}'.format(id)
        if dfba_objective_id not in dynamic_model.dynamic_dfba_objectives:
            raise MultialgorithmError(f"DfbaSubmodel {self.id}: cannot find dynamic_dfba_objective")
        self.dfba_objective = dynamic_model.dynamic_dfba_objectives[dfba_objective_id].wc_lang_expression
        
        self.dfba_solver_options = {
            'dfba_bound_scale_factor': self.DFA_BOUND_SCALE_FACTOR,
            'dfba_coef_scale_factor': self.DFBA_COEF_SCALE_FACTOR,
            'solver': self.SOLVER,
            'presolve': self.PRESOLVE,
            'solver_options': self.SOLVER_OPTIONS,
            'flux_bounds_volumetric_compartment_id': self.FLUX_BOUNDS_VOLUMETRIC_COMPARTMENT_ID,
        }
        if options is not None:
            if 'dfba_bound_scale_factor' in options:
                if options['dfba_bound_scale_factor'] <= 0.:
                    raise MultialgorithmError(f"DfbaSubmodel {self.id}: dfba_bound_scale_factor must"
                        f" be larger than zero but is {options['dfba_bound_scale_factor']}")
                self.dfba_solver_options['dfba_bound_scale_factor'] = options['dfba_bound_scale_factor']
            if 'dfba_coef_scale_factor' in options:
                if options['dfba_coef_scale_factor'] <= 0.:
                    raise MultialgorithmError(f"DfbaSubmodel {self.id}: dfba_coef_scale_factor must"
                        f" be larger than zero but is {options['dfba_coef_scale_factor']}")
                self.dfba_solver_options['dfba_coef_scale_factor'] = options['dfba_coef_scale_factor']
            if 'solver' in options:
                try:
                    selected_solver = conv_opt.Solver(options['solver'])
                except:
                    raise MultialgorithmError(f"DfbaSubmodel {self.id}: {options['solver']}"
                        f" is not a valid Solver")    
                self.dfba_solver_options['solver'] = options['solver']
            if 'presolve' in options:
                try:
                    selected_presolve = conv_opt.Presolve(options['presolve'])
                except:
                    raise MultialgorithmError(f"DfbaSubmodel {self.id}: {options['presolve']}"
                        f" is not a valid Presolve option")
                self.dfba_solver_options['presolve'] = options['presolve']
            if 'solver_options' in options:
                selected_solver = conv_opt.Solver(self.dfba_solver_options['solver'])
                if selected_solver.name not in options['solver_options']:
                    raise MultialgorithmError(f"DfbaSubmodel {self.id}: the solver key in"
                        f" solver_options is not the same as the selected solver"
                        f" '{selected_solver.name}'")
                self.dfba_solver_options['solver_options'] = options['solver_options']
            if 'flux_bounds_volumetric_compartment_id' in options:
                comp_id = options['flux_bounds_volumetric_compartment_id']
                if comp_id != self.FLUX_BOUNDS_VOLUMETRIC_COMPARTMENT_ID:
                    try:
                        flux_bound_comp = self.dynamic_model.dynamic_compartments[comp_id]
                    except:
                        raise MultialgorithmError(f"DfbaSubmodel {self.id}: the user-provided"
                            f" flux_bounds_volumetric_compartment_id '{comp_id}' is not the ID"
                            f" of a compartment in the model")      
                                        
        # log initialization data
        self.log_with_time("init: id: {}".format(id))
        self.log_with_time("init: time_step: {}".format(str(dfba_time_step)))

        ### dFBA specific code ###
        self.set_up_dfba_submodel()
        self.set_up_optimizations()
        
    def set_up_dfba_submodel(self):
        """ Set up a dFBA submodel, by converting to a linear programming matrix """
        coef_scale_factor = self.dfba_solver_options['dfba_coef_scale_factor']
        self.set_up_continuous_time_submodel()

        ### dFBA specific code ###
        # Formulate the optimization problem using the conv_opt package
        self._conv_model = conv_opt.Model(name='model')
        self._conv_variables = {}
        self._conv_metabolite_matrices = collections.defaultdict(list)
        for rxn in self.reactions:
            self._conv_variables[rxn.id] = conv_opt.Variable(
                name=rxn.id, type=conv_opt.VariableType.continuous)
            self._conv_model.variables.append(self._conv_variables[rxn.id])
            for part in rxn.participants:
                self._conv_metabolite_matrices[part.species.id].append(
                    conv_opt.LinearTerm(self._conv_variables[rxn.id], part.coefficient))                             

        for rxn_cls in self.dfba_objective.related_objects.values():
            for rxn_id, rxn in rxn_cls.items():
                if rxn_id not in self._conv_variables:
                    self._conv_variables[rxn.id] = conv_opt.Variable(
                        name=rxn.id, type=conv_opt.VariableType.continuous, lower_bound=0)
                    self._conv_model.variables.append(self._conv_variables[rxn.id])
                    for part in rxn.dfba_obj_species:
                        self._conv_metabolite_matrices[part.species.id].append(
                            conv_opt.LinearTerm(self._conv_variables[rxn.id], 
                                part.value*coef_scale_factor))        

        for met_id, expression in self._conv_metabolite_matrices.items():
            self._conv_model.constraints.append(conv_opt.Constraint(expression, name=met_id, 
                upper_bound=0.0, lower_bound=0.0)) 

        # Set up the objective function
        dfba_obj_expr_objs = self.dfba_objective.lin_coeffs
        for rxn_coeffs in dfba_obj_expr_objs.values():
            for rxn, lin_coef in rxn_coeffs.items():
                self._conv_model.objective_terms.append(conv_opt.LinearTerm(
                    self._conv_variables[rxn.id], lin_coef))
                
        self._conv_model.objective_direction = conv_opt.ObjectiveDirection.maximize

        # Set options for conv_opt solver
        options = conv_opt.SolveOptions(
            solver=conv_opt.Solver(self.dfba_solver_options['solver']),
            presolve=conv_opt.Presolve(self.dfba_solver_options['presolve']),
            solver_options=self.dfba_solver_options['solver_options'])

    def set_up_optimizations(self):
        """ To improve performance, pre-compute and pre-allocate some data structures """
        self.set_up_continuous_time_optimizations()
        # pre-allocate dict of reaction fluxes
        self.reaction_fluxes = {rxn.id: None for rxn in self.reactions}

    def determine_bounds(self):
        """ Determine the minimum and maximum bounds for each reaction. The bounds will be
            scaled by multiplying to `dfba_bound_scale_factor` and written to `self._reaction_bounds`                 
        """
        # Determine all the reaction bounds
        bound_scale_factor = self.dfba_solver_options['dfba_bound_scale_factor']
        flux_comp_id = self.dfba_solver_options['flux_bounds_volumetric_compartment_id']
        if flux_comp_id == self.FLUX_BOUNDS_VOLUMETRIC_COMPARTMENT_ID:
            flux_comp_volume = self.dynamic_model.cell_volume() 
        else:
            flux_comp_volume = self.dynamic_model.dynamic_compartments[flux_comp_id].volume()    
        self._reaction_bounds = {}        
        for reaction in self.reactions:
            # Set the bounds of exchange/demand/sink reactions        
            if reaction.flux_bounds:
                rxn_bounds = reaction.flux_bounds
                if rxn_bounds.min:
                    min_constr = 0. if math.isnan(rxn_bounds.min) else \
                        rxn_bounds.min*flux_comp_volume*scipy.constants.Avogadro*bound_scale_factor
                else:
                    min_constr = rxn_bounds.min
                if rxn_bounds.max:
                    max_constr = 0. if math.isnan(rxn_bounds.max) else \
                        rxn_bounds.max*flux_comp_volume*scipy.constants.Avogadro*bound_scale_factor
                else:           
                    max_constr = rxn_bounds.max
            # Set the bounds of reactions with measured kinetic constants
            elif reaction.rate_laws:                
                for_ratelaw = reaction.rate_laws.get_one(direction=wc_lang.RateLawDirection.forward)
                if for_ratelaw:
                    max_constr = for_ratelaw.expression._parsed_expression.eval({
                                    wc_lang.Species: {i.id: i.distribution_init_concentration.mean \
                                        for i in for_ratelaw.expression.species}
                                        }) * bound_scale_factor
                else:
                    max_constr = None        
                
                rev_ratelaw = reaction.rate_laws.get_one(direction=wc_lang.RateLawDirection.backward)
                if rev_ratelaw:
                    min_constr = -rev_ratelaw.expression._parsed_expression.eval({
                                    wc_lang.Species: {i.id: i.distribution_init_concentration.mean \
                                        for i in rev_ratelaw.expression.species}
                                        }) * bound_scale_factor
                elif reaction.reversible:
                    min_constr = None
                else:
                    min_constr = 0.
            # Set other reactions to be unbounded                
            else:                
                max_constr = None
                if reaction.reversible:
                    min_constr = None                   
                else:
                    min_constr = 0.    
            
            self._reaction_bounds[reaction.id] = (min_constr, max_constr)

    def run_fba_solver(self):
        """ Run the FBA solver for one time step """
        bound_scale_factor = self.dfba_solver_options['dfba_bound_scale_factor']
        coef_scale_factor = self.dfba_solver_options['dfba_coef_scale_factor']
        self.determine_bounds()
        for rxn_id, (min_constr, max_constr) in self._reaction_bounds.items():
            self._conv_variables[rxn_id].lower_bound = min_constr
            self._conv_variables[rxn_id].upper_bound = max_constr
        
        end_time = self.time + self.time_step
        result = self._conv_model.solve()        
        if result.status_code != 0:
            raise DynamicMultialgorithmError(self.time, f"DfbaSubmodel {self.id}: "
                                                        f"No optimal solution found: "
                                                        f"'{result.status_message}' "
                                                        f"for time step [{self.time}, {end_time})")
        self._optimal_obj_func_value = result.value / bound_scale_factor * coef_scale_factor
        for rxn_variable in self._conv_model.variables:
            self.reaction_fluxes[rxn_variable.name] = rxn_variable.primal / bound_scale_factor

        # Calculate the adjustment for each species as sum over reactions of reaction flux * stoichiometry       
        self.current_species_populations()
        for idx, species_id in enumerate(self.species_ids):
            population_change_rate = 0
            for rxn_term in self._conv_metabolite_matrices[species_id]:
                population_change_rate += self.reaction_fluxes[rxn_term.variable.name] * rxn_term.coefficient
            temp_new_population = self.populations[idx] + population_change_rate
            if temp_new_population < 0.:
                raise DynamicMultialgorithmError(self.time, f"DfbaSubmodel {self.id}: "
                                                            f"Negative population found for"
                                                            f"{species_id} from {self.populations[idx]} "
                                                            f"to {temp_new_population} "
                                                            f"for time step [{self.time}, {end_time})")
            self.adjustments[species_id] = population_change_rate
        
        ### store results in local_species_population ###
        self.local_species_population.adjust_continuously(self.time, self.adjustments)

        # flush expressions that depend on species and reactions modeled by this FBA submodel from cache
        # AG: we can rename "ode_flush_after_populations_change" to
        # "continuous_submodel_flush_after_populations_change" and use it both here and in odes.py
        self.dynamic_model.fba_flush_after_populations_change(self.id)

    ### handle DES events ###
    def handle_RunFba_msg(self):
        """ Handle an event containing a RunFba message

        Args:
            event (:obj:`Event`): a simulation event
        """
        self.run_fba_solver()
        self.schedule_next_periodic_analysis()
