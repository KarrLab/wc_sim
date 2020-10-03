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

from wc_sim import message_types
from wc_sim.config import core as config_core_multialgorithm
from wc_sim.multialgorithm_errors import DynamicMultialgorithmError, MultialgorithmError
from wc_sim.species_populations import TempPopulationsLSP
from wc_sim.submodels.dynamic_submodel import ContinuousTimeSubmodel


class DfbaSubmodel(ContinuousTimeSubmodel):
    """ Use dynamic Flux Balance Analysis to predict the dynamics of chemical species in a container

    Attributes:
        DFBA_BOUND_SCALE_FACTOR (:obj:`float`): scaling factor for the reaction bounds,
            the default value is 1.
        DFBA_COEF_SCALE_FACTOR (:obj:`float`): scaling factor for the stoichiometric
            coefficients in the objective reactions, the default value is 1.
        SOLVER (:obj:`str`): name of the selected solver in conv_opt, the default
            value is 'cplex'
        PRESOLVE (:obj:`str`): presolve mode in `conv_opt` ('auto', 'on', 'off'), the default
            value is 'on'
        SOLVER_OPTIONS (:obj:`dict`): parameters for the solver
        dfba_solver_options (:obj:`dict`): options for solving DFBA submodel
        OPTIMIZATION_TYPE (:obj:`str`): direction of optimization ('maximize', 'max',
            'minimize', 'min'), the default value is 'maximize'
        FLUX_BOUNDS_VOLUMETRIC_COMPARTMENT_ID (:obj:`str`): id of the compartment where the
            measured flux bounds are normalized to, the default is the whole-cell
        reaction_fluxes (:obj:`dict`): reaction fluxes data structure, which is pre-allocated
        # APG: let's call this dfba_obj_expr, so its type is clearer
        dfba_objective (:obj:`ParsedExpression`): an analyzed and validated dFBA objective expression
        exchange_rxns (:obj:`set` of :obj:`wc_lang.Reactions`): set of exchange/demand reactions
        _multi_reaction_constraints (:obj:`dict`): constraints to avoid negative populations
        _constrained_exchange_rxns (:obj:`set`): exchange reactions constrained by
            `_multi_reaction_constraints`
        _conv_model (:obj:`conv_opt.Model`): linear programming model in `conv_opt` format
        _conv_variables (:obj:`dict`): a dictionary mapping reaction IDs to the associated
            `conv_opt.Variable` objects
        _conv_metabolite_matrices (:obj:`dict`): a dictionary mapping metabolite species IDs to lists
            of :obj:`conv_opt.LinearTerm` objects; each :obj:`conv_opt.LinearTerm` associates a
            a reaction that the species participates in with the species' stoichiometry in the reaction
        _dfba_obj_rxn_ids (:obj:`list`): a list of IDs of DFBA objective reactions
        _dfba_obj_species (:obj:`list`): all species in :obj:`DfbaObjReaction`\ s used by `dfba_objective`
        _reaction_bounds (:obj:`dict`): a dictionary that maps reaction IDs to tuples of scaled minimum
            and maximum bounds
        _optimal_obj_func_value (:obj:`float`): the value of objective function returned by the solver
    """
    DFBA_BOUND_SCALE_FACTOR = 1.
    DFBA_COEF_SCALE_FACTOR = 1.
    SOLVER = 'cplex'
    PRESOLVE = 'on'
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
    OPTIMIZATION_TYPE = 'maximize'
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

        Raises:
            :obj:`MultiAlgorithmError`: if the `dynamic_dfba_objective` cannot be found,
                or if the provided 'dfba_bound_scale_factor' in options has a negative value,
                or if the provided 'dfba_coef_scale_factor' in options has a negative value,
                or if the provided 'solver' in options is not a valid value,
                or if the provided 'presolve' in options is not a valid value,
                or if the 'solver' value provided in the 'solver_options' in options is not the same
                as the name of the selected `conv_opt.Solver`, or if the provided
                'flux_bounds_volumetric_compartment_id' is not a valid compartment ID in the model
        """
        super().__init__(id, dynamic_model, reactions, species, dynamic_compartments,
                         local_species_population, dfba_time_step, options)

        dfba_objective_id = 'dfba-obj-{}'.format(id)
        if dfba_objective_id not in dynamic_model.dynamic_dfba_objectives: # pragma: no cover
            raise MultialgorithmError(f"DfbaSubmodel {self.id}: cannot find dynamic_dfba_objective "
                                      f"{dfba_objective_id}") # pragma: no cover
        self.dfba_objective = dynamic_model.dynamic_dfba_objectives[dfba_objective_id].wc_lang_expression
        # APG: is this foolproof? could a modeler have a pair of reactions of the form A -> (/), (/) -> A
        # or would a model containing those reactions be rejected?
        self.exchange_rxns = set([rxn for rxn in self.reactions if len(rxn.participants) == 1])

        self.dfba_solver_options = {
            'dfba_bound_scale_factor': self.DFBA_BOUND_SCALE_FACTOR,
            'dfba_coef_scale_factor': self.DFBA_COEF_SCALE_FACTOR,
            'solver': self.SOLVER,
            'presolve': self.PRESOLVE,
            'solver_options': self.SOLVER_OPTIONS,
            'optimization_type': self.OPTIMIZATION_TYPE,
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
                if options['solver'] not in conv_opt.Solver.__members__:
                    raise MultialgorithmError(f"DfbaSubmodel {self.id}: {options['solver']}"
                        f" is not a valid Solver")
                self.dfba_solver_options['solver'] = options['solver']
            if 'presolve' in options:
                if options['presolve'] not in conv_opt.Presolve.__members__:
                    raise MultialgorithmError(f"DfbaSubmodel {self.id}: {options['presolve']}"
                        f" is not a valid Presolve option")
                self.dfba_solver_options['presolve'] = options['presolve']
            if 'solver_options' in options:
                if self.dfba_solver_options['solver'] not in options['solver_options']:
                    raise MultialgorithmError(f"DfbaSubmodel {self.id}: the solver key in"
                        f" solver_options is not the same as the selected solver"
                        f" '{self.dfba_solver_options['solver']}'")
                self.dfba_solver_options['solver_options'] = options['solver_options']
            if 'optimization_type' in options:
                if options['optimization_type'] not in conv_opt.ObjectiveDirection.__members__:
                    raise MultialgorithmError(f"DfbaSubmodel {self.id}: the optimization_type in"
                        f" options can only take 'maximize', 'max', 'minimize' or 'min' as value but is"
                        f" '{options['optimization_type']}'")
                self.dfba_solver_options['optimization_type'] = options['optimization_type']
            if 'flux_bounds_volumetric_compartment_id' in options:
                comp_id = options['flux_bounds_volumetric_compartment_id']
                if comp_id != self.FLUX_BOUNDS_VOLUMETRIC_COMPARTMENT_ID:
                    try:
                        flux_bound_comp = self.dynamic_model.dynamic_compartments[comp_id]
                    except:
                        raise MultialgorithmError(f"DfbaSubmodel {self.id}: the user-provided"
                            f" flux_bounds_volumetric_compartment_id '{comp_id}' is not the ID"
                            f" of a compartment in the model")
                self.dfba_solver_options['flux_bounds_volumetric_compartment_id'] = comp_id

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
            # APG: subtle problem here: since a DfbaObjReaction and a Reaction
            # could have the same id they could collide in _conv_variables. Two possible solutions:
            # 1) prohibit models that have the same rxn id used more than once, or
            # 2) generate ids that include the reaction type
            # E.g.:
            # make unique identifiers by prefixing ids with the class names
            # def gen_uniq_rxn_id(rxn):
            #    return f"{type(rxn).__name__}:{rxn.id}"
            # I prefer #2
            self._conv_variables[rxn.id] = conv_opt.Variable(
                name=rxn.id, type=conv_opt.VariableType.continuous)
            self._conv_model.variables.append(self._conv_variables[rxn.id])
            for part in rxn.participants:
                self._conv_metabolite_matrices[part.species.id].append(
                    conv_opt.LinearTerm(self._conv_variables[rxn.id], part.coefficient))

        self._dfba_obj_rxn_ids = []
        self._dfba_obj_species = []
        # APG: renamed variable to a properly descriptive name
        for rxn_id_2_rxn_map in self.dfba_objective.related_objects.values():
            for rxn_id, rxn in rxn_id_2_rxn_map.items():
                if rxn_id not in self._conv_variables:
                    self._dfba_obj_rxn_ids.append(rxn_id)
                    self._conv_variables[rxn.id] = conv_opt.Variable(
                        name=rxn.id, type=conv_opt.VariableType.continuous, lower_bound=0)
                    self._conv_model.variables.append(self._conv_variables[rxn.id])
                    for part in rxn.dfba_obj_species:
                        self._dfba_obj_species.append(part)
                        self._conv_metabolite_matrices[part.species.id].append(
                            conv_opt.LinearTerm(self._conv_variables[rxn.id],
                                part.value * coef_scale_factor))

        for met_id, expression in self._conv_metabolite_matrices.items():
            self._conv_model.constraints.append(conv_opt.Constraint(expression, name=met_id,
                upper_bound=0.0, lower_bound=0.0))

        # Set up the objective function
        dfba_obj_expr_objs = self.dfba_objective.lin_coeffs
        for rxn_cls in dfba_obj_expr_objs.values():
            for rxn, lin_coef in rxn_cls.items():
                self._conv_model.objective_terms.append(conv_opt.LinearTerm(
                    self._conv_variables[rxn.id], lin_coef))

        self._conv_model.objective_direction = \
            conv_opt.ObjectiveDirection[self.dfba_solver_options['optimization_type']]

        # Set options for conv_opt solver
        options = conv_opt.SolveOptions(
            solver=conv_opt.Solver[self.dfba_solver_options['solver']],
            presolve=conv_opt.Presolve[self.dfba_solver_options['presolve']],
            solver_options=self.dfba_solver_options['solver_options'])

        self._multi_reaction_constraints = self.initialize_neg_species_pop_constraints()

    @staticmethod
    def _get_species_and_stoichiometry(reaction):
        """ Get a reaction's species and their net stoichiometries

        Handles both :obj:`wc_lang.Reaction` and :obj:`wc_lang.DfbaObjReaction` reactions.

        Args:
            reaction (:obj:`wc_lang.Reaction` or :obj:`wc_lang.DfbaObjReaction`): a reaction

        Returns:
            :obj:`dict`: map from species to net stoichiometry for each species in `reaction`,
            with entries which have net stoichiometry of 0. removed
        """
        # get net coefficients, since a species can participate in both sides of a reaction
        species_net_coefficients = collections.defaultdict(float)
        if isinstance(reaction, wc_lang.Reaction):
            for part in reaction.participants:
                species_net_coefficients[part.species] += part.coefficient

        # ignore coverage report of lack of false branch in this elif
        elif isinstance(reaction, wc_lang.DfbaObjReaction):
            for part in reaction.dfba_obj_species:
                species_net_coefficients[part.species] += part.value

        species_to_rm = [s for s in species_net_coefficients if species_net_coefficients[s] == 0.]
        for species in species_to_rm:
            del species_net_coefficients[species]
        return species_net_coefficients

    NEG_POP_CONSTRAINT_PREFIX = 'neg_pop_constraint'
    NEG_POP_CONSTRAINT_SEP = '__'
    LB = '__LB__'
    RB = '__RB__'

    # TODO: in species_id_without_brkts raise error if species_id doesn't have brackets or has codes
    @staticmethod
    def species_id_without_brkts(species_id):
        """ Replace brackets in a species id with codes

        Args:
            species_id (:obj:`str`): WC Lang species id

        Returns:
            :obj:`str`: species id with brackets replaced by codes
        """
        return species_id.replace('[', DfbaSubmodel.LB).replace(']', DfbaSubmodel.RB)

    # TODO: in species_id_without_brkts raise error if species_id doesn't have codes or has brackets
    @staticmethod
    def species_id_with_brkts(species_id):
        """ Replace codes in a species id with brackets

        Args:
            species_id (:obj:`str`): WC Lang species id with brackets replaced by codes

        Returns:
            :obj:`str`: standard WC Lang species id
        """
        return species_id.replace(DfbaSubmodel.LB, '[').replace(DfbaSubmodel.RB, ']')

    @staticmethod
    def gen_neg_species_pop_constraint_id(species_id):
        """ Generate a negative species population constraint id

        Args:
            species_id (:obj:`str`): id of species being constrained

        Returns:
            :obj:`str`: a negative species population constraint id
        """
        return DfbaSubmodel.NEG_POP_CONSTRAINT_PREFIX + DfbaSubmodel.NEG_POP_CONSTRAINT_SEP + \
            DfbaSubmodel.species_id_without_brkts(species_id)

    @staticmethod
    def parse_neg_species_pop_constraint_id(neg_species_pop_constraint_id):
        """ Parse a negative species population constraint id

        Args:
            neg_species_pop_constraint_id (:obj:`str`): a negative species population constraint id

        Returns:
            :obj:`str`: id of species being constrained
        """
        loc = len(DfbaSubmodel.NEG_POP_CONSTRAINT_PREFIX) + len(DfbaSubmodel.NEG_POP_CONSTRAINT_SEP)
        return DfbaSubmodel.species_id_with_brkts(neg_species_pop_constraint_id[loc:])

    def initialize_neg_species_pop_constraints(self):
        """ Make constraints that span multiple reactions

        A separate constraint is made for each species that participates in more than
        one exchange and/or objective reaction. These constraints prevent the species from
        having a negative species population in the next time step.
        Also initializes `self._constrained_exchange_rxns`.
        Call this when a dFBA submodel is initialized.

        Returns:
            :obj:`dict`: a map from constraint id to constraints stored in `self._conv_model.constraints`
        """
        # TODO: AVOID NEG. POPS.
        # either check that all reactions are irreversible or handle reversible reactions

        # map from species to reactions that use the species
        reactions_using_species = collections.defaultdict(set)

        # add species in objective reactions
        for rxn_cls in self.dfba_objective.related_objects:
            for rxn in self.dfba_objective.related_objects[rxn_cls].values():
                for species in self._get_species_and_stoichiometry(rxn):
                    reactions_using_species[species].add(rxn)

        # add species in exchange reactions
        for rxn in self.exchange_rxns:
            for species in self._get_species_and_stoichiometry(rxn):
                reactions_using_species[species].add(rxn)

        coef_scale_factor = self.dfba_solver_options['dfba_coef_scale_factor']
        self._constrained_exchange_rxns = set()
        multi_reaction_constraints = {}
        for species, rxns in reactions_using_species.items():
            if 1 < len(rxns):
                # create an expression for the species' net consumption rate, in molecules / sec
                # net consumption = sum(-coef(species) * flux) for rxns that use species as a reactant -
                #                   sum(coef(species) * flux) for rxns that use species as a product
                # which can be simplified as -sum(coef * flux) for all rxns that use species
                exchange_rxns_in_constr_expr = set()
                constr_expr = []
                for rxn in rxns:
                    for rxn_species, net_coef in self._get_species_and_stoichiometry(rxn).items():
                        if rxn_species == species:
                            net_consumption_coef = -net_coef
                            scaled_net_consumption_coef = net_consumption_coef
                            if rxn not in self.exchange_rxns:
                                scaled_net_consumption_coef = coef_scale_factor * net_consumption_coef
                            constr_expr.append(conv_opt.LinearTerm(self._conv_variables[rxn.id],
                                                                   scaled_net_consumption_coef))
                            if rxn in self.exchange_rxns:
                                exchange_rxns_in_constr_expr.add(rxn)

                # optimization: only create a Constraint when the species can be consumed,
                # which occurs when at least one of its net consumption coefficients is positive
                species_can_be_consumed = any([0 < linear_term.coefficient for linear_term in constr_expr])
                if species_can_be_consumed:
                    # upper_bound will be set by bound_neg_species_pop_constraints() before solving FBA
                    constraint_id = DfbaSubmodel.gen_neg_species_pop_constraint_id(species.id)
                    constraint = conv_opt.Constraint(constr_expr,
                                                     name=constraint_id,
                                                     lower_bound=0.0,
                                                     upper_bound=None)
                    multi_reaction_constraints[constraint_id] = constraint

                    # record the constrained exchange reactions
                    self._constrained_exchange_rxns |= exchange_rxns_in_constr_expr

        return multi_reaction_constraints

    def set_up_optimizations(self):
        """ To improve performance, pre-compute and pre-allocate some data structures """
        self.set_up_continuous_time_optimizations()
        # pre-allocate dict of reaction fluxes
        self.reaction_fluxes = {rxn.id: None for rxn in self.reactions}

    def determine_bounds(self):
        """ Determine the minimum and maximum bounds for each reaction

        Two actions:

        1. Bounds provided by the model are scaled by multiplying by `dfba_bound_scale_factor`
        and written to `self._reaction_bounds`
        2. The bounds in exchange reactions that do not participate in a constraint
        created by `initialize_neg_species_pop_constraints()` are further bounded so they do not cause
        negative species populations in the next time step.
        """
        # Determine all the reaction bounds
        bound_scale_factor = self.dfba_solver_options['dfba_bound_scale_factor']
        flux_comp_id = self.dfba_solver_options['flux_bounds_volumetric_compartment_id']
        # TODO APG: the use of 'flux_comp_volume' means that cached volumes must be invalidated
        # before running dFBA
        if flux_comp_id == self.FLUX_BOUNDS_VOLUMETRIC_COMPARTMENT_ID:
            flux_comp_volume = self.dynamic_model.cell_volume()
        else:
            flux_comp_volume = self.dynamic_model.dynamic_compartments[flux_comp_id].volume()
        self._reaction_bounds = {}
        for reaction in self.reactions:

            # Set the bounds of reactions with measured kinetic constants
            if reaction.rate_laws:
                for_ratelaw = reaction.rate_laws.get_one(direction=wc_lang.RateLawDirection.forward)
                if for_ratelaw:
                    # APG: we need to use the DynamicModel evaluator for the rate law; it needs to use
                    # the current species population, not the initial conc., and be able to use other WC
                    # Lang expressions
                    # APG: if reversible rxns are split, could call self.calc_reaction_rate(reaction)
                    rate = self.dynamic_model.dynamic_rate_laws[for_ratelaw.id].eval(self.time)
                    max_constr = bound_scale_factor * rate
                else:
                    max_constr = None

                rev_ratelaw = reaction.rate_laws.get_one(direction=wc_lang.RateLawDirection.backward)
                if rev_ratelaw:
                    rate = self.dynamic_model.dynamic_rate_laws[rev_ratelaw.id].eval(self.time)
                    min_constr = -bound_scale_factor * rate
                elif reaction.reversible:
                    min_constr = None
                else:
                    min_constr = 0.

            # Set the bounds of exchange/demand/sink reactions
            elif reaction.flux_bounds:

                rxn_bounds = reaction.flux_bounds
                if rxn_bounds.min:
                    # APG: one-line if statements was nicely compact, but coverage testing cant check both branches
                    if math.isnan(rxn_bounds.min):
                        min_constr = None
                    else:
                        min_constr = rxn_bounds.min * flux_comp_volume * scipy.constants.Avogadro * bound_scale_factor
                else:
                    min_constr = rxn_bounds.min
                if rxn_bounds.max:
                    # APG: see above RE one-line if statements
                    if math.isnan(rxn_bounds.max):
                        max_constr = None
                    else:
                        max_constr = rxn_bounds.max * flux_comp_volume * scipy.constants.Avogadro * bound_scale_factor
                else:
                    max_constr = rxn_bounds.max

            # Set other reactions to be unbounded
            else:
                max_constr = None
                if reaction.reversible:
                    min_constr = None
                else:
                    min_constr = 0.

            self._reaction_bounds[reaction.id] = (min_constr, max_constr)

        # bound exchange reactions so they do not cause negative species populations in the
        # next time step
        # if exchange reaction r transports species x[c] out of c
        # then bound the maximum flux of r, flux(r) (1/s) <= p(x, t) / (-s(r, x) * dt)
        # where t = self.time, p(x, t) = population of x at time t,
        # dt = self.time_step, and s(r, x) = stoichiometry of x in r
        for reaction in self.exchange_rxns:
            # to avoid over-constraining them, do not further bound exchange reactions that
            # participate in constraints in `_multi_reaction_constraints`
            if reaction in self._constrained_exchange_rxns:
                continue

            max_flux_rxn = float('inf')
            for participant in reaction.participants:
                # 'participant.coefficient < 0' determines whether the participant is a reactant
                if participant.coefficient < 0:
                    species_id = participant.species.gen_id()
                    species_pop = self.local_species_population.read_one(self.time, species_id)
                    max_flux_species = species_pop / (-participant.coefficient * self.time_step)
                    max_flux_rxn = min(max_flux_rxn, max_flux_species)

            if max_flux_rxn != float('inf'):
                max_flux_rxn = bound_scale_factor * max_flux_rxn

                min_constr, max_constr = self._reaction_bounds[reaction.id]
                if max_constr is None:
                    max_constr = max_flux_rxn
                else:
                    max_constr = min(max_constr, max_flux_rxn)

                self._reaction_bounds[reaction.id] = (min_constr, max_constr)

    def bound_neg_species_pop_constraints(self):
        """ Update bounds in the negative species population constraints that span multiple reactions

        Update the bounds in each of the constraints in `self._multi_reaction_constraints` that
        prevent some species from having a negative species population in the next time step

        Call this before each run of the FBA solver.
        """
        # set bounds in multi-reaction constraints
        for constraint_id, constraint in self._multi_reaction_constraints.items():
            species_id = DfbaSubmodel.parse_neg_species_pop_constraint_id(constraint_id)
            species_pop = self.local_species_population.read_one(self.time, species_id)
            max_consumption_of_species = species_pop / self.time_step
            print('species_id, species_pop, max_consumption_of_species',
                  species_id, species_pop, max_consumption_of_species)
            constraint.upper_bound = max_consumption_of_species
            print(f"appending '{constraint_id}", constraint)
            # todo: get the multi-reaction constraints to actually work
            # self._conv_model.constraints.append(constraint)

    def update_bounds(self):
        """ Update the minimum and maximum bounds of `conv_opt.Variable` based on
            the values in `self._reaction_bounds`
        """
        for rxn_id, (min_constr, max_constr) in self._reaction_bounds.items():
            self._conv_variables[rxn_id].lower_bound = min_constr
            self._conv_variables[rxn_id].upper_bound = max_constr
        self.bound_neg_species_pop_constraints()

    def compute_population_change_rates(self):
        """ Compute the rate of change of the populations of species used by this DFBA
        """
        coef_scale_factor = self.dfba_solver_options['dfba_coef_scale_factor']
        # Calculate the adjustment for each species as sum over reactions of reaction flux * stoichiometry
        self.current_species_populations()
        population_change_rate = {i:0 for i in [j.species.id for j in self._dfba_obj_species] +
            [k.participants[0].species.id for k in self.exchange_rxns]}
        ## Compute for objective species
        for obj_species in self._dfba_obj_species:
            population_change_rate[obj_species.species.id] += \
                self.reaction_fluxes[obj_species.dfba_obj_reaction.id] * -obj_species.value
        ## Compute for exchange species
        for exchange_rxn in self.exchange_rxns:
            population_change_rate[exchange_rxn.participants[0].species.id] += \
                self.reaction_fluxes[exchange_rxn.id] * -exchange_rxn.participants[0].coefficient

        for idx, species_id in enumerate(self.species_ids):
            if species_id in population_change_rate:
                self.adjustments[species_id] = population_change_rate[species_id]
            else:
                self.adjustments[species_id] = 0.

    def run_fba_solver(self):
        """ Run the FBA solver for one time step

        Raises:
            :obj:`DynamicMultiAlgorithmError`: if no optimal solution is found
        """
        bound_scale_factor = self.dfba_solver_options['dfba_bound_scale_factor']
        coef_scale_factor = self.dfba_solver_options['dfba_coef_scale_factor']
        self.determine_bounds()
        self.update_bounds()

        end_time = self.time + self.time_step
        result = self._conv_model.solve()
        if result.status_code != conv_opt.StatusCode(0):
            raise DynamicMultialgorithmError(self.time, f"DfbaSubmodel {self.id}: "
                                                        f"No optimal solution found: "
                                                        f"'{result.status_message}' "
                                                        f"for time step [{self.time}, {end_time}]")
        self._optimal_obj_func_value = result.value / bound_scale_factor * coef_scale_factor
        for rxn_variable in self._conv_model.variables:
            if rxn_variable.name not in self._dfba_obj_rxn_ids:
                self.reaction_fluxes[rxn_variable.name] = rxn_variable.primal / bound_scale_factor
            else:
                self.reaction_fluxes[rxn_variable.name] = rxn_variable.primal / bound_scale_factor \
                    * coef_scale_factor

        # Compute the population change rates
        self.compute_population_change_rates()

        ### store results in local_species_population ###
        self.local_species_population.adjust_continuously(self.time, self.id, self.adjustments,
                                                          time_step=self.time_step)

        # flush expressions that depend on species and reactions modeled by this dFBA submodel from cache
        self.dynamic_model.continuous_submodel_flush_after_populations_change(self.id)

    ### handle DES events ###
    def handle_RunFba_msg(self, event):
        """ Handle an event containing a RunFba message

        Args:
            event (:obj:`Event`): a simulation event
        """
        self.run_fba_solver()
        self.schedule_next_periodic_analysis()