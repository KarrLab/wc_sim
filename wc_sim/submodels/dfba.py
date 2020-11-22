""" A submodel that uses Dynamic Flux Balance Analysis (dFBA) to model a set of reactions

:Author: Yin Hoon Chew <yinhoon.chew@mssm.edu>
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2020-07-29
:Copyright: 2016-2020, Karr Lab
:License: MIT
"""

import collections
import conv_opt
import copy
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
        DFBA_BOUND_SCALE_FACTOR (:obj:`float`): scaling factor for the bounds on reactions and
            constraints that avoid negative species populations; the default value is 1.
        DFBA_COEF_SCALE_FACTOR (:obj:`float`): scaling factor for the coefficients in dFBA objectives;
            the default value is 1.
        SOLVER (:obj:`str`): name of the selected solver in conv_opt, the default value is 'cplex'
        PRESOLVE (:obj:`str`): presolve mode in `conv_opt` ('auto', 'on', 'off'), the default
            value is 'on'
        SOLVER_OPTIONS (:obj:`dict`): parameters for the solver; default values are provided for
            'cplex'
        OPTIMIZATION_TYPE (:obj:`str`): direction of optimization ('maximize', 'max',
            'minimize', 'min'); the default value is 'maximize'
        FLUX_BOUNDS_VOLUMETRIC_COMPARTMENT_ID (:obj:`str`): id of the compartment to which the
            measured flux bounds are normalized, the default is the whole-cell
        NEG_POP_CONSTRAINTS (:obj:`boolean`): whether the constraints that
            prevent negative species over the next time-step should be used; defaults to :obj:`True`
        dfba_solver_options (:obj:`dict` of :obj:`str`: :obj:`any`): options for solving dFBA submodel
        reaction_fluxes (:obj:`dict` of :obj:`str`: :obj:`float`): reaction fluxes data structure,
            which is pre-allocated
        dfba_obj_expr (:obj:`ParsedExpression`): an analyzed and validated dFBA objective expression
        exchange_rxns (:obj:`set` of :obj:`wc_lang.Reactions`): set of exchange and demand reactions
        _multi_reaction_constraints (:obj:`dict` of :obj:`str`: :obj:`conv_opt.Constraint`): a map from
            constraint id to constraints that avoid negative species populations
            in `self._conv_model.constraints`
        _constrained_exchange_rxns (:obj:`set` of :obj:`wc_lang.Reactions`): exchange and demand reactions
            constrained by `_multi_reaction_constraints`
        _conv_model (:obj:`conv_opt.Model`): linear programming model in `conv_opt` format
        _conv_variables (:obj:`dict` of :obj:`str`: :obj:`conv_opt.Variable`): a dictionary mapping
            reaction IDs to their associated `conv_opt.Variable` objects
        _conv_metabolite_matrices (:obj:`dict` of :obj:`str`: :obj:`list`): a dictionary mapping metabolite
            species IDs to lists of :obj:`conv_opt.LinearTerm` objects; each :obj:`conv_opt.LinearTerm`
            associates a reaction that the species participates in with the species' stoichiometry
            in the reaction
        _dfba_obj_rxn_ids (:obj:`list` of :obj:`str`): a list of IDs of dFBA objective reactions
        _dfba_obj_species (:obj:`list` of :obj:`wc_lang.DfbaObjSpecies:`): all species in
            :obj:`DfbaObjReaction`\ s used by `dfba_obj_expr`
        _reaction_bounds (:obj:`dict` of :obj:`str`: :obj:`tuple`): a dictionary that maps reaction IDs
            to tuples of scaled minimum and maximum bounds
        _optimal_obj_func_value (:obj:`float`): the value of objective function returned by the solver
    """
    # default options
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
    NEG_POP_CONSTRAINTS = False

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
                or if the exchange reactions are not all of the form “s ->”,
                or if the provided 'dfba_bound_scale_factor' in options does not have a positive value,
                or if the provided 'dfba_coef_scale_factor' in options does not have a positive value,
                or if the provided 'solver' in options is not a valid value,
                or if the provided 'presolve' in options is not a valid value,
                or if the 'solver' value provided in the 'solver_options' in options is not the same
                as the name of the selected `conv_opt.Solver`,
                or if the provided 'flux_bounds_volumetric_compartment_id' is not a valid
                compartment ID in the model
        """
        super().__init__(id, dynamic_model, reactions, species, dynamic_compartments,
                         local_species_population, dfba_time_step, options)

        self.dfba_solver_options = {
            'dfba_bound_scale_factor': self.DFBA_BOUND_SCALE_FACTOR,
            'dfba_coef_scale_factor': self.DFBA_COEF_SCALE_FACTOR,
            'solver': self.SOLVER,
            'presolve': self.PRESOLVE,
            'solver_options': self.SOLVER_OPTIONS,
            'optimization_type': self.OPTIMIZATION_TYPE,
            'flux_bounds_volumetric_compartment_id': self.FLUX_BOUNDS_VOLUMETRIC_COMPARTMENT_ID,
            'negative_pop_constraints': self.NEG_POP_CONSTRAINTS
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
            if 'negative_pop_constraints' in options:
                self.dfba_solver_options['negative_pop_constraints'] = options['negative_pop_constraints']

        # determine set of exchange reactions, which must have the form “s ->”
        errors = []
        self.exchange_rxns = set()
        for rxn in self.reactions:
            if len(rxn.participants) == 1:
                part = rxn.participants[0]
                if part.coefficient < 0:
                    self.exchange_rxns.add(rxn)
                else:
                    errors.append(rxn.id)
        if errors:
            rxns = ', '.join(errors)
            raise MultialgorithmError(f"exchange reaction(s) don't have the form 's ->': {rxns}")

        # get the dfba objective's expression
        dfba_objective_id = f'dfba-obj-{id}'
        if dfba_objective_id not in dynamic_model.dynamic_dfba_objectives: # pragma: no cover
            raise MultialgorithmError(f"DfbaSubmodel {self.id}: cannot find dynamic_dfba_objective "
                                      f"{dfba_objective_id}")
        self.dfba_obj_expr = dynamic_model.dynamic_dfba_objectives[dfba_objective_id].wc_lang_expression

        # ensure that the dfba objective doesn't contain exchange rxns
        errors = []
        for rxn_id_2_rxn_map in self.dfba_obj_expr.related_objects.values():
            for rxn in rxn_id_2_rxn_map.values():
                if rxn in self.exchange_rxns:
                    errors.append(rxn.id)
        if errors:
            rxns = ', '.join(errors)
            raise MultialgorithmError(f"the dfba objective '{dfba_objective_id}' "
                                      f"uses exchange reactions: {rxns}")

        # log initialization data
        self.log_with_time("init: id: {}".format(id))
        self.log_with_time("init: time_step: {}".format(str(dfba_time_step)))

        ### dFBA specific code ###
        self.set_up_dfba_submodel()
        self.set_up_optimizations()

    def set_up_dfba_submodel(self):
        """ Set up a dFBA submodel, by converting to a linear programming matrix

        Raises:
            :obj:`MultiAlgorithmError`: if the ids in DfbaObjReactions and Reactions intersect
        """
        self.set_up_continuous_time_submodel()

        ### dFBA specific code ###

        # raise an error if the ids in DfbaObjReactions and Reactions intersect
        reaction_ids = set([rxn.id for rxn in self.reactions])
        dfba_obj_reaction_ids = set()
        for rxn_cls in self.dfba_obj_expr.related_objects:
            for rxn in self.dfba_obj_expr.related_objects[rxn_cls].values():
                if isinstance(rxn, wc_lang.DfbaObjReaction):
                    dfba_obj_reaction_ids.add(rxn.id)
        if reaction_ids & dfba_obj_reaction_ids:
            raise MultialgorithmError(f"in model {self.dynamic_model.id} the ids in DfbaObjReactions "
                                      f"and Reactions intersect: {reaction_ids & dfba_obj_reaction_ids}")
        # TODO later: support colliding ids by creating unique ids prefixed by class

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

        self._dfba_obj_rxn_ids = []
        self._dfba_obj_species = []
        for rxn_id_2_rxn_map in self.dfba_obj_expr.related_objects.values():
            for rxn_id, rxn in rxn_id_2_rxn_map.items():
                if rxn_id not in self._conv_variables:
                    self._dfba_obj_rxn_ids.append(rxn_id)
                    self._conv_variables[rxn.id] = conv_opt.Variable(
                        name=rxn.id, type=conv_opt.VariableType.continuous, lower_bound=0)
                    self._conv_model.variables.append(self._conv_variables[rxn.id])
                    for part in rxn.dfba_obj_species:
                        self._dfba_obj_species.append(part)
                        self._conv_metabolite_matrices[part.species.id].append(
                            conv_opt.LinearTerm(self._conv_variables[rxn.id], part.value))

        # Set up the objective function
        dfba_obj_expr_objs = self.dfba_obj_expr.lin_coeffs
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

    def get_conv_model(self):
        """ Get the `conv_opt` model

        Returns:
            :obj:`conv_opt.Model`: the linear programming model in `conv_opt` format
        """
        return self._conv_model

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

        # pragma: no cover false branch in this elif; ignore missing branch coverage report
        elif isinstance(reaction, wc_lang.DfbaObjReaction):
            for part in reaction.dfba_obj_species:
                species_net_coefficients[part.species] += part.value

        species_to_rm = [s for s in species_net_coefficients if species_net_coefficients[s] == 0.]
        for species in species_to_rm:
            del species_net_coefficients[species]
        return species_net_coefficients

    NEG_POP_CONSTRAINT_PREFIX = 'neg_pop_constr'
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

    # TODO: in species_id_with_brkts raise error if species_id doesn't have codes or has brackets
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
        return DfbaSubmodel.NEG_POP_CONSTRAINT_PREFIX + DfbaSubmodel.NEG_POP_CONSTRAINT_SEP + species_id

    @staticmethod
    def parse_neg_species_pop_constraint_id(neg_species_pop_constraint_id):
        """ Parse a negative species population constraint id

        Args:
            neg_species_pop_constraint_id (:obj:`str`): a negative species population constraint id

        Returns:
            :obj:`str`: id of species being constrained
        """
        loc = len(DfbaSubmodel.NEG_POP_CONSTRAINT_PREFIX) + len(DfbaSubmodel.NEG_POP_CONSTRAINT_SEP)
        return neg_species_pop_constraint_id[loc:]

    def initialize_neg_species_pop_constraints(self):
        """ Make constraints that span multiple reactions

        A separate constraint is made for each species that participates in more than
        one exchange and/or dfba objective reactions. These constraints prevent the species population
        from declining so quickly that it becomes negative in the next time step.
        Also initializes `self._constrained_exchange_rxns`.
        Call this when a dFBA submodel is initialized.

        Returns:
            :obj:`dict` of :obj:`str`: :obj:`conv_opt.Constraint`: a map from constraint id to
            constraints stored in `self._conv_model.constraints`
        """
        # TODO: AVOID NEG. POPS.
        # either check that all reactions are irreversible or handle reversible reactions

        self._constrained_exchange_rxns = set()
        if not self.dfba_solver_options['negative_pop_constraints']:
            return {}

        # map from species to reactions that use the species
        reactions_using_species = collections.defaultdict(set)

        # add species in objective reactions
        for rxn_cls in self.dfba_obj_expr.related_objects:
            for rxn in self.dfba_obj_expr.related_objects[rxn_cls].values():
                # ignore wc_lang.Reactions, which cannot have a net species flux
                if isinstance(rxn, wc_lang.DfbaObjReaction):
                    for species in self._get_species_and_stoichiometry(rxn):
                        reactions_using_species[species].add(rxn)

        # add species in exchange reactions
        for rxn in self.exchange_rxns:
            for species in self._get_species_and_stoichiometry(rxn):
                reactions_using_species[species].add(rxn)

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
                                scaled_net_consumption_coef = net_consumption_coef
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
                    self._conv_model.constraints.append(constraint)
                    multi_reaction_constraints[constraint_id] = constraint

                    # record the constrained exchange reactions
                    self._constrained_exchange_rxns |= exchange_rxns_in_constr_expr

        return multi_reaction_constraints

    def set_up_optimizations(self):
        """ To improve performance, pre-compute and pre-allocate some data structures """
        self.set_up_continuous_time_optimizations()

        # pre-allocate dict of reaction fluxes
        self.reaction_fluxes = {rxn.id: None for rxn in self.reactions}

        # initialize adjustments, the dict that will hold the species population change rates
        self.adjustments = {}
        for obj_species in self._dfba_obj_species:
            self.adjustments[obj_species.species.id] = 0.
        for exchange_rxn in self.exchange_rxns:
            self.adjustments[exchange_rxn.participants[0].species.id] = 0.

        for met_id, expression in self._conv_metabolite_matrices.items():
            self._conv_model.constraints.append(conv_opt.Constraint(expression, name=met_id,
                upper_bound=0.0, lower_bound=0.0))

    def determine_bounds(self):
        """ Determine the minimum and maximum bounds for each reaction

        Two actions:

        1. Bounds provided by the model and written to `self._reaction_bounds`

        2. The bounds in exchange and demand reactions that do not participate in a negative
        species population constraint are further bounded so they do not let a reaction's
        species population decline so quickly that it becomes negative in the next time step.
        """
        # Determine all the reaction bounds
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
                    # TODO (APG): if reversible rxns are split, could call self.calc_reaction_rate(reaction)
                    rate = self.dynamic_model.dynamic_rate_laws[for_ratelaw.id].eval(self.time)
                    max_constr = rate
                else:
                    max_constr = None

                rev_ratelaw = reaction.rate_laws.get_one(direction=wc_lang.RateLawDirection.backward)
                if rev_ratelaw:
                    rate = self.dynamic_model.dynamic_rate_laws[rev_ratelaw.id].eval(self.time)
                    min_constr = -rate
                elif reaction.reversible:
                    min_constr = None
                else:
                    min_constr = 0.

            # Set the bounds of exchange and demand reactions
            elif reaction.flux_bounds:

                rxn_bounds = reaction.flux_bounds
                if rxn_bounds.min:
                    if math.isnan(rxn_bounds.min):
                        min_constr = None
                    else:
                        min_constr = rxn_bounds.min * flux_comp_volume * scipy.constants.Avogadro
                else:
                    min_constr = rxn_bounds.min
                if rxn_bounds.max:
                    if math.isnan(rxn_bounds.max):
                        max_constr = None
                    else:
                        max_constr = rxn_bounds.max * flux_comp_volume * scipy.constants.Avogadro
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
                max_flux_rxn = max_flux_rxn

                min_constr, max_constr = self._reaction_bounds[reaction.id]
                if max_constr is None:
                    max_constr = max_flux_rxn
                else:
                    max_constr = min(max_constr, max_flux_rxn)

                self._reaction_bounds[reaction.id] = (min_constr, max_constr)

    def bound_neg_species_pop_constraints(self):
        """ Update bounds in the negative species population constraints that span multiple reactions

        Update the bounds in each of the constraints in `self._multi_reaction_constraints` that
        prevent some species from having a negative species population in the next time step.

        Call this before each run of the FBA solver.
        """
        # set bounds in multi-reaction constraints
        for constraint_id, constraint in self._multi_reaction_constraints.items():
            species_id = DfbaSubmodel.parse_neg_species_pop_constraint_id(constraint_id)
            species_pop = self.local_species_population.read_one(self.time, species_id)
            max_allowed_consumption_of_species = species_pop / self.time_step
            constraint.upper_bound = max_allowed_consumption_of_species

    def update_bounds(self):
        """ Update the minimum and maximum bounds of `conv_opt.Variable` based on
            the values in `self._reaction_bounds`
        """
        for rxn_id, (min_constr, max_constr) in self._reaction_bounds.items():
            self._conv_variables[rxn_id].lower_bound = min_constr
            self._conv_variables[rxn_id].upper_bound = max_constr
        self.bound_neg_species_pop_constraints()

    def compute_population_change_rates(self):
        """ Compute the rate of change of the populations of species used by this dFBA

        Because FBA obtains a steady-state solution for reaction fluxes, only species that
        participate in unbalanced reactions at the edge of the network (exchange reactions and
        dFBA objective reactions) can experience non-zero rates of change.

        Updates the existing dict `self.adjustments`.
        """
        # Calculate the adjustment for each species as sum over reactions of reaction flux * stoichiometry
        self.current_species_populations()

        # Compute for objective species
        for obj_species in self._dfba_obj_species:
            self.adjustments[obj_species.species.id] += \
                self.reaction_fluxes[obj_species.dfba_obj_reaction.id] * obj_species.value

        # Compute for exchange species
        for exchange_rxn in self.exchange_rxns:
            self.adjustments[exchange_rxn.participants[0].species.id] += \
                self.reaction_fluxes[exchange_rxn.id] * exchange_rxn.participants[0].coefficient

    def scale_conv_opt_model(self, conv_opt_model, copy_model=True,
                             dfba_bound_scale_factor=None, dfba_coef_scale_factor=None):
        """ Apply scaling factors to a `conv_opt` model

        Scaling factors can be used to scale the size of bounds and objective function term
        coefficients to address numerical problems with the linear programming solver.
        They are elements of `dfba_solver_options`.
        The `dfba_bound_scale_factor` option scales the bounds on reactions and constraints
        that avoid negative species populations.
        The `dfba_coef_scale_factor` scales the coefficients in dFBA objectives.
        Scaling is done by the this method.
        Symmetrically, the solution results are returned to the scale of the whole-cell model by
        inverting the consequences of these scaling factors.
        This is done by the `unscale_conv_opt_solution` method.

        Args:
            conv_opt_model (:obj:`conv_opt.Model`): a convex optimization model
            copy_model (:obj:`boolean`, optional): whether to copy the convex optimization model
                before scaling it; defaults to :obj:`True`
            dfba_bound_scale_factor (:obj:`float`, optional): factor used to scale the bounds on
                reactions and constraints that avoid negative species populations; if not supplied,
                is taken from `self.dfba_solver_options`
            dfba_coef_scale_factor (:obj:`float`, optional): factor used to scale the coefficients
                in dFBA objectives; if not supplied, is taken from `self.dfba_solver_options`

        Returns:
            :obj:`conv_opt.Model`: the scaled convex optimization model
        """
        if copy_model:
            conv_opt_model = copy.deepcopy(conv_opt_model)

        if dfba_bound_scale_factor is None:
            dfba_bound_scale_factor = self.dfba_solver_options['dfba_bound_scale_factor']
        if dfba_coef_scale_factor is None:
            dfba_coef_scale_factor = self.dfba_solver_options['dfba_coef_scale_factor']

        # scale bounds
        # skip non-numeric bounds, such as None
        for variable in conv_opt_model.variables:
            if isinstance(variable.lower_bound, (int, float)):
                variable.lower_bound *= dfba_bound_scale_factor
            if isinstance(variable.upper_bound, (int, float)):
                variable.upper_bound *= dfba_bound_scale_factor

        # scale bounds in constraints; bound values 0 are unchanged
        for constraint in conv_opt_model.constraints:
            if isinstance(constraint.lower_bound, (int, float)):
                constraint.lower_bound *= dfba_bound_scale_factor
            if isinstance(constraint.upper_bound, (int, float)):
                constraint.upper_bound *= dfba_bound_scale_factor

        # scale coefficient terms in conv_opt model objective
        for objective_term in conv_opt_model.objective_terms:
            objective_term.coefficient *= dfba_coef_scale_factor

        return conv_opt_model

    def unscale_conv_opt_solution(self, dfba_bound_scale_factor=None, dfba_coef_scale_factor=None):
        """ Remove scaling factors from a `conv_opt` model solution

        Args:
            dfba_bound_scale_factor (:obj:`float`, optional): factor used to scale reaction and
                constraint bounds; if not supplied, is taken from `self.dfba_solver_options`
            dfba_coef_scale_factor (:obj:`float`, optional): factor used to scale the coefficients
                in dFBA objectives; if not supplied, is taken from `self.dfba_solver_options`
        """
        if dfba_bound_scale_factor is None:
            dfba_bound_scale_factor = self.dfba_solver_options['dfba_bound_scale_factor']
        if dfba_coef_scale_factor is None:
            dfba_coef_scale_factor = self.dfba_solver_options['dfba_coef_scale_factor']

        print(f'dfba_bound_scale_factor: {dfba_bound_scale_factor}')
        print(f'dfba_coef_scale_factor: {dfba_coef_scale_factor}')
        print(f'self._optimal_obj_func_value: {self._optimal_obj_func_value}')
        self._optimal_obj_func_value *= (dfba_coef_scale_factor / dfba_bound_scale_factor)
        for rxn_variable in self._conv_model.variables:
            if rxn_variable.name not in self._dfba_obj_rxn_ids:
                self.reaction_fluxes[rxn_variable.name] /= dfba_bound_scale_factor
            else:
                self.reaction_fluxes[rxn_variable.name] *= (dfba_coef_scale_factor / dfba_bound_scale_factor)

    def save_fba_solution(self, conv_opt_model, conv_opt_solution):
        """ Assign a FBA solution to local variables

        Args:
            conv_opt_model (:obj:`conv_opt.Model`): the convex optimization model that was solved
            conv_opt_solution (:obj:`conv_opt.Result`): the model's solution
        """
        self._optimal_obj_func_value = conv_opt_solution.value
        for rxn_variable in conv_opt_model.variables:
            self.reaction_fluxes[rxn_variable.name] = rxn_variable.primal

    @staticmethod
    def show_conv_opt_model(conv_opt_model):
        """ Convert a `conv_opt` into a readable representation

        Args:
            conv_opt_model (:obj:`conv_opt.Model`): a convex optimization model

        Returns:
            :obj:`str`: a readable representation of `conv_opt_model`
        """
        import enum
        conv_opt_model_rows = ['']
        conv_opt_model_rows.append('--- conf_opt model ---')
        conv_opt_model_rows.append(f"name: '{conv_opt_model.name}'")


        class ObjToRow(object):
            def __init__(self, col_widths, headers, attrs):
                self.col_widths = col_widths
                self.headers = headers
                self.attrs = attrs

            def header_rows(self):
                rv = []
                for header_row in self.headers:
                    row = ''
                    for col_header, width in zip(header_row, self.col_widths):
                        row += f'{col_header:<{width}}'
                    rv.append(row)
                return rv

            def obj_as_row(self, obj):
                row = ''
                for attr, width in zip(self.attrs, self.col_widths):
                    value = getattr(obj, attr)
                    str_value = f'{str(value):<{width-1}} '
                    if isinstance(value, enum.Enum):
                        str_value = f'{value.name:<{width-1}}'
                    elif isinstance(value, float):
                        str_value = f'{value:>{width-1}.4g} '
                    elif isinstance(value, int):
                        str_value = f'{value:>{width-1}d} '
                    row += str_value
                return row

        # variables: skip 'primal', 'reduced_cost', which aren't used
        conv_opt_model_rows.append('')
        conv_opt_model_rows.append('--- variables ---')
        headers_1 = ('name', 'type', 'lower', 'upper')
        headers_2 = ('', '', 'bound', 'bound',)
        variable_to_row = ObjToRow((18, 18, 14, 14,),
                                    [headers_1, headers_2],
                                    ('name', 'type', 'lower_bound', 'upper_bound'))
        conv_opt_model_rows.extend(variable_to_row.header_rows())
        for variable in conv_opt_model.variables:
            conv_opt_model_rows.append(variable_to_row.obj_as_row(variable))

        # constraints: skip 'dual' which isn't used
        headers_1 = ('name', 'lower', 'upper')
        headers_2 = ('', 'bound', 'bound',)
        constraint_to_row = ObjToRow((22, 10, 10,),
                                     [headers_1, headers_2],
                                     ('name', 'lower_bound', 'upper_bound'))

        # I presume that the lower and upper bounds in constraint terms are ignored
        variable_term_to_row = ObjToRow((18, 18),
                                    None,
                                    ('name', 'type'))
        # linear terms include Variable as a field, so just print the coefficient directly
        conv_opt_model_rows.append('')
        conv_opt_model_rows.append('--- constraints ---')
        conv_opt_model_rows.extend(constraint_to_row.header_rows())
        for constraint in conv_opt_model.constraints:
            conv_opt_model_rows.append(constraint_to_row.obj_as_row(constraint))
            conv_opt_model_rows.append('--- terms ---')
            for linear_term in constraint.terms:
                row = f'{linear_term.coefficient:<8}'
                row += variable_term_to_row.obj_as_row(linear_term.variable)
                conv_opt_model_rows.append(row)
            conv_opt_model_rows.append('')

        conv_opt_model_rows.append(f'objective direction: {conv_opt_model.objective_direction.name}')

        conv_opt_model_rows.append('')
        conv_opt_model_rows.append('--- objective terms ---')
        for objective_term in conv_opt_model.objective_terms:
            row = f'{objective_term.coefficient:<8}'
            row += variable_term_to_row.obj_as_row(objective_term.variable)
            conv_opt_model_rows.append(row)

        return '\n'.join(conv_opt_model_rows)

    def run_fba_solver(self):
        """ Run the FBA solver for one time step

        Raises:
            :obj:`DynamicMultiAlgorithmError`: if no optimal solution is found
        """
        print('_conv_variables')
        print(set(list(self._conv_variables)))
        self.determine_bounds()
        self.update_bounds()
        # print('\n--- wc lang conv opt model ---')
        # print(self.show_conv_opt_model(self.get_conv_model()))

        # scale just before solving
        scaled_conv_opt_model = self.scale_conv_opt_model(self.get_conv_model())
        result = scaled_conv_opt_model.solve()
        end_time = self.time + self.time_step
        if result.status_code != conv_opt.StatusCode(0):
            raise DynamicMultialgorithmError(self.time, f"DfbaSubmodel {self.id}: "
                                                        f"No optimal solution found: "
                                                        f"'{result.status_message}' "
                                                        f"for time step [{self.time}, {end_time}]")
        # save and unscale the solution
        self.save_fba_solution(scaled_conv_opt_model, result)
        self.unscale_conv_opt_solution()

        print()
        print('--- solution ---')
        print('reaction fluxes')
        for rxn_id, flux in self.reaction_fluxes.items():
            print(f"{rxn_id:<20} {flux:>10.2g}")

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