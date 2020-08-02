""" A submodel that uses Gibson and Bruck's Next Reaction Method (NRM) to model a set of reactions

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2020-04-11
:Copyright: 2020, Karr Lab
:License: MIT
"""

from pqdict import PQDict
from scipy.constants import Avogadro
import collections
import math
import numpy as np
import sys

from de_sim.event import Event
from de_sim.simulation_engine import SimulationEngine
from de_sim.simulation_object import ApplicationSimulationObject, SimulationObject
from wc_sim import message_types
from wc_sim.config import core as config_core_multialgorithm
from wc_sim.dynamic_components import DynamicRateLaw
from wc_sim.multialgorithm_errors import MultialgorithmError, DynamicFrozenSimulationError
from wc_sim.submodels.dynamic_submodel import DynamicSubmodel
from wc_utils.util.rand import RandomStateManager

config_multialgorithm = config_core_multialgorithm.get_config()['wc_sim']['multialgorithm']


class NrmSubmodel(DynamicSubmodel):
    """ Use the Next Reaction Method to predict the dynamics of chemical species in a container

    Attributes:
        dependencies (:obj:`list` of :obj:`tuple`): dependencies that rate laws have on executed
            reactions; entry i provides the indices of reactions whose rate laws depend on reaction i
        execution_time_priority_queue (:obj:`PQDict`): NRM indexed priority queue of reactions, with
            the earliest scheduled reaction at the front of the queue
        propensities (:obj:`np.ndarray`): the most recently calculated propensities of the reactions
            modeled by this NRM submodel
        random_state (:obj:`np.random.RandomState`): the random state that is shared across the
            simulation, which enables reproducible checkpoint and restore of a simulation
    """

    # message types sent by NrmSubmodel
    SENT_MESSAGE_TYPES = [message_types.ExecuteAndScheduleNrmReaction]

    # register the message types sent
    messages_sent = SENT_MESSAGE_TYPES

    # at any time instant, process messages in this order
    MESSAGE_TYPES_BY_PRIORITY = [message_types.ExecuteAndScheduleNrmReaction]

    event_handlers = [(message_types.ExecuteAndScheduleNrmReaction, 'handle_ExecuteAndScheduleNrmReaction_msg')]

    def __init__(self, id, dynamic_model, reactions, species, dynamic_compartments,
                 local_species_population):
        """ Initialize an NRM submodel object

        Args:
            id (:obj:`str`): unique id of this dynamic NRM submodel
            dynamic_model (:obj:`DynamicModel`): the aggregate state of a simulation
            reactions (:obj:`list` of :obj:`Reaction`): the reactions modeled by this NRM submodel
            species (:obj:`list` of :obj:`Species`): the species that participate in the reactions modeled
                by this NRM submodel, with their initial concentrations
            dynamic_compartments (:obj:`dict`): :obj:`DynamicCompartment`\ s, keyed by id, that contain
                species which participate in reactions that this NRM submodel models, including
                adjacent compartments used by its transfer reactions
            local_species_population (:obj:`LocalSpeciesPopulation`): the store that maintains this
                NRM submodel's species population

        Raises:
            :obj:`MultialgorithmError`: if the initial NRM wait exponential moving average is not positive
        """
        super().__init__(id, dynamic_model, reactions, species, dynamic_compartments,
                         local_species_population)
        self.random_state = RandomStateManager.instance()

    def prepare(self):
        self.dependencies = self.determine_dependencies()
        self.propensities = self.initialize_propensities()
        self.initialize_execution_time_priorities()

    def determine_dependencies(self):
        """ Determine the dependencies that rate laws have on executed reactions

        In a multi-algorithmic simulation, two types of dependencies arise when NRM considers which
        rate laws to update after a reaction executes:
        1) Dependencies between NRM reactions and rate laws for NRM reactions. Rate laws that depend
        on species whose populations are updated by a reaction, either directly or indirectly through
        expressions that use the species, must be evaluated after the reaction executes.
        2) Rate laws that depend on, again either directly or indirectly, species whose populations can
        be updated by reactions or continuous changes in other submodels, must always be evaluated after any
        NRM reaction executes.
        This method combines both types of dependencies into a single map from executed reaction to
        dependent rate laws.
        It is executed only once, at initialization, so its complexity is fine.

        Returns:
            :obj:`list` of :obj:`tuple`: entry i provides the indices of reactions whose
                rate laws depend on the execution of reaction i, not including reaction i itself
        """

        # Rate laws and reactions used by this NRM model
        ids_of_rate_laws_used_by_self = {rxn.rate_laws[0].id for rxn in self.reactions}
        ids_of_rxns_used_by_self = {rxn.id for rxn in self.reactions}
        rxns_used_by_self = set(self.reactions)

        # Rate laws that depend on species that can be updated by reactions or continuous changes in other submodels
        ids_of_rls_depend_on_other_submodels = set()
        for rxn, dependent_exprs in self.dynamic_model.rxn_expression_dependencies.items():
            if rxn not in rxns_used_by_self:
                for dependent_expr in dependent_exprs:
                    if isinstance(dependent_expr, DynamicRateLaw):
                        ids_of_rls_depend_on_other_submodels.add(dependent_expr.id)

        # Dependencies of rate laws on reactions in this NRM submodel
        rate_laws_depend_on_rxns_as_ids = {rxn_id: set() for rxn_id in ids_of_rxns_used_by_self}
        for rxn, dependent_exprs in self.dynamic_model.rxn_expression_dependencies.items():
            if rxn in rxns_used_by_self:
                for dependent_expr in dependent_exprs:
                    if isinstance(dependent_expr, DynamicRateLaw):
                        rate_laws_depend_on_rxns_as_ids[rxn.id].add(dependent_expr.id)

        # Ids of rate laws that must be updated when a reaction is executed by this NRM model
        for rxn_id, dependent_rate_law_ids in rate_laws_depend_on_rxns_as_ids.items():
            # Add rate laws that depend on species updated by other models
            dependent_rate_law_ids.update(ids_of_rls_depend_on_other_submodels)
            # Delete rate laws not used by this NRM model
            dependent_rate_law_ids.intersection_update(ids_of_rate_laws_used_by_self)

        # Convert ids into indices in self.reactions
        rxn_id_to_idx = {}
        rate_law_id_to_idx = {}
        for idx, rxn in enumerate(self.reactions):
            rxn_id_to_idx[rxn.id] = idx
            rate_law_id_to_idx[rxn.rate_laws[0].id] = idx
        # Dependencies of rate laws on reactions, identified by their indices in self.reactions
        rate_laws_depend_on_rxns_as_indices = collections.defaultdict(set)
        for rxn_id, dependent_rate_law_ids in rate_laws_depend_on_rxns_as_ids.items():
            rate_laws_depend_on_rxns_as_indices[rxn_id_to_idx[rxn_id]] = \
                {rate_law_id_to_idx[rate_law_id] for rate_law_id in dependent_rate_law_ids}

        # Convert dependencies into more compact and faster list of tuples
        rate_laws_depend_on_rxns_as_indices_in_seqs = [tuple()] * len(self.reactions)
        for antecedent_rxn, dependent_rate_laws in rate_laws_depend_on_rxns_as_indices.items():
            # remove antecedent_rxn from dependent_rate_laws to simplify schedule_next_reaction,
            # which handles antecedent_rxn specially
            try:
                dependent_rate_laws.remove(antecedent_rxn)
            except KeyError:
                pass
            rate_laws_depend_on_rxns_as_indices_in_seqs[antecedent_rxn] = tuple(dependent_rate_laws)

        return rate_laws_depend_on_rxns_as_indices_in_seqs

    def initialize_propensities(self):
        """ Get the initial propensities of all reactions that have enough species counts to execute

        Propensities of reactions with inadequate species counts are set to 0.

        Returns:
            :obj:`np.ndarray`: the propensities of the reactions modeled by this NRM submodel which
                have adequate species counts to execute
        """
        propensities = self.calc_reaction_rates()

        # avoid reactions with inadequate species counts
        enabled_reactions = self.identify_enabled_reactions()
        propensities = enabled_reactions * propensities
        return propensities

    def initialize_execution_time_priorities(self):
        """ Initialize the NRM indexed priority queue of reactions
        """
        self.execution_time_priority_queue = PQDict()
        for reaction_idx, propensity in enumerate(self.propensities):
            if propensity == 0:
                self.execution_time_priority_queue[reaction_idx] = float('inf')
            else:
                tau = self.random_state.exponential(1.0/propensity) + self.time
                self.execution_time_priority_queue[reaction_idx] = tau

    def register_nrm_reaction(self, execution_time, reaction_index):
        """ Schedule a NRM reaction event with the simulator

        Args:
            execution_time (:obj:`float`): simulation time at which the reaction will execute
            reaction_index (:obj:`int`): index of the reaction to execute
        """
        self.send_event_absolute(execution_time, self,
                                 message_types.ExecuteAndScheduleNrmReaction(reaction_index))

    def schedule_reaction(self):
        """ Schedule the next reaction
        """
        next_rxn, next_time = self.execution_time_priority_queue.topitem()
        self.register_nrm_reaction(next_time, next_rxn)

    def init_before_run(self):
        """ Initialize this NRM submodel and schedule its first reaction

        This :obj:`ApplicationSimulationObject` method is called when a :obj:`SimulationEngine` is
        initialized.
        """
        self.schedule_reaction()

    def schedule_next_reaction(self, reaction_index):
        """ Schedule the next reaction after a reaction executes

        Args:
            reaction_index (:obj:`int`): index of the reaction that just executed
        """
        # reschedule each reaction whose rate law depends on the execution of reaction reaction_index
        # or depends on species whose populations may have been changed by other submodels
        for reaction_to_reschedule in self.dependencies[reaction_index]:

            # 1. compute new propensity
            propensity_new = self.calc_reaction_rate(self.reactions[reaction_to_reschedule])
            if propensity_new == 0:
                tau_new = float('inf')

            else:
                # 2. compute new tau from old tau
                tau_old = self.execution_time_priority_queue[reaction_to_reschedule]
                propensity_old = self.propensities[reaction_to_reschedule]
                if propensity_old == 0:
                    # If propensity_old == 0 then tau_old is infinity, and these cannot be used in
                    #   (propensity_old/propensity_new) * (tau_old - self.time) + self.time
                    # In this case a new random exponential must be drawn.
                    # This does not increase the number of random draws, because they're not
                    # drawn when the propensity is 0.
                    tau_new = self.random_state.exponential(1.0/propensity_new) + self.time

                else:
                    tau_new = (propensity_old/propensity_new) * (tau_old - self.time) + self.time

            self.propensities[reaction_to_reschedule] = propensity_new

            # 3. update the reaction's order in the indexed priority queue
            self.execution_time_priority_queue[reaction_to_reschedule] = tau_new

        # reschedule the reaction that was executed
        # 1. compute new propensity
        propensity_new = self.calc_reaction_rate(self.reactions[reaction_index])
        self.propensities[reaction_index] = propensity_new

        # 2. compute new tau
        if propensity_new == 0:
            tau_new = float('inf')
        else:
            tau_new = self.random_state.exponential(1.0/propensity_new) + self.time

        # 3. update the reaction's order in the indexed priority queue
        self.execution_time_priority_queue[reaction_index] = tau_new

        # schedule the next reaction
        self.schedule_reaction()

    def execute_nrm_reaction(self, reaction_index):
        """ Execute a reaction now

        Args:
            reaction_index (:obj:`int`): index of the reaction to execute
        """
        self.execute_reaction(self.reactions[reaction_index])

    def handle_ExecuteAndScheduleNrmReaction_msg(self, event):
        """ Handle an event containing an :obj:`ExecuteAndScheduleNrmReaction` message

        Args:
            event (:obj:`Event`): a simulation event
        """
        # execute the reaction
        reaction_index = event.message.reaction_index
        self.execute_nrm_reaction(reaction_index)

        # schedule the next reaction
        self.schedule_next_reaction(reaction_index)
