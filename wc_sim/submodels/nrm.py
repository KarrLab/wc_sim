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
                 local_species_population, options=None):
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
            options (:obj:`dict`, optional): NRM submodel options

        Raises:
            :obj:`MultialgorithmError`: if the initial NRM wait exponential moving average is not positive
        """
        super().__init__(id, dynamic_model, reactions, species, dynamic_compartments,
                         local_species_population)
        self.random_state = RandomStateManager.instance()
        self.execution_time_priority_queue = PQDict()
        self.options = options
        # to enable testing of uninitialized instances auto_initialize controls initialization here
        # auto_initialize defaults to True
        auto_initialize = True
        if options is not None and 'auto_initialize' in options:
            auto_initialize = options['auto_initialize']
        if auto_initialize:
            self.initialize()

    def initialize(self):
        self.dependencies = self.determine_dependencies()
        self.propensities = self.initialize_propensities()
        self.initialize_execution_time_priorities()

    def determine_dependencies(self):
        """ Determine the dependencies that rate laws have on executed reactions

        Returns:
            :obj:`list` of :obj:`tuple`: entry i provides the indices of reactions whose
                rate laws depend on the execution of reaction i
        """
        # in a multi-algorithmic simulation, two types of dependencies arise when a reaction executes:
        # 1) ones used by NRM: rate laws that use species whose populations are updated by the reaction
        # 2) rate laws that use species whose populations might be updated by other submodels

        # dependencies[i] will contain the indices of rate laws that depend on reaction i
        dependencies = {i: set() for i in range(len(self.reactions))}

        # updated_species[i] will contain the ids of species whose populations are updated by reaction i
        updated_species = {i: set() for i in range(len(self.reactions))}

        # used_species[species_id] will contain the indices of rate laws (rxns) that use species with id species_id
        used_species = {species.gen_id(): set() for species in self.species}

        # initialize reaction -> species -> reaction dependency dictionaries
        for reaction_idx, rxn in enumerate(self.reactions):
            net_stoichiometric_coefficients = collections.defaultdict(float)
            for participant in rxn.participants:
                species_id = participant.species.gen_id()
                net_stoichiometric_coefficients[species_id] += participant.coefficient
            for species_id, net_stoich_coeff in net_stoichiometric_coefficients.items():
                if net_stoich_coeff < 0 or 0 < net_stoich_coeff:
                    updated_species[reaction_idx].add(species_id)
            rate_law = rxn.rate_laws[0]
            for species in rate_law.expression.species:
                species_id = species.gen_id()
                used_species[species_id].add(reaction_idx)

        # Sequential case, with one instance each of an SSA, ODE, dFBA submodels:
        # NRM must recompute all rate laws that depend on species in a continuous submodels
        # TODO: possible optimization: compute #2 dynamically, based on the time of the last update of the species
        # get shared species from self.local_species_population
        continuously_modeled_species = self.local_species_population.get_continuous_species()
        # print('continuously_modeled_species:', continuously_modeled_species)
        for updated_species_set in updated_species.values():
            updated_species_set |= continuously_modeled_species

        # Parallel case (to be addressed later), with multiple instances each of an SSA, ODE and dFBA submodels:
        # NRM must recompute all rate laws that depend on species shared with any other submodel

        # compute reaction to rate laws dependencies
        for reaction_idx, rxn in enumerate(self.reactions):
            for species_id in updated_species[reaction_idx]:
                for rate_law_idx in used_species[species_id]:
                    dependencies[reaction_idx].add(rate_law_idx)

        # convert dependencies into more compact and faster list of tuples
        dependencies_list = [None] * len(self.reactions)
        for antecedent_rxn, dependent_rxns in dependencies.items():
            dependencies_list[antecedent_rxn] = tuple(dependent_rxns)

        # print('dependencies_list:', dependencies_list)
        return dependencies_list

    def initialize_propensities(self):
        """ Get the initial propensities of all reactions that have enough species counts to execute

        Propensities of reactions with inadequate species counts are set to 0.

        Returns:
            :obj:`np.ndarray`: the propensities of the reactions modeled by this NRM submodel which
                have adequate species counts to execute
        """
        # TODO: raise exception if any propensities are < 0
        propensities = self.calc_reaction_rates()

        # avoid reactions with inadequate species counts
        enabled_reactions = self.identify_enabled_reactions()
        propensities = enabled_reactions * propensities
        return propensities

    def initialize_execution_time_priorities(self):
        """ Initialize the NRM indexed priority queue of reactions
        """
        taus = self.random_state.exponential(1.0/self.propensities)
        for reaction_idx, tau in enumerate(taus):
            self.execution_time_priority_queue[reaction_idx] = self.time + tau

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

    def send_initial_events(self):
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
            if reaction_to_reschedule != reaction_index:

                # 1. compute new propensity
                propensity_new = self.calc_reaction_rate(self.reactions[reaction_to_reschedule])

                # 2. compute new tau from old tau
                tau_old = self.execution_time_priority_queue[reaction_to_reschedule]
                propensity_old = self.propensities[reaction_to_reschedule]
                tau_new = (propensity_old/propensity_new) * (tau_old - self.time) + self.time
                self.propensities[reaction_to_reschedule] = propensity_new

                # 3. update the reaction's order in the indexed priority queue
                self.execution_time_priority_queue[reaction_to_reschedule] = tau_new

        # reschedule reaction reaction_index
        # 1. compute new propensity
        propensity_new = self.calc_reaction_rate(self.reactions[reaction_index])

        # 2. compute new tau
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
