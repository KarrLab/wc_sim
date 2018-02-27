""" A simplistic, configurable submodel for testing submodels and the simulator

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016_12_06
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

from wc_sim.core.event import Event
from wc_sim.core.simulation_object import SimulationObject
from wc_sim.multialgorithm import message_types
from wc_sim.multialgorithm.message_types import ALL_MESSAGE_TYPES
from wc_sim.multialgorithm.submodels.dynamic_submodel import DynamicSubmodel


class SkeletonSubmodel(DynamicSubmodel):
    """ Init a SkeletonSubmodel

    A SkeletonSubmodel is a simple submodel with externally controlled behavior.

    Attributes:
        behavior (:obj:`dict`): a dictionary of behavioral parameters
        next_reaction (:obj:`int`): the index of the next reaction to execute
    """
    REACTION_TO_EXECUTE = 'REACTION_TO_EXECUTE'
    INTER_REACTION_TIME = 'INTER_REACTION_TIME'
    def __init__(self, id, reactions, species, parameters, dynamic_compartments,
        local_species_population, behavior):
        self.behavior = behavior
        self.next_reaction = 0
        super().__init__(id, reactions, species, parameters, dynamic_compartments,
            local_species_population)

    # The next three methods override DynamicSubmodel methods.
    def send_initial_events(self):
        self.schedule_the_next_reaction()

    # register the event handler for each type of message received
    event_handlers =[
            (message_types.GivePopulation, 'handle_GivePopulation_event'),
            (message_types.ExecuteSsaReaction, 'handle_ExecuteSsaReaction_event')]

    # register the message types sent
    messages_sent = ALL_MESSAGE_TYPES

    def schedule_the_next_reaction(self):
        """ Schedule the next reaction for this `SkeletonSubmodel`

        If a REACTION_TO_EXECUTE is defined then a `SkeletonSubmodel` executes that reaction;
        otherwise, it simply executes reactions in round-robin order.
        """
        dt = self.behavior[SkeletonSubmodel.INTER_REACTION_TIME]
        if SkeletonSubmodel.REACTION_TO_EXECUTE in self.behavior:
            self.send_event(dt, self, message_types.ExecuteSsaReaction(
                self.behavior[SkeletonSubmodel.REACTION_TO_EXECUTE]))
        else:
            self.send_event(dt, self, message_types.ExecuteSsaReaction(self.next_reaction))
            self.next_reaction += 1
            self.next_reaction %= len(self.reactions)

    def handle_ExecuteSsaReaction_event(self, event):
        """ Handle a simulation event that contains an ExecuteSsaReaction message

        Args:
            event (:obj:`Event`): an :obj:`Event` to execute
        """
        # reaction_index is the reaction to execute
        reaction_index = event.message.reaction_index
        # Execute an enabled reaction
        if self.enabled_reaction(self.reactions[reaction_index]):
            self.execute_reaction(self.reactions[reaction_index])
        self.schedule_the_next_reaction()

    # TODO(Arthur): cover after MVP wc_sim done
    def handle_GivePopulation_event(self, event):   # pragma: no cover
        """ Handle a simulation event that contains a GivePopulation message

        Args:
            event (:obj:`wc_sim.core.Event`): an Event to process
        """
        # populations is a GivePopulation_body instance
        populations = event.message
        self.access_species_population.species_population_cache.cache_population(
            self.time, populations)
