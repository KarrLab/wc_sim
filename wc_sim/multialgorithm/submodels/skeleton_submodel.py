""" A simplistic, configurable submodel for testing submodels and the simulator

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016_12_06
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""
from builtins import super

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
    def __init__(self, id, reactions, species, parameters, dynamic_compartments,
        local_species_population, behavior):
        self.behavior = behavior
        self.next_reaction = 0
        super().__init__(id, reactions, species, parameters, dynamic_compartments,
            local_species_population)

    # these first three methods override DynamicSubmodel methods
    def send_initial_events(self):
        # print("{} executing handle_ExecuteSsaReaction_event".format(self.id))
        pass

    @classmethod
    def register_subclass_handlers(cls):
        SimulationObject.register_handlers(cls, [
            # TODO(Arthur): cover after MVP wc_sim done
            # (message_types.GivePopulation, cls.handle_GivePopulation_event),
            (message_types.ExecuteSsaReaction, cls.handle_ExecuteSsaReaction_event),
        ])

    @classmethod
    def register_subclass_sent_messages(cls):
        SimulationObject.register_sent_messages(cls, ALL_MESSAGE_TYPES)

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

    def schedule_the_next_reaction(self):
        # Schedule the next reaction
        dt = self.behavior['INTER_REACTION_TIME']
        self.next_reaction += 1
        self.next_reaction %= len(self.reactions)
        self.send_event(dt, self, message_types.ExecuteSsaReaction(self.next_reaction))

    def handle_ExecuteSsaReaction_event(self, event):
        """ Handle a simulation event that contains an ExecuteSsaReaction message

        Args:
            event (:obj:`wc_sim.core.Event`): an Event to process
        """
        # reaction_index is the reaction to execute
        reaction_index = event.message.reaction_index
        # Execute a reaction
        if self.enabled_reaction(self.reactions[reaction_index]):
            self.execute_reaction(self.reactions[reaction_index])
            # TODO(Arthur): convert print() to log message
            # print("{} executing '{}'".format(self.id, self.reactions[reaction_index].id))
        self.schedule_the_next_reaction()
