""" A simplistic, configurable submodel for testing during development.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016_12_06
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""
from builtins import super

from wc_sim.core.simulation_object import SimulationObject
from wc_sim.multialgorithm import message_types
from wc_sim.multialgorithm.species_populations import AccessSpeciesPopulations
from wc_sim.multialgorithm.submodels.dynamic_submodel import DynamicSubmodel
from wc_sim.multialgorithm.message_types import ALL_MESSAGE_TYPES

class SkeletonSubmodel(DynamicSubmodel):
    """ Init a SkeletonSubmodel

    A SkeletonSubmodel is a simple submodel with externally controlled behavior.

    Attributes:
        behavior (:obj:`wc_sim.core.Event`): an Event to process

        next_reaction
    """
    def __init__(self, id, reactions, species, parameters, dynamic_compartment,
        local_species_population, behavior):
        """ Init a SkeletonSubmodel."""
        self.behavior = behavior
        self.local_species_population = local_species_population
        self.next_reaction = 0
        super().__init__(id, reactions, species, parameters, dynamic_compartment, local_species_population)

    def handle_GivePopulation_event(self, event):
        """ Handle a simulation event.

        Args:
            event (:obj:`wc_sim.core.Event`): an Event to process
        """
        # populations is a GivePopulation_body instance
        populations = event.event_body
        self.access_species_population.species_population_cache.cache_population(
            self.time, populations)

    def handle_ExecuteSsaReaction_event(self, event):
        """ Handle a simulation event.

        Args:
            event (:obj:`wc_sim.core.Event`): an Event to process
        """
        # reaction_index is the reaction to execute
        reaction_index = event.event_body.reaction_index
        # Execute a reaction
        # TODO(Arthur): first need to identify_enabled_reactions
        if self.enabled_reaction(self.reactions[reaction_index]):
            self.execute_reaction(self.access_species_population,
                self.reactions[reaction_index])
            # TODO(Arthur): convert print() to log message
            # print("{} executing '{}'".format(self.name, str(self.reactions[reaction_index])))

        # Schedule the next reaction
        dt = self.behavior['INTER_REACTION_TIME']
        self.next_reaction += 1
        if self.next_reaction == len(self.reactions):
            self.next_reaction = 0
        self.send_event(dt, self, message_types.ExecuteSsaReaction,
            message_types.ExecuteSsaReaction(self.next_reaction))

        # issue read request for species population at time of next reaction
        species_ids = set([s.id for s in self.species])
        # TODO(Arthur): IMPORTANT: only prefetch reactants
        self.access_species_population.prefetch(dt, species_ids)

    @classmethod
    def register_subclass_handlers(this_class):
        SimulationObject.register_handlers(this_class, [
            (message_types.GivePopulation, this_class.handle_GivePopulation_event),
            (message_types.ExecuteSsaReaction, this_class.handle_ExecuteSsaReaction_event),
        ])

    @classmethod
    def register_subclass_sent_messages(this_class):
        SimulationObject.register_sent_messages(this_class, ALL_MESSAGE_TYPES)
