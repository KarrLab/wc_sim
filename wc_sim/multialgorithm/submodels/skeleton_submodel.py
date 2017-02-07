'''A simplistic, configurable submodel for testing during development.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016_12_06
:Copyright: 2016, Karr Lab
:License: MIT
'''

from wc_utils.util.misc import isclass_by_name
from wc_sim.core.simulation_engine import MessageTypesRegistry
from wc_sim.core.simulation_object import SimulationObject, EventQueue
from wc_sim.multialgorithm import message_types
from wc_sim.multialgorithm.species_populations import AccessSpeciesPopulations
from wc_sim.multialgorithm.submodels.submodel import Submodel
from wc_sim.multialgorithm.message_types import ALL_MESSAGE_TYPES

class SkeletonSubmodel(Submodel):
    '''Init a SkeletonSubmodel.
    
    A SkeletonSubmodel is a simple submodel with externally controlled behavior.
    
    Attributes:
        behavior
        next_reaction
    '''

    def __init__(self, behavior, model, name, access_species_population,
        reactions, species, parameters):
        '''Init a SkeletonSubmodel.'''
        super(SkeletonSubmodel, self).__init__(model, name, access_species_population,
            reactions, species, parameters)
        self.behavior = behavior
        self.access_species_population = access_species_population
        self.next_reaction = 0

    def handle_event(self, event_list):
        '''Handle a SkeletonSubmodel simulation event.

        Args:
            event_list (:obj:`list` of :obj:`wc_sim.core.Event`): list of Events to process.

        Raises:
            TBD
        '''
        SimulationObject.handle_event(self, event_list)

        for event_message in event_list:

            if isclass_by_name(event_message.event_type, message_types.GivePopulation):
                # populations is a GivePopulation_body instance
                populations = event_message.event_body
                self.access_species_population.species_population_cache.cache_population(
                    self.time, populations)

            elif isclass_by_name(event_message.event_type, message_types.ExecuteSsaReaction):
                # reaction_index is the reaction to execute
                reaction_index = event_message.event_body.reaction_index
                # Execute a reaction
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
                    message_types.ExecuteSsaReaction.Body(self.next_reaction))

                # issue read request for species population at time of next reaction
                species_ids = set([s.id for s in self.species])
                # TODO(Arthur): IMPORTANT: only prefetch reactants
                self.access_species_population.prefetch(dt, species_ids)

            else:
                raise ValueError("Error: event_message.event_type '{}' should "\
                "be covered in the if statement above".format(event_message.event_type))

MessageTypesRegistry.set_sent_message_types(SkeletonSubmodel, ALL_MESSAGE_TYPES)
MESSAGE_TYPES_BY_PRIORITY = [
    message_types.GivePopulation,
    message_types.ExecuteSsaReaction]
MessageTypesRegistry.set_receiver_priorities(SkeletonSubmodel, MESSAGE_TYPES_BY_PRIORITY)
