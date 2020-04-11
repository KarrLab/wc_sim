""" A submodel that employs Gibson and Bruck's Next Reaction Method (NRM) to model a set of reactions

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2020-04-11
:Copyright: 2020, Karr Lab
:License: MIT
"""

import sys
import math
import numpy as np
from scipy.constants import Avogadro

from de_sim.event import Event
from de_sim.simulation_object import SimulationObject
from wc_sim import message_types
from wc_sim.config import core as config_core_multialgorithm
from wc_sim.multialgorithm_errors import MultialgorithmError, DynamicFrozenSimulationError
from wc_sim.submodels.dynamic_submodel import DynamicSubmodel
from wc_utils.util.rand import RandomStateManager

config_multialgorithm = config_core_multialgorithm.get_config()['wc_sim']['multialgorithm']


class NrmSubmodel(DynamicSubmodel):
    """ Use the Next Reaction Method to predict the dynamics of chemical species in a container

    Attributes:
        dependencies (:obj:`list` of :obj:`set`): dependencies between reactions; maps each reaction index
            to the rate laws that depend on its execution
        random_state (:obj:`numpy.random.RandomState`): the random state that is shared across the
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
        self.options = options
        self.random_state = RandomStateManager.instance()

    def prepare(self):
        self.dependencies = self.determine_dependencies()

    def determine_dependencies(self):
        """ Determine the dependencies that rate laws have on executed reactions

        Returns:
            :obj:`list` of :obj:`set`: dependencies between reactions; map of each reaction index
                to the rate laws that depend on its execution
        """
        # in a multi-algorithmic simulation, two types of dependencies arise:
        # 1) ones identified by NRM -- rate laws that use species whose populations are updated by the reaction
        # 2) rate laws that use species whose populations might be updated by other submodels

        # dependencies[i] contains the indices of rate laws that depend on reaction i
        dependencies = {i: set() for i in range(len(self.reactions))}

        # updated_species[i] contains the ids of species whose populations are updated by reaction i
        updated_species = {i: set() for i in range(len(self.reactions))}

        # used_species[species_id] contains the indices of rate laws (reactions) that use species with id species_id
        used_species = {species.gen_id(): set() for species in self.species}

        # initialize reaction -> species -> reaction dependency dictionaries
        for reaction_idx, rxn in enumerate(self.reactions):
            for participant in rxn.participants:
                if participant.coefficient != 0:
                    species_id = participant.species.gen_id()
                    updated_species[reaction_idx].add(species_id)
            rate_law = rxn.rate_laws[0]
            for species in rate_law.expression.species:
                species_id = species.gen_id()
                used_species[species_id].add(reaction_idx)
        # TODO: add species updated by other submodels to updated_species
        '''
        get shared species from self.local_species_population
        '''
        # TODO: possible optimization: compute #2 dynamically, based on the time of the last update of the species

        # compute reaction to rate laws dependencies
        for reaction_idx, rxn in enumerate(self.reactions):
            for species_id in updated_species[reaction_idx]:
                for rate_law_idx in used_species[species_id]:
                    dependencies[reaction_idx].add(rate_law_idx)

        # todo: convert dependencies into list of tuples
        return dependencies

    def handle_ExecuteAndScheduleNrmReaction_msg(self, event):
        """ Handle an event containing a :obj:`ExecuteSsaReaction` message

        Args:
            event (:obj:`Event`): a simulation event
        """
        pass
