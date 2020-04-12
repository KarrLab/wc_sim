""" A submodel that employs Gibson and Bruck's Next Reaction Method (NRM) to model a set of reactions

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2020-04-11
:Copyright: 2020, Karr Lab
:License: MIT
"""

from pprint import pprint
import collections
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
        dependencies (:obj:`list` of :obj:`tuple`): entry i provides the indices of reactions whose
            rate laws depend on the execution of reaction i
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
            print(rxn.id)
            net_stoichiometric_coefficients = collections.defaultdict(float)
            for participant in rxn.participants:
                species_id = participant.species.gen_id()
                net_stoichiometric_coefficients[species_id] += participant.coefficient
                print(participant.species.gen_id(), participant.coefficient) 
            pprint(net_stoichiometric_coefficients)
            for species_id, net_stoich_coeff in net_stoichiometric_coefficients.items():
                if net_stoich_coeff < 0 or 0 < net_stoich_coeff:
                    updated_species[reaction_idx].add(species_id)
            rate_law = rxn.rate_laws[0]
            for species in rate_law.expression.species:
                species_id = species.gen_id()
                used_species[species_id].add(reaction_idx)
        print('updated_species:')
        pprint(updated_species)

        # Sequential case, with one instance each of an SSA, ODE, dFBA submodels:
        # NRM must recompute all rate laws that depend on species in a continuous submodels
        # TODO: possible optimization: compute #2 dynamically, based on the time of the last update of the species
        # will be: self.local_species_population.get_continuous_species()
        # TODO: move this code to LocalSpeciesPopulation and test
        # get shared species from self.local_species_population
        continuously_modeled_species = set()
        for species_id, dynamic_species_state in self.local_species_population._population.items():
            if dynamic_species_state.modeled_continuously:
                continuously_modeled_species.add(species_id)
        print('continuously_modeled_species:')
        pprint(continuously_modeled_species)
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

        print('dependencies_list':)
        pprint(dependencies_list)
        return dependencies_list

    def initial_propensities(self):
        """ Determine the dependencies that rate laws have on executed reactions

        Returns:
            :obj:`list` of :obj:`set`: dependencies between reactions; map of each reaction index
                to the rate laws that depend on its execution
        """
        pass

    def handle_ExecuteAndScheduleNrmReaction_msg(self, event):
        """ Handle an event containing a :obj:`ExecuteSsaReaction` message

        Args:
            event (:obj:`Event`): a simulation event
        """
        pass
