""" A generic dynamic submodel; a multi-algorithmic model is constructed of multiple dynamic submodel subclasses

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2016-03-22
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

from scipy.constants import Avogadro
from wc_lang import Compartment, Species, Reaction, Parameter
from de_sim.simulation_object import ApplicationSimulationObject
from wc_sim.multialgorithm import message_types, distributed_properties
from wc_sim.multialgorithm.debug_logs import logs as debug_logs
from wc_sim.multialgorithm.dynamic_components import DynamicCompartment, DynamicModel
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError, SpeciesPopulationError
from wc_sim.multialgorithm.utils import get_species_and_compartment_from_name
import numpy as np

# TODO(Arthur): reactions -> dynamic reactions
# TODO(Arthur): species -> dynamic species, or morph into species populations species
# TODO(Arthur): add logging/debugging to dynamic reactions, dynamic species, etc.
# TODO(Arthur): use lists instead of sets for reproducibility


class DynamicSubmodel(ApplicationSimulationObject):
    """ Provide eneric dynamic submodel functionality

    Subclasses of `DynamicSubmodel` are combined into a multi-algorithmic model.

    Attributes:
        id (:obj:`str`): unique id of this dynamic submodel / simulation object
        dynamic_model (:obj:`DynamicModel`): the aggregate state of a simulation
        reactions (:obj:`list` of :obj:`Reaction`): the reactions modeled by this dynamic submodel
        rates (:obj:`np.array`): array to hold reaction rates
        species (:obj:`list` of :obj:`Species`): the species that participate in the reactions modeled
            by this dynamic submodel, with their initial concentrations
        dynamic_compartments (:obj:`dict` of :obj:`str`, :obj:`DynamicCompartment`): the dynamic compartments containing
            species that participate in reactions that this dynamic submodel models, including adjacent
            compartments used by its transfer reactions
        local_species_population (:obj:`LocalSpeciesPopulation`): the store that maintains this
            dynamic submodel's species population
        logger (:obj:`logging.Logger`): debug logger
    """

    def __init__(self, id, dynamic_model, reactions, species, dynamic_compartments, local_species_population):
        """ Initialize a dynamic submodel
        """
        super().__init__(id)
        self.id = id
        self.dynamic_model = dynamic_model
        self.reactions = reactions
        self.rates = np.full(len(self.reactions), np.nan)
        self.log_with_time("submodel: {}; reactions: {}".format(self.id,
                                                                [reaction.id for reaction in reactions]))
        self.species = species
        self.dynamic_compartments = dynamic_compartments
        self.local_species_population = local_species_population
        self.logger = debug_logs.get_log('wc.debug.file')

    # The next 3 methods implement the abstract methods in ApplicationSimulationObject
    def send_initial_events(self):
        pass    # pragma: no cover

    GET_STATE_METHOD_MESSAGE = 'object state to be provided by subclass'

    def get_state(self):
        return DynamicSubmodel.GET_STATE_METHOD_MESSAGE

    # At any time instant, event messages are processed in this order
    # TODO(Arthur): cover after MVP wc_sim done
    event_handlers = [(message_types.GetCurrentProperty, 'handle_get_current_prop_event')]  # pragma: no cover

    # TODO(Arthur): cover after MVP wc_sim done
    messages_sent = [message_types.GiveProperty]    # pragma: no cover

    def get_compartment_masses(self):
        """ Get the mass (g) of each compartment

        Returns:
            :obj:`dict`: dictionary that maps the ids of compartments to their masses (g)
        """
        return {id: comp.mass() for id, comp in self.dynamic_compartments.items()}

    def get_species_ids(self):
        """ Get ids of species used by this dynamic submodel

        Returns:
            :obj:`list`: ids of species used by this dynamic submodel
        """
        return [s.id for s in self.species]

    def get_species_counts(self):
        """ Get a dictionary of current species counts for this dynamic submodel

        Returns:
            :obj:`dict`: a map: species_id -> current copy number
        """
        species_ids = set(self.get_species_ids())
        return self.local_species_population.read(self.time, species_ids)

    def get_num_submodels(self):
        """ Provide the number of submodels

        Returns:
            :obj:`int`: the number of submodels
        """
        return self.dynamic_model.get_num_submodels()

    def calc_reaction_rates(self):
        """ Calculate the rates for this dynamic submodel's reactions

        Rates computed by eval'ing reactions provided in this dynamic submodel's definition,
        with species concentrations obtained by lookup from the dict
        `species_concentrations`. This assumes that all reversible reactions have been split
        into two forward reactions, as is done by `wc_lang.transform.SplitReversibleReactionsTransform`.

        Returns:
            :obj:`np.ndarray`: a numpy array of reaction rates, indexed by reaction index
        """
        # TODO(Arthur): optimization: get counts only for modifiers in the reactions
        species_counts = self.get_species_counts()
        compartment_masses = self.get_compartment_masses()

        for idx_reaction, rxn in enumerate(self.reactions):
            if rxn.rate_laws:
                self.rates[idx_reaction] = rxn.rate_laws[0].expression._parsed_expression.eval({
                    Species: species_counts,
                    Compartment: compartment_masses,
                    })

        # TODO(Arthur): optimization: get this if to work:
        # if self.logger.isEnabledFor(self.logger.getEffectiveLevel()):
        # print('self.logger.getEffectiveLevel())', self.logger.getEffectiveLevel())
        msg = str([(self.reactions[i].id, self.rates[i]) for i in range(len(self.reactions))])
        debug_logs.get_log('wc.debug.file').debug(msg, sim_time=self.time)
        return self.rates

    # These methods - enabled_reaction, identify_enabled_reactions, execute_reaction - are used
    # by discrete time submodels like SsaSubmodel and the SkeletonSubmodel.
    def enabled_reaction(self, reaction):
        """ Determine whether the cell state has adequate specie counts to run a reaction

        Indicate whether the current specie counts are large enough to execute `reaction`, based on
        its stoichiometry.

        Args:
            reaction (:obj:`Reaction`): the reaction to evaluate

        Returns:
            :obj:`bool`: True if `reaction` is stoichiometrically enabled
        """
        for participant in reaction.participants:
            species_id = participant.species.gen_id()
            count = self.local_species_population.read_one(self.time, species_id)
            # 'participant.coefficient < 0' determines whether the participant is a reactant
            is_reactant = participant.coefficient < 0
            if is_reactant and count < -participant.coefficient:
                return False
        return True

    def identify_enabled_reactions(self):
        """ Determine which reactions have adequate specie counts to run

        Returns:
            np array: an array indexed by reaction number; 0 indicates reactions without adequate
                species counts
        """
        enabled = np.full(len(self.reactions), 1)
        for idx_reaction, rxn in enumerate(self.reactions):
            if not self.enabled_reaction(rxn):
                enabled[idx_reaction] = 0

        return enabled

    def execute_reaction(self, reaction):
        """ Update species counts to reflect the execution of a reaction

        Called by discrete submodels, like SSA. Counts are updated in the `AccessSpeciesPopulations`
        that store them.

        Args:
            reaction (:obj:`Reaction`): the reaction being executed

        Raises:
            :obj:`MultialgorithmError:` if the species population cannot be updated
        """
        adjustments = {}
        for participant in reaction.participants:
            species_id = participant.species.gen_id()
            if not species_id in adjustments:
                adjustments[species_id] = 0
            adjustments[species_id] += participant.coefficient
        try:
            self.local_species_population.adjust_discretely(self.time, adjustments)
        except SpeciesPopulationError as e:
            raise MultialgorithmError("{:7.1f}: dynamic submodel '{}' cannot execute reaction: {}: {}".format(
                self.time, self.id, reaction.id, e))

    # TODO(Arthur): cover after MVP wc_sim done
    def handle_get_current_prop_event(self, event):   # pragma: no cover
        """ Handle a GetCurrentProperty simulation event.

        Args:
            event (:obj:`de_sim.event.Event`): an `Event` to process

        Raises:
            MultialgorithmError: if an `GetCurrentProperty` message requests an unknown property
        """
        property_name = event.message.property_name
        if property_name == distributed_properties.MASS:
            '''
            # TODO(Arthur): rethink this, as, strictly speaking, a dynamic submodel doesn't have mass, but its compartment does
                    self.send_event(0, event.sending_object, message_types.GiveProperty,
                        message=message_types.GiveProperty(property_name, self.time,
                            self.mass()))
            '''
            raise MultialgorithmError("Error: not handling distributed_properties.MASS")
        else:
            raise MultialgorithmError("Error: unknown property_name: '{}'".format(property_name))
