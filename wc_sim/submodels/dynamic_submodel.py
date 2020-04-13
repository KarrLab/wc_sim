""" A generic dynamic submodel; a multi-algorithmic model is constructed of multiple dynamic submodel subclasses

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2016-03-22
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

from de_sim.utilities import FastLogger
from scipy.constants import Avogadro
import numpy as np

from de_sim.simulation_object import ApplicationSimulationObject
from wc_lang import Compartment, Species, Reaction, Parameter
from wc_sim import message_types, distributed_properties
from wc_sim.debug_logs import logs as debug_logs
from wc_sim.dynamic_components import DynamicCompartment, DynamicModel
from wc_sim.multialgorithm_errors import (DynamicMultialgorithmError, MultialgorithmError,
                                          DynamicSpeciesPopulationError)


# TODO(Arthur): rename reactions -> dynamic reactions
# TODO(Arthur): species -> dynamic species, or morph into species populations species
# TODO(Arthur): use lists instead of sets for reproducibility


class DynamicSubmodel(ApplicationSimulationObject):
    """ Provide generic dynamic submodel functionality

    All submodels are implemented as subclasses of `DynamicSubmodel`. Instances of them are combined
    to make a multi-algorithmic model.

    Attributes:
        id (:obj:`str`): unique id of this dynamic submodel and simulation object
        dynamic_model (:obj:`DynamicModel`): the aggregate state of a simulation
        reactions (:obj:`list` of :obj:`Reaction`): the reactions modeled by this dynamic submodel
        rates (:obj:`np.array`): array to hold reaction rates
        species (:obj:`list` of :obj:`Species`): the species that participate in the reactions modeled
            by this dynamic submodel, with their initial concentrations
        dynamic_compartments (:obj:`dict` of :obj:`str`, :obj:`DynamicCompartment`): the dynamic compartments
            containing species that participate in reactions that this dynamic submodel models, including
            adjacent compartments used by its transfer reactions
        local_species_population (:obj:`LocalSpeciesPopulation`): the store that maintains this
            dynamic submodel's species population
        fast_debug_file_logger (:obj:`FastLogger`): a fast logger for debugging messages
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
        self.fast_debug_file_logger = FastLogger(debug_logs.get_log('wc.debug.file'), 'debug')

    # The next 2 methods implement the abstract methods in ApplicationSimulationObject
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

    def calc_reaction_rate(self, reaction):
        """ Calculate a reaction's current rate

        The rate is computed by eval'ing the reaction's rate law,
        with species populations obtained from the simulations's :obj:`LocalSpeciesPopulation`.

        Args:
            reaction (:obj:`Reaction`): the reaction to evaluate

        Returns:
            :obj:`float`: the reaction's rate
        """
        rate_law_id = reaction.rate_laws[0].id
        return self.dynamic_model.dynamic_rate_laws[rate_law_id].eval(self.time)

    def calc_reaction_rates(self):
        """ Calculate the rates for this dynamic submodel's reactions

        Rates are computed by eval'ing rate laws for reactions used by this dynamic submodel,
        with species populations obtained from the simulations's :obj:`LocalSpeciesPopulation`.
        This assumes that all reversible reactions have been split
        into two forward reactions, as is done by `wc_lang.transform.SplitReversibleReactionsTransform`.

        Returns:
            :obj:`np.ndarray`: a numpy array of reaction rates, indexed by reaction index
        """
        for idx_reaction, rxn in enumerate(self.reactions):
            if rxn.rate_laws:
                self.rates[idx_reaction] = self.calc_reaction_rate(rxn)

        if self.fast_debug_file_logger.is_active():
            msg = str([(self.reactions[i].id, self.rates[i]) for i in range(len(self.reactions))])
            self.fast_debug_file_logger.fast_log(msg, sim_time=self.time)
        return self.rates

    # These methods - enabled_reaction, identify_enabled_reactions, execute_reaction - are used
    # by discrete time submodels like SsaSubmodel and the SkeletonSubmodel.
    def enabled_reaction(self, reaction):
        """ Determine whether the cell state has adequate species counts to run a reaction

        Indicate whether the current species counts are large enough to execute `reaction`, based on
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
        """ Determine which reactions have adequate species counts to run

        Returns:
            :obj:`np.array`: an array indexed by reaction number; 0 indicates reactions without adequate
                species counts
        """
        enabled = np.full(len(self.reactions), 1)
        for idx_reaction, rxn in enumerate(self.reactions):
            if not self.enabled_reaction(rxn):
                enabled[idx_reaction] = 0

        return enabled

    def execute_reaction(self, reaction):
        """ Update species counts to reflect the execution of a reaction

        Called by discrete submodels, like SSA. Counts are updated in the :obj:`LocalSpeciesPopulation`
        that stores them.

        Args:
            reaction (:obj:`Reaction`): the reaction being executed

        Raises:
            :obj:`DynamicMultialgorithmError:` if the species population cannot be updated
        """
        adjustments = {}
        for participant in reaction.participants:
            species_id = participant.species.gen_id()
            if not species_id in adjustments:
                adjustments[species_id] = 0
            adjustments[species_id] += participant.coefficient
        try:
            self.local_species_population.adjust_discretely(self.time, adjustments)
        except DynamicSpeciesPopulationError as e:
            raise DynamicMultialgorithmError(self.time, "dynamic submodel '{}' cannot execute reaction: {}: {}".format(
                self.id, reaction.id, e))

    # TODO(Arthur): cover after MVP wc_sim done
    def handle_get_current_prop_event(self, event):   # pragma: no cover    not used
        """ Handle a GetCurrentProperty simulation event.

        Args:
            event (:obj:`de_sim.event.Event`): an `Event` to process

        Raises:
            DynamicMultialgorithmError: if an `GetCurrentProperty` message requests an unknown property
        """
        property_name = event.message.property_name
        if property_name == distributed_properties.MASS:
            '''
            # TODO(Arthur): rethink this, as, strictly speaking, a dynamic submodel doesn't have mass, but its compartment does
                    self.send_event(0, event.sending_object, message_types.GiveProperty,
                        message=message_types.GiveProperty(property_name, self.time,
                            self.mass()))
            '''
            raise DynamicMultialgorithmError(self.time, "Error: not handling distributed_properties.MASS")
        else:
            raise DynamicMultialgorithmError(self.time, "Error: unknown property_name: '{}'".format(property_name))
