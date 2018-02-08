""" A generic submodel; a multi-algorithmic model is constructed of multiple submodel subclasses

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Author: Jonathan Karr, karr@mssm.edu
:Date: 2016-03-22
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

import numpy as np
from scipy.constants import Avogadro
from builtins import super

from wc_lang.core import Species, Reaction, Compartment, Parameter
from wc_sim.multialgorithm.dynamic_components import DynamicCompartment
from wc_sim.core.simulation_object import SimulationObject, SimulationObjectInterface
from wc_sim.multialgorithm.utils import get_species_and_compartment_from_name
from wc_sim.multialgorithm.debug_logs import logs as debug_logs
from wc_sim.multialgorithm import message_types, distributed_properties

# TODO(Arthur): reactions -> dynamic reactions
# TODO(Arthur): species -> dynamic species, or morph into species populations species
# TODO(Arthur): this submodel's parameters -> dynamic parameters
# TODO(Arthur): use lists instead of sets for reproducibility


class DynamicSubmodel(SimulationObject, SimulationObjectInterface):
    """ Generic submodel functionality; subclasses of `DynamicSubmodel` are combined into a multi-algorithmic model

    Attributes:
        id (:obj:`str`): unique id of this submodel / simulation object
        reactions (:obj:`list` of `Reaction`): the reactions modeled by this submodel
        species (:obj:`list` of `Species`): the species that participate in the reactions modeled
            by this submodel, with their initial concentrations
        parameters (:obj:`list` of `Parameter`): the model's parameters
        dynamic_compartment (:obj:`DynamicCompartment`): the compartment which this submodel models
        local_species_population (:obj:`LocalSpeciesPopulation`): the store that maintains this
            submodel's species population
    """

    def __init__(self, id, reactions, species, parameters, dynamic_compartment, local_species_population):
        """ Initialize a submodel
        """
        self.id = id
        self.reactions = reactions
        self.species = species
        self.parameters = parameters
        self.dynamic_compartment = dynamic_compartment
        self.local_species_population = local_species_population
        super().__init__(id)

    # The next 3 methods implement the abstract SimulationObjectInterface methods
    def send_initial_events(self):
        pass    # pragma: no cover

    @classmethod
    def register_subclass_handlers(cls):
        SimulationObject.register_handlers(cls, [
            # At any time instant, event messages are processed in this order
            (message_types.GetCurrentProperty, cls.handle_get_current_prop_event),
        ])

    @classmethod
    def register_subclass_sent_messages(cls):
        SimulationObject.register_sent_messages(cls,
            [message_types.GiveProperty])

    def get_species_ids(self):
        """ Get ids of species used by this submodel

        Returns:
            :obj:`list`: ids of species used by this submodel
        """
        return [s.id() for s in self.species]

    def get_specie_counts(self):
        """ Get a dictionary of current species counts for this submodel

        Returns:
            :obj:`dict`: a map: species_id -> current copy number
        """
        species_ids = set(self.get_species_ids())
        return self.local_species_population.read(self.time, species_ids)

    def get_specie_concentrations(self):
        """ Get the current species concentrations for this submodel

        concentration ~ count/volume
        Provide concentrations for only species stored in this submodel's compartment, whose
        volume is known.

        Returns:
            :obj:`dict`: a map: species_id -> species concentration
        """
        counts = self.get_specie_counts()
        ids = self.get_species_ids()
        concentrations = {}
        for specie_id in ids:
            (_, compartment) = get_species_and_compartment_from_name(specie_id)
            if compartment == self.dynamic_compartment.id:
                concentrations[specie_id] = (counts[specie_id]/self.dynamic_compartment.volume())/Avogadro
        return concentrations

    @staticmethod
    def calc_reaction_rates(reactions):
        """ Calculate the rates for a submodel's reactions

        Rates computed by eval'ing reactions provided in this submodel's definition,
        with species concentrations obtained by lookup from the dict
        `species_concentrations`.

        Args:
            reactions (list): reactions modeled by a calling submodel

        Returns:
            A numpy array of reaction rates, indexed by reaction index.
        """
        rates = np.full(len(reactions), np.nan)
        species_concentrations = self.get_specie_concentrations()
        for idx_reaction, rxn in enumerate(reactions):
            if rxn.rate_law:
                rates[idx_reaction] = DynamicSubmodel.eval_rate_law(rxn, species_concentrations)
        return rates

    # These methods - enabled_reaction, identify_enabled_reactions, execute_reaction - are used
    # by discrete time submodels like SSA and the SkeletonSubmodel
    def enabled_reaction(self, reaction):
        """ Determine whether the cell state has adequate specie counts to run a reaction

        Args:
            reaction (:obj:`Reaction`): the reaction to evaluate

        Returns:
            :obj:`bool`: True if `reaction` is stoichiometrically enabled
        """
        for participant in reaction.participants:
            species_id = Species.gen_id(participant.species.species_type,
                participant.species.compartment)
            count = self.local_species_population.read_one(self.time, species_id)
            # 'participant.coefficient < 0' determines whether the participant is a reactant
            is_reactant = participant.coefficient < 0
            if is_reactant and count < -participant.coefficient:
                return False
        return True

    def identify_enabled_reactions(self, propensities):
        """ Determine which reactions have adequate specie counts to run

        A reaction's mass action kinetics, as computed by calc_reaction_rates(),
        may be positive when insufficient species are available to execute the reaction.
        Identify reactions with specie counts smaller than their stoichiometric coefficient.

        Args:
            propensities: np array: the current propensities for this submodel's reactions

        Returns:
            np array: an array indexed by reaction number; 0 indicates reactions with a
            propensity of 0 or without adequate species counts
        """
        enabled = np.full(len(self.reactions), 1)
        for idx_reaction, rxn in enumerate(self.reactions):
            # ignore reactions with propensities that are already 0
            if propensities[idx_reaction] <= 0:
                enabled[idx_reaction] = 0
                continue

            # compare each reaction with its stoichiometry,
            # reaction disabled if the specie count of any reactant is less than its coefficient
            if not self.enabled_reaction(rxn):
                enabled[idx_reaction] = 0

                log = debug_logs.get_log('wc.debug.file')
                log.debug(
                    "reaction: {} of {}: insufficient counts".format(idx_reaction, len(self.reactions)),
                    sim_time=self.time)
        return enabled

    def execute_reaction(self, reaction):
        """ Update species counts to reflect the execution of a reaction

        Called by discrete submodels, like SSA. Counts are updated in the `AccessSpeciesPopulations`
        that store them.

        Args:
            reaction (:obj:`Reaction`): the reaction being executed

        Raises:
            :obj:`ValueError:` if the species population cannot be updated
        """
        adjustments = {}
        for participant in reaction.participants:
            species_id = Species.gen_id(participant.species.species_type,
                participant.species.compartment)
            adjustments[species_id] = participant.coefficient
        try:
            self.local_species_population.adjust_discretely(self.time, adjustments)
        except ValueError as e:
            raise ValueError("{:7.1f}: submodel {}: reaction: {}: {}".format(self.time, self.id,
                reaction.id, e))

    def handle_get_current_prop_event(self, event):
        """ Handle a GetCurrentProperty simulation event.

        Args:
            event (:obj:`wc_sim.core.Event`): an `Event` to process

        Raises:
            ValueError: if an `GetCurrentProperty` message requests an unknown
                property.
        """
        property_name = event.event_body.property_name
        if property_name == distributed_properties.MASS:
            '''
            # TODO(Arthur): rethink this, as, strictly speaking, a submodel doesn't have mass, but its compartment does
                    self.send_event(0, event.sending_object, message_types.GiveProperty,
                        event_body=message_types.GiveProperty(property_name, self.time,
                            self.mass()))
            '''
            raise ValueError("Error: not handling distributed_properties.MASS")
        else:
            raise ValueError("Error: unknown property_name: '{}'".format(property_name))
