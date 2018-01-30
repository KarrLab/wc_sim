""" A generic submodel; multiple subclasses of submodel are combined into a multi-algorithmic model

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Author: Jonathan Karr, karr@mssm.edu
:Date: 2016-03-22
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

from itertools import chain
import numpy as np
from scipy.constants import Avogadro
from builtins import super

from wc_lang.core import Model, Species, Reaction, Compartment, Parameter
from wc_sim.core.simulation_object import SimulationObject, SimulationObjectInterface
from wc_sim.multialgorithm.utils import get_species_and_compartment_from_name
from wc_sim.multialgorithm.debug_logs import logs as debug_logs
from wc_sim.multialgorithm import message_types, distributed_properties


# TODO(Arthur): use list instead of set for reproducibility

class Submodel(SimulationObject, SimulationObjectInterface):
    """ Generic submodel functionality; subclasses of `Submodel` are combined into a multi-algorithmic model

    # TODO(Arthur): replace model with just the data a submodel needs
    # more generally, create a different dynamic representation

    Attributes:
        model (:obj:`Model`): a copy of the model to which this submodel belongs
        id (:obj:`str`): unique id of this submodel / simulation object
        access_species_pop (:obj:`AccessSpeciesPopulations`): an interface to the stores
            of all the species populations used by this submodel
        reactions (:obj:`list` of `Reaction`): the reactions modeled by this submodel
        species (:obj:`list` of `Species`): the species that participate in the reactions modeled
            by this submodel, with their initial concentrations
        compartment (:obj:`Compartment`): the compartment which this submodel models
        volume (:obj:`float`): the initial volume of this submodel's compartment
        parameters (:obj:`list` of `Parameter`): the model's parameters
        density (:obj:`float`): the density in the submodel, which is constant
    """

    def __init__(self, model, id, access_species_pop, reactions, species, compartment, parameters):
        """ Initialize a submodel
        """
        self.model = model
        self.id = id
        self.access_species_pop = access_species_pop
        self.reactions = reactions
        self.species = species
        self.compartment = compartment
        self.volume = compartment.initial_volume
        # TODO(Arthur): determine whether self.parameters is used
        self.parameters = parameters
        # density is assumed constant; calculate it once from initial values
        # TODO(Arthur): need mass of a compartment to determine the compartment's volume
        self.density = self.mass()/self.volume
        super().__init__(id)

    def send_initial_events(self):
        pass    # pragma: no cover

    def get_species_ids(self):
        """ Get ids of species used by this model

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
        return self.access_species_pop.read(self.time, species_ids)

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
            if compartment == self.compartment.id:
                concentrations[specie_id] = (counts[specie_id]/self.volume)/Avogadro
        return concentrations

    def mass(self):
        """ Provide the mass of the species modeled by this submodel

        This only measures molecules that are PRIVATELY modeled by this submodel,
        and does not include molecules that are shared with other submodels.

        Returns:
            :obj:`float`: the mass (g) of the molecules in this submodel's `LocalSpeciesPopulation`
        """
        return self.access_species_pop.local_pop_store.mass()

    # TODO(Arthur): need mass of a compartment to determine the compartment's volume
    def volume(self):
        """ Provide the volume of this submodel's compartment

        volume is a distributed property over all the submodels that model reactions in a compartment

        Returns:
            float: the volume (L) of this submodel
        """
        return self.mass()/self.density

    @staticmethod
    def calc_reaction_rates(reactions, species_concentrations):
        """ Calculate the rates for a submodel's reactions.

        Rates computed by eval'ing reactions provided in the model definition,
        with species concentrations obtained by lookup from the dict
        `species_concentrations`.

        Args:
            reactions (list): reactions modeled by a calling submodel
            species_concentrations (dict): `species_id` -> `Species()` object; current
                concentration of species participating in reactions

        Returns:
            A numpy array of reaction rates, indexed by reaction index.
        """
        rates = np.full(len(reactions), np.nan)
        for idx_reaction, rxn in enumerate(reactions):
            if rxn.rate_law:
                rates[idx_reaction] = Submodel.eval_rate_law(rxn, species_concentrations)
        return rates

    # TODO(Arthur): move any of the next 3 methods which are only needed by SSA to SSASubmodel
    def enabled_reaction(self, reaction):
        """ Determine whether the cell state has adequate specie counts to run a reaction.

        Args:
            reaction (:obj:`Reaction`): the reaction to evaluate

        Returns:
            :obj:`bool`: True if `reaction` is stoichiometrically enabled
        """
        for participant in reaction.participants:
            species_id = Species.gen_id(participant.species.species_type,
                participant.species.compartment)
            count = self.access_species_pop.read_one(self.time, species_id)
            # 'participant.coefficient < 0' determines whether the participant is a reactant
            is_reactant = participant.coefficient < 0
            if is_reactant and count < -participant.coefficient:
                return False
        return True

    def identify_enabled_reactions(self, propensities):
        """ Determine which reactions have adequate specie counts to run.

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
            self.access_species_pop.adjust_discretely(self.time, adjustments)
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
            self.send_event(0, event.sending_object, message_types.GiveProperty,
                event_body=message_types.GiveProperty(property_name, self.time,
                    self.mass()))
        else:
            raise ValueError("Error: unknown property_name: '{}'".format(property_name))

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
