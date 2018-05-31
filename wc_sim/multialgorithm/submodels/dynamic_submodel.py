""" A generic dynamic submodel; a multi-algorithmic model is constructed of multiple dynamic submodel subclasses

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Author: Jonathan Karr, karr@mssm.edu
:Date: 2016-03-22
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

import numpy as np
from scipy.constants import Avogadro

from wc_lang.core import Species, Reaction, Compartment, Parameter
from wc_lang.rate_law_utils import RateLawUtils
from wc_sim.multialgorithm.dynamic_components import DynamicCompartment
from wc_sim.core.simulation_object import SimulationObject, ApplicationSimulationObject
from wc_sim.multialgorithm.utils import get_species_and_compartment_from_name
from wc_sim.multialgorithm.debug_logs import logs as debug_logs
from wc_sim.multialgorithm import message_types, distributed_properties
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError, SpeciesPopulationError

# TODO(Arthur): reactions -> dynamic reactions
# TODO(Arthur): species -> dynamic species, or morph into species populations species
# TODO(Arthur): this dynamic submodel's parameters -> dynamic parameters
# TODO(Arthur): add logging/debugging to dynamic reactions, dynamic species, etc.
# TODO(Arthur): use lists instead of sets for reproducibility

class DynamicSubmodel(ApplicationSimulationObject):
    """ Provide eneric dynamic submodel functionality

    Subclasses of `DynamicSubmodel` are combined into a multi-algorithmic model.

    Attributes:
        id (:obj:`str`): unique id of this dynamic submodel / simulation object
        reactions (:obj:`list` of `Reaction`): the reactions modeled by this dynamic submodel
        species (:obj:`list` of `Species`): the species that participate in the reactions modeled
            by this dynamic submodel, with their initial concentrations
        parameters (:obj:`list` of `Parameter`): the model's parameters
        dynamic_compartments (:obj:`list` of `DynamicCompartment`): the dynamic compartments containing
            species that participate in reactions that this dynamic submodel models, including adjacent
            compartments used by its transfer reactions
        local_species_population (:obj:`LocalSpeciesPopulation`): the store that maintains this
            dynamic submodel's species population
    """
    def __init__(self, id, reactions, species, parameters, dynamic_compartments, local_species_population):
        """ Initialize a dynamic submodel
        """
        self.id = id
        self.reactions = reactions
        self.species = species
        self.parameters = parameters
        self.dynamic_compartments = dynamic_compartments
        self.local_species_population = local_species_population
        super().__init__(id)

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

    def get_species_ids(self):
        """ Get ids of species used by this dynamic submodel

        Returns:
            :obj:`list`: ids of species used by this dynamic submodel
        """
        return [s.id() for s in self.species]

    def get_specie_counts(self):
        """ Get a dictionary of current species counts for this dynamic submodel

        Returns:
            :obj:`dict`: a map: species_id -> current copy number
        """
        species_ids = set(self.get_species_ids())
        return self.local_species_population.read(self.time, species_ids)

    def get_specie_concentrations(self):
        """ Get the current species concentrations for this dynamic submodel

        Concentrations are obtained from species counts.
        concentration ~ count/volume
        Provide concentrations for only species stored in this dynamic submodel's compartments, whose
        volume is known.

        Returns:
            :obj:`dict`: a map: species_id -> species concentration

        Raises:
            :obj:`MultialgorithmError:` if a dynamic compartment cannot be found for a specie being modeled,
                or if the compartments volume is 0
        """
        counts = self.get_specie_counts()
        concentrations = {}
        for specie_id in self.get_species_ids():
            (_, compartment_id) = get_species_and_compartment_from_name(specie_id)
            if compartment_id not in self.dynamic_compartments:
                raise MultialgorithmError("dynamic submodel '{}' lacks dynamic compartment '{}' for specie '{}'".format(
                    self.id, compartment_id, specie_id))
            dynamic_compartment = self.dynamic_compartments[compartment_id]
            if dynamic_compartment.volume() == 0:
                raise MultialgorithmError("dynamic submodel '{}' cannot compute concentration in "
                    "compartment '{}' with volume=0".format(self.id, compartment_id))

            concentrations[specie_id] = counts[specie_id]/(dynamic_compartment.volume()*Avogadro)
        return concentrations

    def get_parameter_values(self):
        """ Get the current parameter values for this dynamic submodel

        Returns:
            :obj:`dict`: a map: parameter_id -> parameter value
        """
        vals = {}
        for param in self.parameters:
            vals[param.id] = param.value
        return vals

    def calc_reaction_rates(self):
        """ Calculate the rates for this dynamic submodel's reactions

        Rates computed by eval'ing reactions provided in this dynamic submodel's definition,
        with species concentrations obtained by lookup from the dict
        `species_concentrations`. This assumes that all reversible reactions have been split
        into two forward reactions, as is done by `wc_lang.transform.SplitReversibleReactionsTransform`.

        Returns:
            :obj:`np.ndarray`: a numpy array of reaction rates, indexed by reaction index
        """
        # TODO(Arthur): optimization: since len(self.reactions) is constant, preallocate this array
        rates = np.full(len(self.reactions), np.nan)
        # TODO(Arthur): optimization: get concentrations only for modifiers in the reactions
        species_concentrations = self.get_specie_concentrations()
        for idx_reaction, rxn in enumerate(self.reactions):            
            if rxn.rate_laws:
                parameter_values = {param.id: param.value for param in rxn.rate_laws[0].equation.parameters}
                rates[idx_reaction] = RateLawUtils.eval_rate_law(rxn.rate_laws[0], species_concentrations, parameter_values)
        return rates

    # These methods - enabled_reaction, identify_enabled_reactions, execute_reaction - are used
    # by discrete time submodels like SSASubmodel and the SkeletonSubmodel.
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
            species_id = Species.gen_id(participant.species.species_type,
                participant.species.compartment)
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
            species_id = Species.gen_id(participant.species.species_type,
                participant.species.compartment)
            adjustments[species_id] = participant.coefficient
        try:
            self.local_species_population.adjust_discretely(self.time, adjustments)
        except SpeciesPopulationError as e:
            raise MultialgorithmError("{:7.1f}: dynamic submodel '{}' cannot execute reaction: {}: {}".format(
                self.time, self.id, reaction.id, e))

    # TODO(Arthur): cover after MVP wc_sim done
    def handle_get_current_prop_event(self, event):   # pragma: no cover
        """ Handle a GetCurrentProperty simulation event.

        Args:
            event (:obj:`wc_sim.core.Event`): an `Event` to process

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
