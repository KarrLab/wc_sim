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
        self.species = species
        self.dynamic_compartments = dynamic_compartments
        self.local_species_population = local_species_population
        self.fast_debug_file_logger = FastLogger(debug_logs.get_log('wc.debug.file'), 'debug')
        self.fast_debug_file_logger.fast_log(f"DynamicSubmodel.__init__: submodel: {self.id}; "
                                             f"reactions: {[reaction.id for reaction in reactions]}",
                                             sim_time=self.time)

    GET_STATE_METHOD_MESSAGE = 'object state to be provided by subclass'

    def get_state(self):
        return DynamicSubmodel.GET_STATE_METHOD_MESSAGE

    def prepare(self):
        """ If necessary, prepare a submodel after the :obj:`DynamicModel` has been fully initialized
        """
        pass

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
        rate = self.dynamic_model.dynamic_rate_laws[rate_law_id].eval(self.time)
        self.fast_debug_file_logger.fast_log(f"DynamicSubmodel.calc_reaction_rate: "
                                             f"rate of reaction {rate_law_id} = {rate}", sim_time=self.time)
        return rate

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
            msg = 'DynamicSubmodel.calc_reaction_rates: reactions and rates: ' + \
                str([(self.reactions[i].id, self.rates[i]) for i in range(len(self.reactions))])
            self.fast_debug_file_logger.fast_log(msg, sim_time=self.time)
        return self.rates

    # These methods - enabled_reaction, identify_enabled_reactions, execute_reaction - are used
    # by discrete time submodels like SsaSubmodel and NrmSubmodel.
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
        # flush entries that depend on reaction from cache
        self.dynamic_model.flush_after_reaction(reaction)

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


class ContinuousTimeSubmodel(DynamicSubmodel):
    """ Provide functionality that is shared by multiple continuous time submodels

    Discrete time submodels represent changes in species populations as step functions.
    Continuous time submodels model changes in species populations as continuous functions, currently
    piece-wise linear functions.
    ODEs and dFBA are continuous time submodels.

    Attributes:
        time_step (:obj:`float`): time interval between continuous time submodel analyses
        num_steps (:obj:`int`): number of analyses made
        options (:obj:`dict`): continuous time submodel options
        species_ids (:obj:`list`): ids of the species used by this continuous time submodel
        species_ids_set (:obj:`set`): ids of the species used by this continuous time submodel
        adjustments (:obj:`dict`): pre-allocated adjustments for passing changes to LocalSpeciesPopulation
        num_species (:obj:`int`): number of species in `species_ids`
        populations (:obj:`numpy.ndarray`): pre-allocated numpy array for storing species populations
    """
    def __init__(self, id, dynamic_model, reactions, species, dynamic_compartments,
                 local_species_population, time_step, options=None):
        super().__init__(id, dynamic_model, reactions, species, dynamic_compartments,
                         local_species_population)
        cls_name = type(self).__name__
        if not isinstance(time_step, (float, int)):
            raise MultialgorithmError(f"{cls_name} {self.id}: time_step must be a number but is "
                                      f"{type(time_step).__name__}")
        if time_step <= 0:
            raise MultialgorithmError(f"{cls_name} {self.id}: time_step must be positive, but is {time_step}")
        self.time_step = time_step
        self.num_steps = 0
        self.options = options

    def set_up_continuous_time_submodel(self):
        """ Begin setting up a continuous time submodel """

        # find species in reactions modeled by this continuous time submodel
        species = []
        for idx, rxn in enumerate(self.reactions):
            for species_coefficient in rxn.participants:
                species_id = species_coefficient.species.gen_id()
                species.append(species_id)
        self.species_ids = det_dedupe(species)

    def set_up_optimizations(self):
        """ To improve performance, pre-compute and pre-allocate some data structures """
        # make fixed set of species ids used by this continuous time submodel
        self.species_ids_set = set(self.species_ids)
        # pre-allocate dict of adjustments used to pass changes to LocalSpeciesPopulation
        self.adjustments = {species_id: None for species_id in self.species_ids}
        # pre-allocate numpy arrays for populations
        self.num_species = len(self.species_ids)
        self.populations = np.zeros(self.num_species)

    def current_species_populations(self):
        """ Obtain the current populations of species modeled by this continuous time submodel

        The current populations are written into `self.populations`.
        """
        pops_dict = self.local_species_population.read(self.time, self.species_ids_set, round=False)
        for idx, species_id in enumerate(self.species_ids):
            self.populations[idx] = pops_dict[species_id]
