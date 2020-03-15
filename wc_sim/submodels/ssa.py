""" A sub-model that employs Gillespie's Stochastic Simulation Algorithm (SSA) to model a set of reactions.

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2016-07-14
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

import sys
import math
import numpy as np
from scipy.constants import Avogadro

from wc_utils.util.rand import RandomStateManager
from wc_utils.util.stats import ExponentialMovingAverage

from de_sim.simulation_object import SimulationObject
from de_sim.event import Event
from wc_sim import message_types
from wc_sim.config import core as config_core_multialgorithm
from wc_sim.submodels.dynamic_submodel import DynamicSubmodel
from wc_sim.multialgorithm_errors import MultialgorithmError, DynamicFrozenSimulationError

config_multialgorithm = config_core_multialgorithm.get_config()['wc_sim']['multialgorithm']


class SsaSubmodel(DynamicSubmodel):
    """ Use the Stochastic Simulation Algorithm to predict the dynamics of chemical species in a container

    This implementation supports a partition of the species populations into private, locally stored
    populations and shared, remotely stored populations. These are accessed through the ADT provided
    by the `DynamicSubmodel`'s `LocalSpeciesPopulation`. Appropriate optimizations are made if no
    populations are stored remotely.

    # TODO(Arthur): update the rest of this doc string
    # TODO(Arthur): ensure that this doc string formats properly
    Algorithm::

        implement the 'direct' method, except under unusual circumstances

        determine_reaction_propensities():
            determine reactant concentrations
            determine propensities
            eliminate reactions that are not stoichiometrically enabled
            return propensities, total_propensities

        schedule_next_event():
            determine_reaction_propensities()
            if total_propensities == 0: *
                schedule_SsaWait()      *
            else:
                reaction delay = random sample of exp(1/total_propensities)
                select and schedule the next reaction

        execute_reaction():
            if scheduled reaction is stoichiometrically enabled:
                execute it
            schedule_next_event()

        *  2nd order recovery because other submodels can modify shared species counts

    Attributes:
        num_SsaWaits (:obj:`int`): count of :obj:`SsaWaits` executed by this :obj:`SsaSubmodel`
        ema_of_inter_event_time (:obj:`ExponentialMovingAverage`): an EMA of the time between
            :obj:`ExecuteSsaReaction` events; when total propensities == 0, `ema_of_inter_event_time`
            is used as the time delay between :obj:`SsaWait` events
        random_state (:obj:`numpy.random.RandomState`): the random state that is shared across the
            simulation, which enables reproducible checkpoint and restore of a simulation
    """

    # message types sent by SsaSubmodel
    SENT_MESSAGE_TYPES = [
        message_types.AdjustPopulationByDiscreteSubmodel,
        message_types.ExecuteSsaReaction,
        message_types.GetPopulation,
        message_types.SsaWait]

    # register the message types sent
    messages_sent = SENT_MESSAGE_TYPES

    # at any time instant, process messages in this order
    MESSAGE_TYPES_BY_PRIORITY = [
        message_types.SsaWait,
        message_types.GivePopulation,
        message_types.ExecuteSsaReaction]

    # TODO(Arthur): report a compile time error if messages_sent or event_handlers is undefined
    # TODO(Arthur): make example with multiple event handlers
    event_handlers = [(sim_msg_type, 'handle_{}_msg'.format(sim_msg_type.__name__))
        for sim_msg_type in MESSAGE_TYPES_BY_PRIORITY]

    def __init__(self, id, dynamic_model, reactions, species, dynamic_compartments,
                 local_species_population, default_center_of_mass=None, options=None):
        """ Initialize an SSA submodel object.

        Args:
            id (:obj:`str`): unique id of this dynamic SSA submodel
            dynamic_model (:obj:`DynamicModel`): the aggregate state of a simulation
            reactions (:obj:`list` of :obj:`Reaction`): the reactions modeled by this SSA submodel
            species (:obj:`list` of :obj:`Species`): the species that participate in the reactions modeled
                by this SSA submodel, with their initial concentrations
            dynamic_compartments (:obj:`dict`): :obj:`DynamicCompartment`\ s, keyed by id, that contain
                species which participate in reactions that this SSA submodel models, including
                adjacent compartments used by its transfer reactions
            local_species_population (:obj:`LocalSpeciesPopulation`): the store that maintains this
                SSA submodel's species population
            default_center_of_mass (:obj:`float`, optional): the center-of-mass for the
                :obj:`ExponentialMovingAverage`
            options (:obj:`dict`, optional): SSA submodel options

        Raises:
            :obj:`MultialgorithmError`: if the initial SSA wait exponential moving average is not positive
        """
        super().__init__(id, dynamic_model, reactions, species, dynamic_compartments,
                         local_species_population)
        self.options = options
        self.num_SsaWaits=0
        # `initial_ssa_wait_ema` must be positive, as otherwise an infinite sequence of SsaWait
        # messages will be executed at the start of a simulation if no reactions are enabled
        initial_ssa_wait_ema = config_multialgorithm['initial_ssa_wait_ema']
        if initial_ssa_wait_ema <= 0:   # pragma: no cover
            raise MultialgorithmError(f"'initial_ssa_wait_ema' must be positive to avoid infinite sequence of "
            "SsaWait messages, but it is {initial_ssa_wait_ema}")
        if default_center_of_mass is None:
            default_center_of_mass = config_multialgorithm['default_center_of_mass']
        self.ema_of_inter_event_time = ExponentialMovingAverage(initial_ssa_wait_ema,
                                                                center_of_mass=default_center_of_mass)
        self.random_state = RandomStateManager.instance()

        self.log_with_time("init: id: {}".format(id))
        self.log_with_time("init: species: {}".format(str([s.id for s in species])))

    def send_initial_events(self):
        """ Send this SSA submodel's initial events
        """
        self.schedule_next_events()

    def determine_reaction_propensities(self):
        """ Determine the current reaction propensities for this submodel.

        Method:
        1. calculate concentrations
        # TODO(Arthur): IMPORTANT: optimization: simply use counts to calculate propensities
        # TODO(Arthur): IMPORTANT: create benchmark & profile data to evaluate possible optimizations
        2. calculate propensities for this submodel
        3. avoid reactions with inadequate species counts

        Returns:
            reaction (propensities, total_propensities)

        Raises:
            :obj:`DynamicFrozenSimulationError`: if the simulation has 1 submodel and the total propensities are 0
        """

        # TODO(Arthur): optimization: only calculate new reaction rates only for species whose counts have changed
        # propensities can be proportional because only relative values are considered
        # thus, they don't need to be multiplied by volume * Avogadro
        proportional_propensities = np.maximum(0, self.calc_reaction_rates())
        self.log_with_time("submodel: {}; proportional_propensities: {}".format(self.id, proportional_propensities))

        # avoid reactions with inadequate species counts
        # TODO(Arthur): incorporate generalization in the COPASI paper
        enabled_reactions = self.identify_enabled_reactions()
        proportional_propensities = enabled_reactions * proportional_propensities
        total_proportional_propensities = np.sum(proportional_propensities)
        assert not math.isnan(total_proportional_propensities), "total propensities is 'NaN'"   # pragma, no cover
        # TODO(Arthur): generalize: if other submodels can't change terms in the rate laws used by this SSA
        if total_proportional_propensities == 0 and self.get_num_submodels() == 1:
            raise DynamicFrozenSimulationError(self.time, f"Simulation with 1 SSA submodel - '{self.id}' - with total "
                                        f"propensities == 0 won't change species populations")
        return (proportional_propensities, total_proportional_propensities)

    def get_reaction_propensities(self):
        """ Get reaction propensities and handle boundary conditions

        Returns:
            (propensities, total_propensities)
        """
        try:
            propensities, total_propensities = self.determine_reaction_propensities()
        except DynamicFrozenSimulationError:
            # TODO(Arthur): remove this send_event(), which isn't needed
            # schedule event for time = infinity so that other activities like checkpointing continue
            # for the remainder of the simulation
            self.send_event(float('inf'), self, message_types.SsaWait())
            return (None, None)

        if total_propensities == 0:
            self.schedule_SsaWait()

        return (propensities, total_propensities)

    def schedule_SsaWait(self):
        """ Schedule an SsaWait.
        """
        self.send_event(self.ema_of_inter_event_time.get_ema(), self, message_types.SsaWait())
        self.num_SsaWaits += 1
        # TODO(Arthur): avoid arbitrarily slow progress which arises when 1) no reactions
        # are enabled & 2) EMA of the time between ExecuteSsaReaction events is arbitrarily small
        # Solution(s): a) if sequence of SsaWait occurs, increase EMA delay, or b) terminate

    def schedule_ExecuteSsaReaction(self, dt, reaction_index):
        """ Schedule an :obj:`ExecuteSsaReaction` event.

        Args:
            dt (:obj:`float`): simulation delay until the :obj:`ExecuteSsaReaction` will execute
            reaction_index (:obj:`int`): index of the reaction to execute
        """
        self.send_event(dt, self, message_types.ExecuteSsaReaction(reaction_index))

        # maintain EMA of the time between ExecuteSsaReaction events
        self.ema_of_inter_event_time.add_value(dt)

    def schedule_next_SSA_reaction(self):
        """ Schedule the next SSA reaction for this SSA submodel.

        If the sum of propensities is positive, schedule a reaction, otherwise schedule a wait. The
        delay until the next reaction is an exponential sample with mean 1/sum(propensities).

        Method:

        1. calculate propensities
        2. if total propensity == 0:
               schedule a wait equal to the weighted mean inter reaction time
               return
        3. select time of next reaction
        4. select next reaction
        5. schedule the next reaction

        Returns:
            :obj:`float`: the delay until the next SSA reaction, or `None` if no reaction is scheduled
        """
        (propensities, total_propensities) = self.get_reaction_propensities()
        if total_propensities is None or total_propensities == 0:
            return

        # Select time to next reaction from exponential distribution
        dt = self.random_state.exponential(1/total_propensities)

        # schedule next reaction
        # dividing propensities by total_propensities isn't needed -- it wouldn't change relative propensities
        # however, numpy choice() requires valid probabilities for p
        # TODO(Arthur): optimization: consider using Python's random.choice() which accepts weights
        reaction_index = self.random_state.choice(len(propensities), p=propensities/total_propensities)
        self.schedule_ExecuteSsaReaction(dt, reaction_index)
        return dt

    def schedule_next_events(self):
        """ Schedule the next events for this submodel
        """

        # schedule next SSA reaction, or a SSA wait if no reaction is ready to fire
        time_to_next_reaction = self.schedule_next_SSA_reaction()
        return

        # TODO(Arthur): cover after MVP wc_sim done
        pass    # pragma: no cover
        # prefetch into cache
        if not (math.isnan(time_to_next_reaction) or self.access_species_pop is None):  # pragma: no cover
            self.access_species_pop.prefetch(time_to_next_reaction, self.get_species_ids())

    def execute_SSA_reaction(self, reaction_index):
        """ Execute a reaction now.

        Args:
            event (:obj:`Event`): a simulation event
        """
        self.log_with_time("submodel: {} "
            "executing reaction {}".format(self.id, self.reactions[reaction_index].id))
        self.execute_reaction(self.reactions[reaction_index])

    def handle_SsaWait_msg(self, event):
        """ Handle an event containing a :obj:`SsaWait` message

        Args:
            event (:obj:`Event`): a simulation event
        """
        self.log_event(event)
        # TODO(Arthur): generate WARNING(s) if SsaWaits are numerous, or a high fraction of events
        # no reaction to execute
        self.schedule_next_events()

    def handle_GivePopulation_msg(self, event): # pragma: no cover
        # TODO(Arthur): cover after MVP wc_sim done
        """ Handle an event containing a GivePopulation message
        """
        self.log_event(event)

        # population_values is a GivePopulation body attribute
        population_values = event.message.population
        # store population_values in the AccessSpeciesPopulations cache
        self.access_species_pop.species_population_cache.cache_population(self.now,
            population_values)

        self.log_with_time("GivePopulation: {}".format(str(event.message)))

    def handle_ExecuteSsaReaction_msg(self, event):
        """ Handle an event containing a :obj:`ExecuteSsaReaction` message

        Args:
            event (:obj:`Event`): a simulation event
        """
        self.log_event(event)

        reaction_index = event.message.reaction_index

        # if the selected reaction is still enabled execute it, otherwise try to choose another
        if self.enabled_reaction(self.reactions[reaction_index]):
            self.execute_SSA_reaction(reaction_index)

        else:
            propensities, total_propensities = self.get_reaction_propensities()
            if total_propensities is None or total_propensities == 0:
                self.log_with_time("submodel: {}: no reaction to execute".format(self.id))
                return

            else:
                # select a reaction
                reaction_index = self.random_state.choice(len(propensities), p=propensities/total_propensities)
                self.execute_SSA_reaction(reaction_index)

        self.schedule_next_events()

    def log_event(self, event):
        """ Log a SsaSubmodel simulation event.

        Args:
            event (:obj:`Event`): a simulation event
        """
        # TODO(Arthur): provide a similar function at a higher level, in DynamicSubmodel or SimulationObject
        if not self.num_events % config_multialgorithm['ssa_event_logging_spacing']:
            # TODO(Arthur): perhaps log this msg to console
            self.log_with_time("submodel {}, event {}, message type {}".format(self.id, self.num_events,
                event.message.__class__.__name__))

        self.log_with_time("EMA of inter event time: "
            "{:3.2f}; num_events: {}; num_SsaWaits: {}".format(
                self.ema_of_inter_event_time.get_ema(), self.num_events, self.num_SsaWaits))
