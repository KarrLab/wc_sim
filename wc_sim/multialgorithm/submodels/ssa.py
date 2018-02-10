'''A sub-model that employs Gillespie's Stochastic Simulation Algorithm (SSA) to model a set of reactions.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-07-14
:Copyright: 2016-2018, Karr Lab
:License: MIT
'''

import sys
import math
import numpy as np
from scipy.constants import Avogadro as N_AVOGADRO
from builtins import super

from scipy.constants import Avogadro
from wc_utils.config.core import ConfigManager
from wc_utils.util.rand import RandomStateManager
from wc_utils.util.misc import isclass_by_name
from wc_utils.util.stats import ExponentialMovingAverage

from wc_sim.core.config import paths as config_paths_core
from wc_sim.core.simulation_object import SimulationObject
from wc_sim.multialgorithm import message_types
from wc_sim.multialgorithm.config import paths as config_paths_multialgorithm
from wc_sim.multialgorithm.submodels.dynamic_submodel import DynamicSubmodel

config_core = ConfigManager(config_paths_core.core).get_config()['wc_sim']['core']
config_multialgorithm = \
    ConfigManager(config_paths_multialgorithm.core).get_config()['wc_sim']['multialgorithm']


class SSASubmodel(DynamicSubmodel):
    """ Use the Stochastic Simulation Algorithm to predict the dynamics of chemical species in a container

    This implementation supports a partition of the species populations into private, locally stored
    populations and shared, remotely stored populations. These are accessed through the ADT provided
    by the `DynamicSubmodel`'s `LocalSpeciesPopulation`. Appropriate optimizations are made if no
    populations are stored remotely.

    # TODO(Arthur): update the rest of this doc string
    Algorithm::

        implement the 'direct' method, except under unusual circumstances.

        determine_reaction_propensities():
            determine reactant concentrations
            determine propensities
            eliminate reactions that are not stoichiometrically enabled *
            return propensities, total_propensities

        schedule_next_event():
            determine_reaction_propensities()
            if total_propensities == 0: **
                schedule_SsaWait() **
            else:
                reaction delay = random sample of exp(1/sum(propensities))
                select and schedule the next reaction

        execute_reaction():
            if scheduled reaction is stoichiometrically enabled:
                execute it
            else: *
                determine_reaction_propensities()
                if total_propensities == 0: **
                    schedule_SsaWait() **
                    return
                else:
                    select and schedule a reaction

            schedule_next_event()

        *   avoid reactions that are not stoichiometrically enabled
        **  2nd order recovery because other submodels can modify shared species counts

    Attributes:
        random: a numpy RandomState() instance object; private PRNG; may be reproducible, as
            determined by the value of config_core['reproducible_seed'] and how the main program,
            MultiAlgorithm, calls ReproducibleRandom.init()
        num_SsaWaits: integer; count of SsaWaits
        ema_of_inter_event_time: an ExponentialMovingAverage; an EMA of the time between
            ExecuteSsaReaction events; when total propensities == 0, ema_of_inter_event_time
            is used as the time between SsaWait events
        Plus see superclasses.

    Event messages:
        ExecuteSsaReaction
        SsaWait
        # messages after future enhancement
        AdjustPopulationByDiscreteSubmodel
        GetPopulation
        GivePopulation
    """

    # message types sent by SSASubmodel
    SENT_MESSAGE_TYPES = [
        message_types.AdjustPopulationByDiscreteSubmodel,
        message_types.ExecuteSsaReaction,
        message_types.GetPopulation,
        message_types.SsaWait]

    # at any time instant, process messages in this order
    MESSAGE_TYPES_BY_PRIORITY = [
        message_types.SsaWait,
        message_types.GivePopulation,
        message_types.ExecuteSsaReaction]

    def __init__(self, id, reactions, species, parameters, dynamic_compartment,
        local_species_population, default_center_of_mass=None):
        """Initialize an SSA submodel object.

        Args:
            Also see pydocs of super classes.
            default_center_of_mass (type): the center_of_mass for the ExponentialMovingAverage
        """
        super().__init__(id, reactions, species, parameters, dynamic_compartment, local_species_population)

        self.num_SsaWaits=0
        # The 'initial_ssa_wait_ema' must be positive, as otherwise an infinite sequence of SsaWait
        # messages will be executed at the start of a simulation if no reactions are enabled
        if default_center_of_mass is None:
            default_center_of_mass = config_core['default_center_of_mass']
        if config_multialgorithm['initial_ssa_wait_ema'] <= 0:
            raise ValueError("'initial_ssa_wait_ema' must be positive to avoid infinite sequence of "
            "SsaWait messages, but it is {}".format(config_multialgorithm['initial_ssa_wait_ema']))
        self.ema_of_inter_event_time=ExponentialMovingAverage(
            config_multialgorithm['initial_ssa_wait_ema'],
            center_of_mass=default_center_of_mass)
        self.random_state = RandomStateManager.instance()

        self.log_with_time("init: name: {}".format(name))
        self.log_with_time("init: species: {}".format(str([s.serialize() for s in species])))

    def send_initial_events(self):
        """Send this SSA submodel's initial events
        """
        self.schedule_next_events()

    def determine_reaction_propensities(self):
        """Determine the current reaction propensities for this submodel.

        Method:
        1. calculate concentrations
        # TODO(Arthur): IMPORTANT: optimization: simply use counts to calculate propensities
        # TODO(Arthur): IMPORTANT: create benchmark & profile data to evaluate possible optimizations
        2. calculate propensities for this submodel
        3. avoid reactions with inadequate specie counts

        Returns:
            reaction (propensities, total_propensities)
        """

        # TODO(Arthur): optimization: only calculate new reaction rates for species whose
        # speciesConcentrations (counts) have changed
        # TODO(Arthur): IMPORTANT: provide volume for model; probably via get_volume(self.model)
        # DC: use DC volume
        propensities = self.model.volume * Avogadro * np.maximum(0,
            DynamicSubmodel.calc_reaction_rates(self.reactions))

        # avoid reactions with inadequate specie counts
        # todo: incorporate generalization in the COPASI paper
        enabled_reactions = self.identify_enabled_reactions(propensities)
        propensities = enabled_reactions * propensities
        total_propensities = np.sum(propensities)
        return (propensities, total_propensities)

    def schedule_SsaWait(self):
        """Schedule an SsaWait.
        """
        self.send_event(self.ema_of_inter_event_time.get_value(), self, message_types.SsaWait)
        self.num_SsaWaits += 1
        # TODO(Arthur): IMPORTANT: avoid arbitrarily slow progress which arises when 1) no reactions
        # are enabled & 2) EMA of the time between ExecuteSsaReaction events is arbitrarily small
        # solution: a) if sequence of SsaWait occurs, increase EMA delay

    def schedule_ExecuteSsaReaction(self, dt, reaction_index):
        """Schedule an ExecuteSsaReaction.
        """
        self.send_event(dt, self,
            message_types.ExecuteSsaReaction, message_types.ExecuteSsaReaction(reaction_index))

        # maintain EMA of the time between ExecuteSsaReaction events
        self.ema_of_inter_event_time.add_value(dt)

    def schedule_next_SSA_reaction(self):
        """Schedule the next SSA reaction for this SSA submodel.

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
            float: the delay until the next SSA reaction, or NaN if no reaction is scheduled
        """
        (propensities, total_propensities) = self.determine_reaction_propensities()
        if total_propensities <= 0:
            self.schedule_SsaWait()
            return float('NaN')

        # Select time to next reaction from exponential distribution
        dt = self.random_state.exponential(1/total_propensities)

        # schedule next reaction
        reaction_index = self.random_state.choice(len(propensities), p = propensities/total_propensities)
        self.schedule_ExecuteSsaReaction(dt, reaction_index)
        return dt

    def schedule_next_events(self):
        """Schedule the next events for this submodel"""

        # schedule next SSA reaction, or a SSA wait if no reaction is ready to fire
        time_to_next_reaction = self.schedule_next_SSA_reaction()

        # prefetch into cache
        if not (math.isnan(time_to_next_reaction) or self.access_species_pop is None):
            self.access_species_pop.prefetch(time_to_next_reaction, self.get_species_ids())

    def execute_SSA_reaction(self, reaction_index):
        """Execute a reaction now.
        """
        self.log_with_time("submodel: {} "
            "executing reaction {}".format(self.name, self.reactions[reaction_index].id))
        self.execute_reaction(self.reactions[reaction_index])

    # todo: restructure
    def handle_event(self, event_list):
        """Handle a SSASubmodel simulation event.

        Args:
            event_list: list of event messages to process
        """
        # call handle_event() in class SimulationObject which performs generic tasks on the event list
        SimulationObject.handle_event(self, event_list)
        if not self.num_events % config_multialgorithm['ssa_event_logging_spacing']:
            # TODO(Arthur): perhaps log this msg to console
            self.log_with_time("submodel {}, event {}".format(self.name, self.num_events))

        for event_message in event_list:
            if isclass_by_name(event_message.event_type, message_types.GivePopulation):

                # population_values is a GivePopulation body attribute
                population_values = event_message.event_body.population
                # store population_values in the AccessSpeciesPopulations cache
                self.access_species_pop.species_population_cache.cache_population(self.now,
                    population_values)

                self.log_with_time("GivePopulation: {}".format(str(event_message.event_body)))

            elif isclass_by_name(event_message.event_type, message_types.ExecuteSsaReaction):

                reaction_index = event_message.event_body.reaction_index

                # if the selected reaction is still enabled execute it, otherwise try to choose another
                if self.enabled_reaction(self.reactions[reaction_index]):
                    self.execute_SSA_reaction(reaction_index)

                else:
                    (propensities, total_propensities) = self.determine_reaction_propensities()
                    if total_propensities == 0:
                        self.log_with_time("submodel: {}: no reaction to execute".format(
                            self.name))
                        self.schedule_SsaWait()
                        continue

                    else:
                        # select a reaction
                        reaction_index = self.random_state.choice(len(propensities),
                            p = propensities/total_propensities)
                        self.execute_SSA_reaction(reaction_index)

                self.schedule_next_events()

            elif isclass_by_name(event_message.event_type, message_types.SsaWait):

                # TODO(Arthur): generate WARNING(s) if SsaWaits are numerous, or a high fraction of events
                # no reaction to execute
                self.schedule_next_events()

            else:
                assert False, "Error: the 'if' statement should handle " \
                "event_message.event_type '{}'".format(event_message.event_type)

        self.log_with_time("EMA of inter event time: "
            "{:3.2f}; num_events: {}; num_SsaWaits: {}".format(
                self.ema_of_inter_event_time.get_value(), self.num_events, self.num_SsaWaits))
