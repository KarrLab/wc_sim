"""
Gillespie's Stochastic Simulation Algorithm (SSA). This is 'simple' in that
1) it executes explicit chemical reactions, as opposed to rules, and that
2) it only uses shared_cell_states, and not private_cell_state.
I may be able to design a SSA sub-model simulation object that executes either.

Created 2016/07/14
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
"""

import sys
import numpy as np

from scipy.constants import Avogadro
from wc_utils.config.core import ConfigManager
from wc_utils.util.rand import RandomStateManager
from wc_utils.util.misc import isclass_by_name
from wc_utils.util.stats import ExponentialMovingAverage

from wc_sim.core.config import paths as config_paths_core
from wc_sim.core.simulation_object import (EventQueue, SimulationObject)
from wc_sim.core.simulation_engine import MessageTypesRegistry
from wc_sim.multialgorithm.config import paths as config_paths_multialgorithm
from wc_sim.multialgorithm.submodels.submodel import Submodel
from wc_sim.multialgorithm.message_types import *

config_core = ConfigManager(config_paths_core.core).get_config()['wc_sim']['core']
config_multialgorithm = ConfigManager(config_paths_multialgorithm.core).get_config()['wc_sim']['multialgorithm']


class simple_SSA_submodel( Submodel ):
    """
    simple_SSA_submodel employs Gillespie's Stochastic Simulation Algorithm
    to predict the dynamics of a set of chemical species in a 'well-mixed' container.

    algorithm:::

        implement the 'direct' method, except under unusual circumstances.

        determine_reaction_propensities():
            determine reactant concentrations
            determine propensities
            eliminate reactions that are not stoichiometrically enabled *
            return propensities, total_propensities

        schedule_next_event():
            determine_reaction_propensities()
            if total_propensities == 0: **
                schedule_SSAWait() **
            else:
                reaction delay = random sample of exp( 1/sum(propensities) )
                select and schedule the next reaction

        execute_reaction():
            if scheduled reaction is stoichiometrically enabled:
                execute it
            else: *
                determine_reaction_propensities()
                if total_propensities == 0: **
                    schedule_SSAWait() **
                    return
                else:
                    select and schedule a reaction

            schedule_next_event()

        *   avoid reactions that are not stoichiometrically enabled
        **  2nd order recovery because other submodels can modify shared species counts

    Attributes:
        random: a numpy RandomState() instance object; private PRNG; may be reproducible, as
            determined by the value of config_core['reproducible_seed'] and how the main program, MultiAlgorithm,
            calls ReproducibleRandom.init()
        num_SSAWaits: integer; count of SSAWaits
        ema_of_inter_event_time: an ExponentialMovingAverage; an EMA of the time between
            ExecuteSSAReaction events; when total propensities == 0, ema_of_inter_event_time
            is used as the time between SSAWait events
        Plus see superclasses.

    Event messages:
        ExecuteSSAReaction
        SSAWait
        # messages after future enhancement
        AdjustPopulationByDiscreteModel
        GetPopulation
        GivePopulation
    """

    SENT_MESSAGE_TYPES = [ AdjustPopulationByDiscreteModel,
        ExecuteSSAReaction, GetPopulation, SSAWait ]

    MessageTypesRegistry.set_sent_message_types( 'simple_SSA_submodel', SENT_MESSAGE_TYPES )

    # at any time instant, process messages in this order
    MESSAGE_TYPES_BY_PRIORITY = [
        SSAWait,
        GivePopulation,
        ExecuteSSAReaction ]

    MessageTypesRegistry.set_receiver_priorities( 'simple_SSA_submodel', MESSAGE_TYPES_BY_PRIORITY )

    def __init__( self, model, name, id, private_cell_state, shared_cell_states,
        reactions, species, default_center_of_mass=config_core['default_center_of_mass']):
        """Initialize a simple_SSA_submodel object.

        # TODO(Arthur): expand description

        Args:
            See pydocs of super classes.
            default_center_of_mass: number; the center_of_mass for the ExponentialMovingAverage

        """
        Submodel.__init__( self, model, name, id, private_cell_state, shared_cell_states,
            reactions, species )
        # TODO(Arthur): use private_cell_state & shared_cell_states, or get rid of them

        self.num_SSAWaits=0
        # INITIAL_SSA_WAIT_EMA should be positive, as otherwise an infinite loop of SSAWait messages
        # will form at the start of a simulation if no reactions are enabled
        self.ema_of_inter_event_time=ExponentialMovingAverage(config_multialgorithm['initial_ssa_wait_ema'],
            center_of_mass=default_center_of_mass )
        self.random_state = RandomStateManager.instance()

        self.log_with_time( "init: name: {}".format( name ) )
        self.log_with_time( "init: id: {}".format( id ) )
        self.log_with_time( "init: species: {}".format( str([s.name for s in species]) ) )

        self.set_up_ssa_submodel()

    def set_up_ssa_submodel( self ):
        """Set up this SSA submodel for simulation.

        Creates initial event(s) for this SSA submodel.
        """
        # Submodel.set_up_submodel( self )
        self.schedule_next_event()

    def determine_reaction_propensities(self):
        """Determine the current reaction propensities for this submodel.

        Method:
        1. calculate concentrations
        # TODO(Arthur): optimization: simply use counts to calculate propensities
        2. calculate propensities for this submodel
        3. avoid reactions with inadequate specie counts

        Returns:
            reaction (propensities, total_propensities)
        """

        # TODO(Arthur): optimization: I understand physicality of concentrations and propensities,
        # but wasteful to divide by volume & Avogadro and then multiply by them; stop doing this
        # TODO(Arthur): optimization: only calculate new reaction rates for species whose
        # speciesConcentrations (counts) have changed
        propensities = np.maximum(0, Submodel.calcReactionRates(self.reactions, self.get_specie_concentrations())
            * self.model.volume * Avogadro)
        # avoid reactions with inadequate specie counts
        enabled_reactions = self.identify_enabled_reactions( propensities )
        propensities = enabled_reactions * propensities
        total_propensities = np.sum(propensities)
        return (propensities, total_propensities)

    def schedule_SSAWait(self):
        """Schedule an SSAWait.
        """
        self.send_event( self.ema_of_inter_event_time.get_value(), self, SSAWait )
        self.num_SSAWaits += 1
        # TODO(Arthur): IMPORTANT: avoid possible infinite loop / infinitely slow progress
        # arises when 1) no reactions are enabled & 2) EMA of the time between ExecuteSSAReaction events
        # is very small (initializing to 0 may be bad)

    def schedule_ExecuteSSAReaction(self, dt, reaction_index):
        """Schedule an ExecuteSSAReaction.
        """
        self.send_event( dt, self,
            ExecuteSSAReaction, ExecuteSSAReaction.body(reaction_index) )

        # maintain EMA of the time between ExecuteSSAReaction events
        self.ema_of_inter_event_time.add_value( dt )

    def schedule_next_event(self):
        """Schedule the next event for this SSA submodel.

        If the sum of propensities is positive, schedule a reaction, otherwise schedule
        a wait. The delay until the next reaction is an exponential sample with mean 1/sum(propensities).

        Method:

        1. calculate propensities
        2. if total propensity == 0:

               | schedule a wait equal to the weighted mean inter reaction time
               | return

        3. select time of next reaction
        4. select next reaction
        5. schedule the next reaction
        """

        (propensities, total_propensities) = self.determine_reaction_propensities()
        if total_propensities <= 0:
            self.schedule_SSAWait()
            return

        # Select time to next reaction from exponential distribution
        dt = self.random_state.exponential(1/total_propensities)

        # schedule next reaction
        reaction_index = self.random_state.choice( len(propensities), p = propensities/total_propensities)
        self.schedule_ExecuteSSAReaction( dt, reaction_index )

    def execute_SSA_reaction(self, reaction_index):
        """Execute a reaction now.
        """
        self.log_with_time( "submodel: {} "
            "executing reaction {}".format( self.name, self.reactions[reaction_index].id ) )
        self.executeReaction( self.model.the_SharedMemoryCellState, self.reactions[reaction_index] )

    def handle_event( self, event_list ):
        """Handle a simple_SSA_submodel simulation event.

        In this shared-memory SSA, the only event is ExecuteSSAReaction, and event_list should
        always contain one event.

        Args:
            event_list: list of event messages to process
        """
        # call handle_event() in class SimulationObject which performs generic tasks on the event list
        SimulationObject.handle_event( self, event_list )
        if not self.num_events % config_multialgorithm['ssa_event_logging_spacing']:
            # TODO(Arthur): perhaps log this msg to console
            self.log_with_time( "submodel {}, event {}".format( self.name, self.num_events ) )

        for event_message in event_list:
            if isclass_by_name( event_message.event_type, GivePopulation ):

                continue
                # TODO(Arthur): add this functionality; currently, handling accessing memory directly

                # population_values is a GivePopulation body attribute
                population_values = event_message.event_body.population

                self.log_with_time( "GivePopulation: {}".format( str(event_message.event_body) ) )
                # store population_values in some cache ...

            elif isclass_by_name( event_message.event_type, ExecuteSSAReaction ):

                reaction_index = event_message.event_body.reaction_index

                # if the selected reaction is still enabled execute it, otherwise try to choose another
                if self.enabled_reaction( self.reactions[reaction_index] ):
                    self.execute_SSA_reaction( reaction_index )

                else:
                    (propensities, total_propensities) = self.determine_reaction_propensities()
                    if total_propensities == 0:
                        self.log_with_time( "submodel: {}: no reaction to execute".format(
                            self.name ) )
                        self.schedule_SSAWait()
                        continue

                    else:

                        # select a reaction
                        reaction_index = self.random_state.choice( len(propensities),
                            p = propensities/total_propensities)
                        self.execute_SSA_reaction( reaction_index )

                self.schedule_next_event()

            elif isclass_by_name( event_message.event_type, SSAWait ):

                # TODO(Arthur): generate error(s) if SSAWaits are numerous, or a high fraction of events
                # no reaction to execute
                self.schedule_next_event()

            else:
                assert False, "Error: the 'if' statement should handle " \
                "event_message.event_type '{}'".format(event_message.event_type)

        self.log_with_time( "EMA of inter event time: "
            "{:3.2f}; num_events: {}; num_SSAWaits: {}".format(
                self.ema_of_inter_event_time.get_value(), self.num_events, self.num_SSAWaits ) )