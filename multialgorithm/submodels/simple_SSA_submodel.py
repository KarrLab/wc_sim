"""
Gillespie's Stochastic Simulation Algorithm (SSA). This is 'simple' in that 
1) it executes explicit chemical reactions, as opposed to rules, and that 
2) it only uses shared_cell_states, and not private_cell_state. 
I may be able to design a SSA sub-model simulation object that executes either.

Created 2016/07/14
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
"""

# TODO(Arthur): IMPORTANT: need debugging logging framework: log entries automatically include
# submodel algorithm and id, time, message type, etc.
    
import sys
import logging
import numpy as np

from Sequential_WC_Simulator.core.LoggingConfig import setup_logger
from Sequential_WC_Simulator.core.SimulationObject import (EventQueue, SimulationObject)
from Sequential_WC_Simulator.core.SimulationEngine import MessageTypesRegistry
from Sequential_WC_Simulator.core.utilities import N_AVOGADRO, ExponentialMovingAverage
from Sequential_WC_Simulator.multialgorithm.submodels.submodel import Submodel

from Sequential_WC_Simulator.multialgorithm.MessageTypes import (MessageTypes, 
    ADJUST_POPULATION_BY_DISCRETE_MODEL_body, 
    GET_POPULATION_body, 
    GIVE_POPULATION_body )
    
class simple_SSA_submodel( Submodel ):
    """
    simple_SSA_submodel employs Gillespie's Stochastic Simulation Algorithm 
    to predict the dynamics of a set of chemical species in a 'well-mixed' container. 
    
    # TODO(Arthur): expand description

    Attributes:
        random: a numpy RandomState() instance object; private PRNG; may be reproducible, as
            determined by the main program, MultiAlgorithm
        num_SSA_WAITs: integer; count of SSA_WAITs
        ema_of_inter_event_time: an ExponentialMovingAverage; an EMA of the time between
            EXECUTE_SSA_REACTION events; when total propensities == 0, ema_of_inter_event_time
            is used as the time between SSA_WAIT events
        Plus see superclasses.

    Event messages:
        EXECUTE_SSA_REACTION
        SSA_WAIT
        # messages after future enhancement
        ADJUST_POPULATION_BY_DISCRETE_MODEL
        GET_POPULATION
        GIVE_POPULATION
    """

    SENT_MESSAGE_TYPES = [ MessageTypes.ADJUST_POPULATION_BY_DISCRETE_MODEL, 
        MessageTypes.EXECUTE_SSA_REACTION, MessageTypes.GET_POPULATION,
        MessageTypes.SSA_WAIT ]

    MessageTypesRegistry.set_sent_message_types( 'simple_SSA_submodel', SENT_MESSAGE_TYPES )

    # at any time instant, process messages in this order
    MESSAGE_TYPES_BY_PRIORITY = [ 
        MessageTypes.SSA_WAIT,
        MessageTypes.GIVE_POPULATION, 
        MessageTypes.EXECUTE_SSA_REACTION ]

    MessageTypesRegistry.set_receiver_priorities( 'simple_SSA_submodel', MESSAGE_TYPES_BY_PRIORITY )

    def __init__( self, model, name, id, private_cell_state, shared_cell_states, 
        reactions, species, numpy_random, debug=False, write_plot_output=False, default_center_of_mass=10 ):
        """Initialize a simple_SSA_submodel object.
        
        # TODO(Arthur): expand description
        
        Args:
            See pydocs of super classes.
            debug: boolean; log debugging output
            write_plot_output: boolean; log output for plotting simulation; simply passed to SimulationObject
            default_center_of_mass: number; the center_of_mass for the ExponentialMovingAverage
                
        """
        Submodel.__init__( self, model, name, id, private_cell_state, shared_cell_states,
            reactions, species, debug=debug, write_plot_output=write_plot_output )

        self.num_SSA_WAITs=0
        self.ema_of_inter_event_time=ExponentialMovingAverage( 0, center_of_mass=default_center_of_mass )
        # TODO(Arthur): IMPORTANT: deploy use of ReproducibleRandom everywhere, as has been done here
        self.random = numpy_random
        self.logger_name = "simple_SSA_submodel_{}".format( name )
        if debug:
            # make a logger for this simple_SSA_submodel
            # TODO(Arthur): eventually control logging when creating SimulationObjects, and pass in the logger
            setup_logger( self.logger_name, level=logging.DEBUG )
            mylog = logging.getLogger(self.logger_name)
            # write initialization data
            mylog.debug( "init: name: {}".format( name ) )
            mylog.debug( "init: id: {}".format( id ) )
            mylog.debug( "init: species: {}".format( str([s.name for s in species]) ) )
            mylog.debug( "init: write_plot_output: {}".format( str(write_plot_output) ) )
            mylog.debug( "init: debug: {}".format( str(debug) ) )
        self.set_up_ssa_submodel()
        
    def set_up_ssa_submodel( self ):
        """Set up this SSA submodel for simulation.
        
        Creates initial event(s) for this SSA submodel.
        """
        # Submodel.set_up_submodel( self )
        self.schedule_next_reaction()

    def determine_reaction_propensities(self):
        """Determine the current reaction propensities for this submodel.
        
        Method:
        1. calculate concentrations
        # TODO(Arthur): optimization: simply use counts to calculate propensities
        2. calculate propensities for this submodel
        3. avoid reactions with inadequate specie counts
        4. if totalPropensities == 0:
            schedule SSA_WAIT
        
        Returns:
            reaction (propensities, totalPropensities)
        """

        # calculate propensities
        # TODO(Arthur): optimization: I understand physicality of concentrations and propensities, 
        # but wasteful to divide by volume & N_AVOGADRO and then multiply by them; stop doing this
        # TODO(Arthur): optimization: only calculate new reaction rates for species whose 
        # speciesConcentrations (counts) have changed
        propensities = np.maximum(0, Submodel.calcReactionRates(self.reactions, self.get_specie_concentrations()) 
            * self.model.volume * N_AVOGADRO)
        # avoid reactions with inadequate specie counts
        enabled_reactions = self.identify_enabled_reactions( propensities ) 
        propensities = enabled_reactions * propensities
        totalPropensities = np.sum(propensities)

        # handle totalPropensities == 0
        if totalPropensities == 0:
            # schedule a wait
            self.send_event( self.ema_of_inter_event_time.get_value(), self, MessageTypes.SSA_WAIT )
            self.num_SSA_WAITs += 1
        return (propensities, totalPropensities)

    def schedule_next_reaction(self):
        """Schedule the execution of the next reaction for this SSA submodel.
        
        If the sum of propensities is positive, the time of the next reaction is an exponential
        sample with mean 1/sum(propensities). Otherwise, wait and try again.
        
        Method:
        1. calculate propensities
            if total propensity == 0:
                give up
        2. select time of next reaction
        """

        (propensities, totalPropensities) = self.determine_reaction_propensities()
        if totalPropensities == 0:
            return

        # Select time to next reaction from exponential distribution
        dt = self.random.exponential(1/totalPropensities)
        
        # schedule next event
        self.send_event( dt, self, MessageTypes.EXECUTE_SSA_REACTION )
        
        # maintain EMA of the time between EXECUTE_SSA_REACTION events
        self.ema_of_inter_event_time.add_value( dt )
        
    def handle_event( self, event_list ):
        """Handle a simple_SSA_submodel simulation event.
        
        In this shared-memory SSA, the only event is EXECUTE_SSA_REACTION, and event_list should
        always contain one event.
        
        Args:
            event_list: list of event messages to process
        """
        # call handle_event() in class SimulationObject which performs generic tasks on the event list
        SimulationObject.handle_event( self, event_list )
        if not self.num_events % 100:
            print "{:7.1f}: submodel {}, event {}".format( self.time, self.name, self.num_events )

        for event_message in event_list:
            if event_message.event_type == MessageTypes.GIVE_POPULATION:
                
                pass
                # TODO(Arthur): add this functionality; currently, handling EXECUTE_SSA_REACTION accesses memory directly

                # population_values is a GIVE_POPULATION_body object
                population_values = event_message.event_body

                logging.getLogger( self.logger_name ).debug( "GIVE_POPULATION: {}".format( str(population_values) ) ) 
                # store population_values in some cache ...
                    
            elif event_message.event_type == MessageTypes.EXECUTE_SSA_REACTION:
            
                # select the reaction
                (propensities, totalPropensities) = self.determine_reaction_propensities()
                if totalPropensities == 0:
                    logging.getLogger( self.logger_name ).debug( "{:8.3f}: {} submodel: "
                    "no reaction to execute".format( self.time, self.name ) ) 
                    return

                iRxn = self.random.choice( len(propensities), p = propensities/totalPropensities)
                logging.getLogger( self.logger_name ).debug( "{:8.2f}: {} submodel: "
                "executing reaction {}".format( self.time, self.name, self.reactions[iRxn].id ) ) 
                self.executeReaction( self.model.the_SharedMemoryCellState, self.reactions[iRxn] )
                self.schedule_next_reaction()
                logging.getLogger( self.logger_name ).debug( "{:8.2f}: "
                "ema_of_inter_event_time: {:3.2f}; num_events: {}; num_SSA_WAITs: {}".format( self.time, 
                self.ema_of_inter_event_time.get_value(), self.num_events, self.num_SSA_WAITs ) ) 

            elif event_message.event_type == MessageTypes.SSA_WAIT:
    
                # TODO(Arthur): generate error for many, or a high fraction of, SSA_WAITs
                # no reaction to execute
                logging.getLogger( self.logger_name ).debug( "SSA_WAIT: at {}".format( self.time ) )
                self.schedule_next_reaction()

            else:
                assert False, "Error: the 'if' statement should handle "
                "event_message.event_type '{}'".format(event_message.event_type)
