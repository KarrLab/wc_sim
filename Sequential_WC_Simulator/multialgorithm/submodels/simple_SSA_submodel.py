"""
Gillespie's Stochastic Simulation Algorithm (SSA). This is 'simple' in that 
1) it executes explicit chemical reactions, as opposed to rules, and that 
2) it only uses shared_cell_states, and not private_cell_state. 
I may be able to design a SSA sub-model simulation object that executes either.

Created 2016/07/14
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
"""
    
import sys
import logging
import numpy as np

from Sequential_WC_Simulator.core.LoggingConfig import setup_logger
from Sequential_WC_Simulator.core.SimulationObject import (EventQueue, SimulationObject)
from Sequential_WC_Simulator.core.SimulationEngine import MessageTypesRegistry
from Sequential_WC_Simulator.core.utilities import (N_AVOGADRO, ExponentialMovingAverage, 
    compare_name_with_class, ReproducibleRandom, dict_2_key_sorted_str)
from Sequential_WC_Simulator.multialgorithm.config import WC_SimulatorConfig
from Sequential_WC_Simulator.multialgorithm.submodels.submodel import Submodel
from Sequential_WC_Simulator.multialgorithm.MessageTypes import *
    
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
                schedule_SSA_WAIT() **
            else:
                reaction delay = random sample of exp( 1/sum(propensities) )
                select and schedule the next reaction

        execute_reaction():
            if scheduled reaction is stoichiometrically enabled:
                execute it
            else: *
                determine_reaction_propensities()
                if total_propensities == 0: **
                    schedule_SSA_WAIT() **
                    return
                else:
                    select and schedule a reaction
                    
            schedule_next_event()
        
        *   avoid reactions that are not stoichiometrically enabled
        **  2nd order recovery because other submodels can modify shared species counts

    Attributes:
        random: a numpy RandomState() instance object; private PRNG; may be reproducible, as
            determined by ReproducibleRandomthe main program, MultiAlgorithm
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

    SENT_MESSAGE_TYPES = [ ADJUST_POPULATION_BY_DISCRETE_MODEL, 
        EXECUTE_SSA_REACTION, GET_POPULATION, SSA_WAIT ]

    MessageTypesRegistry.set_sent_message_types( 'simple_SSA_submodel', SENT_MESSAGE_TYPES )

    # at any time instant, process messages in this order
    MESSAGE_TYPES_BY_PRIORITY = [ 
        SSA_WAIT,
        GIVE_POPULATION, 
        EXECUTE_SSA_REACTION ]

    MessageTypesRegistry.set_receiver_priorities( 'simple_SSA_submodel', MESSAGE_TYPES_BY_PRIORITY )

    def __init__( self, model, name, id, private_cell_state, shared_cell_states, 
        reactions, species, debug=False, write_plot_output=False, default_center_of_mass=10 ):
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
        # TODO(Arthur): use private_cell_state & shared_cell_states, or get rid of them

        self.num_SSA_WAITs=0
        self.ema_of_inter_event_time=ExponentialMovingAverage( 0, center_of_mass=default_center_of_mass )
        # TODO(Arthur): IMPORTANT: deploy use of ReproducibleRandom everywhere, as has been done here
        self.numpy_random = ReproducibleRandom.get_numpy_random()
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
        # but wasteful to divide by volume & N_AVOGADRO and then multiply by them; stop doing this
        # TODO(Arthur): optimization: only calculate new reaction rates for species whose 
        # speciesConcentrations (counts) have changed
        propensities = np.maximum(0, Submodel.calcReactionRates(self.reactions, self.get_specie_concentrations()) 
            * self.model.volume * N_AVOGADRO)
        # avoid reactions with inadequate specie counts
        enabled_reactions = self.identify_enabled_reactions( propensities ) 
        propensities = enabled_reactions * propensities
        total_propensities = np.sum(propensities)
        return (propensities, total_propensities)

    def schedule_SSA_WAIT(self):
        """Schedule an SSA_WAIT. 
        """
        self.send_event( self.ema_of_inter_event_time.get_value(), self, SSA_WAIT )
        self.num_SSA_WAITs += 1

    def schedule_EXECUTE_SSA_REACTION(self, dt, reaction_index):
        """Schedule an EXECUTE_SSA_REACTION. 
        """
        self.send_event( dt, self,
            EXECUTE_SSA_REACTION, EXECUTE_SSA_REACTION.body(reaction_index) )
        
        # maintain EMA of the time between EXECUTE_SSA_REACTION events
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
            self.schedule_SSA_WAIT()
            return

        # Select time to next reaction from exponential distribution
        dt = self.numpy_random.exponential(1/total_propensities)
        
        # schedule next reaction
        reaction_index = self.numpy_random.choice( len(propensities), p = propensities/total_propensities)
        self.schedule_EXECUTE_SSA_REACTION( dt, reaction_index )
        
    def execute_SSA_reaction(self, reaction_index):
        """Execute a reaction now. 
        """
        logging.getLogger( self.logger_name ).debug( "{:8.2f}: {} submodel: "
            "executing reaction {}".format( self.time, self.name, self.reactions[reaction_index].id ) ) 
        self.executeReaction( self.model.the_SharedMemoryCellState, self.reactions[reaction_index] )

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
            print( "{:7.1f}: submodel {}, event {}".format( self.time, self.name, self.num_events ) )

        for event_message in event_list:
            if compare_name_with_class( event_message.event_type, GIVE_POPULATION ):
                
                continue
                # TODO(Arthur): add this functionality; currently, handling accessing memory directly

                # population_values is a GIVE_POPULATION body attribute
                population_values = event_message.event_body.population

                logging.getLogger( self.logger_name ).debug( "GIVE_POPULATION: {}".format( 
                    str(event_message.event_body) ) ) 
                # store population_values in some cache ...
                    
            elif compare_name_with_class( event_message.event_type, EXECUTE_SSA_REACTION ):
            
                reaction_index = event_message.event_body.reaction_index

                # if the selected reaction is still enabled execute it, otherwise try to choose another
                if self.enabled_reaction( self.reactions[reaction_index] ):
                    self.execute_SSA_reaction( reaction_index )
                
                else:
                    (propensities, total_propensities) = self.determine_reaction_propensities()
                    if total_propensities == 0:
                        logging.getLogger( self.logger_name ).debug( "{:8.3f}: {} submodel: "
                        "no reaction to execute".format( self.time, self.name ) ) 
                        self.schedule_SSA_WAIT()
                        continue
                        
                    else:

                        # select a reaction
                        reaction_index = self.numpy_random.choice( len(propensities), p = propensities/total_propensities)
                        self.execute_SSA_reaction( reaction_index )

                self.schedule_next_event()

            elif compare_name_with_class( event_message.event_type, SSA_WAIT ):
    
                # TODO(Arthur): generate error for many, or a high fraction of, SSA_WAITs
                # no reaction to execute
                self.schedule_next_event()

            else:
                assert False, "Error: the 'if' statement should handle " \
                "event_message.event_type '{}'".format(event_message.event_type)

        logging.getLogger( self.logger_name ).debug( "{:8.2f}: "
        "ema_of_inter_event_time: {:3.2f}; num_events: {}; num_SSA_WAITs: {}".format( self.time, 
        self.ema_of_inter_event_time.get_value(), self.num_events, self.num_SSA_WAITs ) ) 
