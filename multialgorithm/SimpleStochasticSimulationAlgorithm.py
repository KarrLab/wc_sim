"""
Gillespie's Stochastic Simulation Algorithm (SSA). This is 'simple' in that it
executes explicit chemical reactions, as opposed to rules. I may be able to design a SSA sub-model
simulation object that executes either.
# TODO(Arthur): evaluate this

Created 2016/07/14
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
"""
    
import sys
import logging
from Sequential_WC_Simulator.core.LoggingConfig import setup_logger
from Sequential_WC_Simulator.core.SimulationObject import (EventQueue, SimulationObject)
from Sequential_WC_Simulator.core.SimulationEngine import MessageTypesRegistry
from MessageTypes import (MessageTypes, 
    ADJUST_POPULATION_BY_DISCRETE_MODEL_body, 
    GET_POPULATION_body, 
    GIVE_POPULATION_body, 
    EXECUTE_SSA_REACTION_body )
    
class SimpleStochasticSimulationAlgorithm( SimulationObject ):
    """
    SimpleStochasticSimulationAlgorithm employs Gillespie's Stochastic Simulation Algorithm 
    to predict the dynamics of a set of chemical species in a 'well-mixed' container. 
    
    # TODO(Arthur): expand description

    Attributes:
        private_cell_state: a CellState that stores the copy numbers of the species involved in reactions
            that are modeled only by this SimpleStochasticSimulationAlgorithm instance.
        shared_cell_state: a CellState that stores the copy numbers of the species that are modeled by 
            this SimpleStochasticSimulationAlgorithm instance AND other sub-models.
        reactions: DS TBD; the set of reactions modeled by this SSA
            # TODO(Arthur): INCLUDE the data structure for reactions
        
    Event messages:

        ADJUST_POPULATION_BY_DISCRETE_MODEL
        GET_POPULATION
        GIVE_POPULATION
        EXECUTE_SSA_REACTION

    """
    # TODO(Arthur): spellcheck my comments

    SENT_MESSAGE_TYPES = [ MessageTypes.ADJUST_POPULATION_BY_DISCRETE_MODEL, 
        MessageTypes.EXECUTE_SSA_REACTION ]

    MessageTypesRegistry.set_sent_message_types( 'SimpleStochasticSimulationAlgorithm', SENT_MESSAGE_TYPES )

    # at any time instant, process messages in this order
    MESSAGE_TYPES_BY_PRIORITY = [ 
        MessageTypes.GIVE_POPULATION, 
        MessageTypes.EXECUTE_SSA_REACTION ]

    MessageTypesRegistry.set_receiver_priorities( 'SimpleStochasticSimulationAlgorithm', MESSAGE_TYPES_BY_PRIORITY )

    def __init__( self, name, random_seed=None, debug=False, write_plot_output=False, ):
        """Initialize a SimpleStochasticSimulationAlgorithm object.
        
        Initialize a SimpleStochasticSimulationAlgorithm object.
        
        # TODO(Arthur): expand description
        
        Args:
            name: string; name of this simulation object
            shared_random_seed: hashable object; optional; set to deterministically initialize the random number
                generators used by this SimpleStochasticSimulationAlgorithm
            debug: boolean; log debugging output
            write_plot_output: boolean; log output for plotting simulation; simply passed to SimulationObject

        Raises:
            ??
        """
        super( SimpleStochasticSimulationAlgorithm, self ).__init__( name, plot_output=write_plot_output )
        # TODO(Arthur): init reactions

        self.logger_name = "SimpleStochasticSimulationAlgorithm_{}".format( name )
        if debug:
            # make a logger for this SimpleStochasticSimulationAlgorithm
            # TODO(Arthur): eventually control logging when creating SimulationObjects, and pass in the logger
            setup_logger( self.logger_name, level=logging.DEBUG )
            mylog = logging.getLogger(self.logger_name)
            # write initialization data
            mylog.debug( "random_seed: {}".format( str(random_seed) ) )
            mylog.debug( "write_plot_output: {}".format( str(write_plot_output) ) )
            mylog.debug( "debug: {}".format( str(debug) ) )
        pass
    

    def handle_event( self, event_list ):
        """Handle a SimpleStochasticSimulationAlgorithm simulation event.
        
        # TODO(Arthur): More desc. 
        
        Args:
            event_list: list of event messages to process
            
        Raises:
            ???
        """
        # call handle_event() in class SimulationObject which performs generic tasks on the event list
        super( SimpleStochasticSimulationAlgorithm, self ).handle_event( event_list )
        '''
        
        '''
        for event_message in event_list:
            # switch/case on event message type
            if event_message.event_type == MessageTypes.GIVE_POPULATION:

                # population_values is a GIVE_POPULATION_body object
                population_values = event_message.event_body

                logging.getLogger( self.logger_name ).debug( "GIVE_POPULATION: {}".format( str(population_values) ) ) 
                # store population_values in some cache ...
                pass
                    
            elif event_message.event_type == MessageTypes.EXECUTE_SSA_REACTION:
            
                # reaction is an EXECUTE_SSA_REACTION_body object
                reaction = event_message.event_body
                logging.getLogger( self.logger_name ).debug( "EXECUTE_SSA_REACTION: {}".format( str(reaction) ) ) 
                # execute reaction
                # schedule next reaction ...
                pass

            else:
                assert False, "Shouldn't get here - event_message.event_type should be covered in the "
                "if statement above"

