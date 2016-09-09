'''
Use multiple simulation objects (curently 3) to test the simulator. One object (a 
UniversalSenderReceiverSimulationObject) sends initialization events. Another object, a CellState, manages the
population of one species, 'x'. And another object, a TestSimulationObject, monitors the population of x and 
compares the correct population with the simulated population.

'''

import unittest
import sys
import re
import math

# TODO(Arthur): test the exceptions in these modules
from Sequential_WC_Simulator.core.simulation_object import EventQueue, SimulationObject
from Sequential_WC_Simulator.core.simulation_engine import SimulationEngine, MessageTypesRegistry
from Sequential_WC_Simulator.core.utilities import ReproducibleRandom, compare_name_with_class
from Sequential_WC_Simulator.multialgorithm.message_types import *
from Sequential_WC_Simulator.multialgorithm.cell_state import CellState
from tests.universal_sender_receiver_simulation_object import UniversalSenderReceiverSimulationObject

def parse_population_history( pop_history ):
    # build pop_history_dict from pop_history
    pop_history_dict = {}   # event_type -> event_time -> (Pop_adjust, Flux, Population)
    
    for line in pop_history.split('\n')[2:]:
        line=line.strip()
        values = re.split( '\s+', line )
        for i in range(len(values)):
            t = values[i]
            try:
                t = float( values[i] )
            except:
                pass
            values[i] = t
        (Time, Event, Pop_adjust, Flux, Population) = values
        if not Event in pop_history_dict:
            pop_history_dict[Event] = {}
        pop_history_dict[Event][Time] = (Pop_adjust, Flux, Population)
    return pop_history_dict
    
class TestSimulationObject(SimulationObject):

    SENT_MESSAGE_TYPES = [ GetPopulation ]
    MessageTypesRegistry.set_sent_message_types( 'TestSimulationObject', SENT_MESSAGE_TYPES )

    # at any time instant, process messages in this order
    MESSAGE_TYPES_BY_PRIORITY = [ GivePopulation ]
    MessageTypesRegistry.set_receiver_priorities( 'TestSimulationObject', MESSAGE_TYPES_BY_PRIORITY )
    
    def __init__( self, name, pop_history, specie, debug=False, write_plot_output=False):
        self.debug = debug
        self.pop_history_dict = parse_population_history( pop_history )
        self.specie = specie
        super(TestSimulationObject, self).__init__( name, plot_output=write_plot_output )

    def handle_event( self, event_list ):
        super( TestSimulationObject, self).handle_event( event_list )

        '''
        Test for the population of x, at various times in a sequence of ADJUST_POPULATION_BY_* events. These 
        events are described in pop_history and pop_history_dict and scheduled for the CellState object by
        test_CellState_with_other_SimObj() below. 
        test_CellState_with_other_SimObj() also schedules GetPopulation events sent by TestSimObj, 
        an instance of this TestSimulationObject, at the times in the sequence.
        '''
        
        for event_message in event_list:
            # switch/case on event message type
            if compare_name_with_class( event_message.event_type, GivePopulation ):
            
                # populations is a GivePopulation_body instance
                populations = event_message.event_body
                
                if self.specie in populations.population:
                    if event_message.event_time in self.pop_history_dict['get_pop']:
                        (Pop_adjust, Flux, correct_pop) = self.pop_history_dict['get_pop'][event_message.event_time]
                        # Key point: TestSimulationObject.TestCaseRef is a reference to a unittest.TestCase, 
                        # which is set by TestSimulation.testSimulation( ) below
                        # This test works for any sequence of the stochastic rounding because either round matches
                        # on 2016/07/20 I saw a single, non-reproducible (in over 10,000 attempts) failure of this test; 
                        # the msg will make such a failure be more easily tracked if it reappears
                        TestSimulationObject.TestCaseRef.assertTrue( 
                            populations.population[self.specie] == math.ceil(correct_pop) or
                            populations.population[self.specie] == math.floor(correct_pop), 
                            msg="At event_time {} for specie: '{}': with rounding, the correct population "
                            "should be {} or {}; but the actual population is {}".format( 
                                event_message.event_time, self.specie,
                                math.floor(correct_pop), math.ceil(correct_pop), 
                                populations.population[self.specie] ))
            else:
                raise ValueError( "Error: shouldn't get here - event_message.event_type '{}' should "\
                "be covered in the if statement above".format( event_message.event_type ) )

class TestSimulation(unittest.TestCase):

    def setUp(self):
        SimulationEngine.reset()    
        
    def testSimulation( self ):

        # TODO(Arthur): to test situation when multiple objects have events at minimum simulation time, add recipient column to pop_history:
        specie = 'x'
        # Population dynamics of specie 'x'. See spreadsheet test_pop_history.ods in the current directory.
        # This table is used to both create a test simulation and to verify its execution.
        pop_history_of_x = '''
Time                 Event                Pop_adjust           Flux                 Population
0                    init                 NA                   0                    3
0                    get_pop              NA                   NA                   3
1                    discrete_adjust      1                    NA                   4
1                    get_pop              NA                   NA                   4
2                    continuous_adjust    2                    0.5                  6
2                    get_pop              NA                   NA                   6
3                    get_pop              NA                   NA                   6.5
4                    discrete_adjust      -1                   NA                   6
4                    get_pop              NA                   NA                   6
5                    get_pop              NA                   NA                   6.5'''
        pop_history_dict = parse_population_history( pop_history_of_x )
        (unused_Pop_adjust, init_flux, init_pop) = pop_history_dict['init'][0]
        cs1 = _CellStateMaker.make_CellState( { specie: init_pop }, {specie: init_flux} )


        TestSimObj = TestSimulationObject( 'TestSimObj', pop_history_of_x, specie )
        # give TestSimulationObject.TestCaseRef a reference to this unittest.TestCase:
        TestSimulationObject.TestCaseRef = self

        usr = UniversalSenderReceiverSimulationObject( 'usr1' )
    
        # create initial events
        # AdjustPopulationByDiscreteModel
        for time, (Pop_adjust, unused_Flux, unused_Population) in pop_history_dict['discrete_adjust'].items():
            usr.send_event( time, cs1, AdjustPopulationByDiscreteModel, 
                event_body=AdjustPopulationByDiscreteModel.body( { specie:Pop_adjust } ) )
        
        # AdjustPopulationByContinuousModel
        for time, (Pop_adjust, Flux, unused_Population) in pop_history_dict['continuous_adjust'].items():
            usr.send_event( time, cs1, AdjustPopulationByContinuousModel, 
                event_body=AdjustPopulationByContinuousModel.body( 
                    { specie: Continuous_change( Pop_adjust, Flux ) }
                )
            )

        # GetPopulation
        for time in pop_history_dict['get_pop']:
            TestSimObj.send_event( time, cs1, GetPopulation, 
                event_body=GetPopulation.body( set(['x']) )
            )
        SimulationEngine.simulate( 5.0 )
    
    def testSimulationDefaultPopulation( self ):
        # test code that assumes an unknown specie is initialized with a population of 0

        cs = CellState( _CellStateMaker.get_name(), {} )
        pop_history_of_y = '''
Time                 Event                Pop_adjust           Flux                 Population
1                    discrete_adjust      1                    NA                   1
2                    get_pop              NA                   NA                   1'''
        specie = 'y'
        pop_history_dict = parse_population_history( pop_history_of_y )
        TestSimObj = TestSimulationObject( 'TestSimObj', pop_history_of_y, specie )
        # give TestSimulationObject.TestCaseRef a reference to this unittest.TestCase:
        TestSimulationObject.TestCaseRef = self

        usr = UniversalSenderReceiverSimulationObject( 'usr1' )

        # create initial events
        # AdjustPopulationByDiscreteModel
        for time, (Pop_adjust, unused_Flux, unused_Population) in pop_history_dict['discrete_adjust'].items():
            usr.send_event( time, cs, AdjustPopulationByDiscreteModel, 
                event_body=AdjustPopulationByDiscreteModel.body( { specie:Pop_adjust } ) )

        # GetPopulation
        for time in pop_history_dict['get_pop']:
            TestSimObj.send_event( time, cs, GetPopulation, 
                event_body=GetPopulation.body( set([ specie ]) )
            )
        SimulationEngine.simulate( 5.0 )
        
    def testSimulation_exception( self ):
        # test code that raises exception when AdjustPopulationByContinuousModel message requests population of unknown species

        SimulationEngine.reset()
        cs = CellState( _CellStateMaker.get_name(), {}, debug=True )
        pop_history = '''
Time                 Event                Pop_adjust           Flux                 Population
1                    continuous_adjust      1                    0                   1'''
        specie = 'y'
        pop_history_dict = parse_population_history( pop_history )

        # create AdjustPopulationByContinuousModel event
        usr = UniversalSenderReceiverSimulationObject( 'usr1' )
        for time, (Pop_adjust, Flux, unused_Population) in pop_history_dict['continuous_adjust'].items():
            usr.send_event( time, cs, AdjustPopulationByContinuousModel, 
                event_body=AdjustPopulationByContinuousModel.body( 
                    { specie: Continuous_change( Pop_adjust, Flux ) }
                )
            )

        with self.assertRaises(ValueError) as context:
            SimulationEngine.simulate( 5.0 )
        self.assertIn( "Error: AdjustPopulationByContinuousModel message requests population of unknown species 'y'", 
            str(context.exception) )


class _CellStateMaker(object):
    id = 0
    @staticmethod
    def get_name():
        _CellStateMaker.id += 1
        return "CellState_{:d}".format( _CellStateMaker.id )
     
    @staticmethod
    def make_CellState( pop, init_flux, debug=False, write_plot_output=False, name=None ):
        if name is None:
            name = _CellStateMaker.get_name()
        if debug:
            print("Creating CellState( {}, --population--, debug={}, write_plot_output={} ) ".format(
                name, debug, write_plot_output ))
        ReproducibleRandom.init( seed=123 )
        return CellState( name, pop, initial_fluxes=init_flux, debug=debug, write_plot_output=write_plot_output )
