#!/usr/bin/env python

'''
Use multiple simulation objects (curently 3) to test the simulator. One object (a 
UniversalSenderReceiverSimulationObject) sends initialization events. Another object, a CellState, manages the
population of one species, 'x'. And another object, a TestSimulationObject, monitors the population of x and 
compares the correct population with the simulated population.

'''

import unittest
import sys
import re

# TODO(Arthur): test the exceptions in these modules
from SequentialSimulator.core.SimulationObject import (EventQueue, SimulationObject)
from SequentialSimulator.core.SimulationEngine import SimulationEngine
from SequentialSimulator.multialgorithm.MessageTypes import (MessageTypes, ADJUST_POPULATION_BY_DISCRETE_MODEL_body, 
    Continuous_change, ADJUST_POPULATION_BY_CONTINUOUS_MODEL_body, GET_POPULATION_body, GIVE_POPULATION_body)
from SequentialSimulator.multialgorithm.CellState import (Specie, CellState)
from UniversalSenderReceiverSimulationObject import UniversalSenderReceiverSimulationObject

# population dynamics of specie 'x'
# See spreadsheet test_pop_history.ods in the current directory.
pop_history = '''
Time                 Event                Pop_adjust           Flux                 Population
0                    init                 NA                   NA                   3
0                    get_pop              NA                   NA                   3
1                    discrete_adjust      1                    NA                   4
1                    get_pop              NA                   NA                   4
2                    continuous_adjust    2                    0.5                  6
2                    get_pop              NA                   NA                   6
3                    get_pop              NA                   NA                   6.5
4                    discrete_adjust      -1                   NA                   6
4                    get_pop              NA                   NA                   6
5                    get_pop              NA                   NA                   6.5'''
pop_history_dict = {}   # event_type -> event_time -> (Pop_adjust, Flux, Population)
# build pop_history_dict from pop_history
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

class TestSimulationObject(SimulationObject):

    SENT_MESSAGE_TYPES = [ MessageTypes.GET_POPULATION ]
    MessageTypes.set_sent_message_types( 'TestSimulationObject', SENT_MESSAGE_TYPES )

    # at any time instant, process messages in this order
    MESSAGE_TYPES_BY_PRIORITY = [ MessageTypes.GIVE_POPULATION ]
    MessageTypes.set_receiver_priorities( 'TestSimulationObject', MESSAGE_TYPES_BY_PRIORITY )

    def __init__( self, name, debug=False, write_plot_output=False):
        self.debug = debug
        super(TestSimulationObject, self).__init__( name, plot_output=write_plot_output )

    def handle_event( self, event_list ):
        super( TestSimulationObject, self).handle_event( event_list )

        '''
        Test for the population of x, at various times in a sequence of ADJUST_POPULATION_BY_* events. These 
        events are described in pop_history and pop_history_dict and scheduled for the CellState object by
        test_CellState_with_other_SimObj() below. 
        test_CellState_with_other_SimObj() also schedules GET_POPULATION events sent by TestSimObj, 
        an instance of this TestSimulationObject, at the times in the sequence.
        '''
        specie = 'x'
        
        for event_message in event_list:
            # switch/case on event message type
            if event_message.event_type == MessageTypes.GIVE_POPULATION:
            
                # populations is a GIVE_POPULATION_body instance
                populations = event_message.event_body
                
                if specie in populations.population:
                    if event_message.event_time in pop_history_dict['get_pop']:
                        (Pop_adjust, Flux, correct_pop) = pop_history_dict['get_pop'][event_message.event_time]
                        # Key point: TestSimulationObject.TestCaseRef is a reference to a unittest.TestCase, 
                        # which is set by TestSimulation.testSimulation( ) below
                        TestSimulationObject.TestCaseRef.assertEqual( populations.population[specie], correct_pop )
            else:
                print "Shouldn't get here - event_message.event_type should be covered in the "
                "if statement above"

class TestSimulation(unittest.TestCase):

    def setUp(self):
        SimulationEngine.reset()

    id = 0
    @staticmethod
    def get_name():
        TestSimulation.id += 1
        return "CellState_{:d}".format( TestSimulation.id )
     
    @staticmethod
    def make_CellState( pop, debug=False, write_plot_output=False, name=None ):
        if not name:
            name = TestSimulation.get_name()
        if debug:
            print "Creating CellState( {}, --population--, debug={}, write_plot_output={} ) ".format(
                name, debug, write_plot_output )
        return CellState( name, pop, debug=debug, write_plot_output=write_plot_output )
        
    def testSimulation( self ):

        specie = 'x'
        (unused_Pop_adjust, unused_Flux, init_pop) = pop_history_dict['init'][0]
        cs1 = TestSimulation.make_CellState( { specie: init_pop } )
        TestSimObj = TestSimulationObject( 'TestSimObj' )
        # give TestSimulationObject.TestCaseRef a reference to this unittest.TestCase:
        TestSimulationObject.TestCaseRef = self

        usr = UniversalSenderReceiverSimulationObject( 'usr1' )
    
        # create initial events
        # ADJUST_POPULATION_BY_DISCRETE_MODEL
        for time, (Pop_adjust, unused_Flux, unused_Population) in pop_history_dict['discrete_adjust'].items():
            usr.send_event( time, cs1, MessageTypes.ADJUST_POPULATION_BY_DISCRETE_MODEL, 
                event_body=ADJUST_POPULATION_BY_DISCRETE_MODEL_body( { specie:Pop_adjust } ) )
        
        # ADJUST_POPULATION_BY_CONTINUOUS_MODEL
        for time, (Pop_adjust, Flux, unused_Population) in pop_history_dict['continuous_adjust'].items():
            usr.send_event( time, cs1, MessageTypes.ADJUST_POPULATION_BY_CONTINUOUS_MODEL, 
                event_body=ADJUST_POPULATION_BY_CONTINUOUS_MODEL_body( 
                    { specie: Continuous_change( Pop_adjust, Flux ) }
                )
            )

        # GET_POPULATION
        for time in pop_history_dict['get_pop'].keys():
            TestSimObj.send_event( time, cs1, MessageTypes.GET_POPULATION, 
                event_body=GET_POPULATION_body( set(['x']) )
            )
        SimulationEngine.simulate( 5.0 )
    
if __name__ == '__main__':
    try:
        unittest.main()
    except KeyboardInterrupt:
        pass
