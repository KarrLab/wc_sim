#!/usr/bin/env python

import sys
import re

from SequentialSimulator.core.SimulationObject import (EventQueue, SimulationObject)
from SequentialSimulator.core.SimulationEngine import SimulationEngine
from SequentialSimulator.multialgorithm.MessageTypes import (MessageTypes, ADJUST_POPULATION_BY_DISCRETE_MODEL_body, 
    Continuous_change, ADJUST_POPULATION_BY_CONTINUOUS_MODEL_body, GET_POPULATION_body, GIVE_POPULATION_body)
from SequentialSimulator.multialgorithm.CellState import (Specie, CellState)

# population dynamics of specie 'x'
# See spreadsheet test_pop_history.ods
pop_history = '''
Time	Event	Pop_adjust	Flux	Population
0	init	NA	NA	3
0	get_pop	NA	NA	3
1	discrete_adjust	1	NA	4
1	get_pop	NA	NA	4
2	continuous_adjust	2	0.5	6
2	get_pop	NA	NA	6
3	get_pop	NA	NA	6.5
4	discrete_adjust	-1	NA	6
4	get_pop	NA	NA	6
5	get_pop	NA	NA	6.5'''
pop_history_dict = {}   # event_type -> time -> (Pop_adjust, Flux, Population)
for line in pop_history.split('\n')[2:]:
    line=line.strip()
    values = re.split( '\s', line )
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

    def __init__( self, name, debug=False, write_plot_output=False):
        self.debug = debug
        super(TestSimulationObject, self).__init__( name, plot_output=write_plot_output )

    def handle_event( self, event_list ):
        super( TestSimulationObject, self).handle_event( event_list )
        # print events
        if self.debug:
            self.print_event_queue( )

        '''
        Test for the population of x, at various times in a sequence of ADJUST_POPULATION_BY_* events. These 
        events are described in pop_history_dict and scheduled for the CellState object by test_CellState_with_other_SimObj() 
        below. 
        test_CellState_with_other_SimObj() also schedules GET_POPULATION events sent by otherSimObj, an instance of this
        TestSimulationObject, at the times in the sequence.
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
                        print 'is', populations.population[specie], 'should_be', correct_pop
                        assert populations.population[specie] == correct_pop
            else:
                print "Shouldn't get here - event_message.event_type should be covered in the "
                "if statement above"

class test_CellStateMultiObj( object ):

    id = 0
    @staticmethod
    def get_name():
        test_CellStateMultiObj.id += 1
        return "CellState_{:d}".format( test_CellStateMultiObj.id )
     
    @staticmethod
    def test_CellState_with_other_SimObj():
        
        specie = 'x'
        (unused_Pop_adjust, unused_Flux, init_pop) = pop_history_dict['init'][0]
        cs1 = test_CellStateMultiObj.make_CellState( { specie: init_pop, 'y': 3 },
            write_plot_output=False, debug=False )
        otherSimObj = TestSimulationObject( 'otherSimObj', write_plot_output=False )
    
        # events
        # ADJUST_POPULATION_BY_DISCRETE_MODEL
        for time, (Pop_adjust, unused_Flux, unused_Population) in pop_history_dict['discrete_adjust'].items():
            cs1.send_event( time, cs1, MessageTypes.ADJUST_POPULATION_BY_DISCRETE_MODEL, 
                event_body=ADJUST_POPULATION_BY_DISCRETE_MODEL_body( { specie:Pop_adjust } ) )
        
        # ADJUST_POPULATION_BY_CONTINUOUS_MODEL
        for time, (Pop_adjust, Flux, unused_Population) in pop_history_dict['continuous_adjust'].items():
            cs1.send_event( time, cs1, MessageTypes.ADJUST_POPULATION_BY_CONTINUOUS_MODEL, 
                event_body=ADJUST_POPULATION_BY_CONTINUOUS_MODEL_body( 
                    { specie: Continuous_change( Pop_adjust, Flux ) }
                )
            )

        # GET_POPULATION
        for time in pop_history_dict['get_pop'].keys():
            otherSimObj.send_event( time, cs1, MessageTypes.GET_POPULATION, 
                event_body=GET_POPULATION_body( set(['x']) )
            )
        SimulationEngine.simulate( 5.0 )
    
    @staticmethod
    def make_CellState( pop, debug=False, write_plot_output=False, name=None ):
        if not name:
            name = test_CellStateMultiObj.get_name()
        print "Creating CellState( {}, --population--, debug={}, write_plot_output={} ) ".format(
            name, debug, write_plot_output )
        return CellState( name, pop, debug=debug, write_plot_output=write_plot_output ) 
    
if __name__ == '__main__':
    try:
        test_CellStateMultiObj.test_CellState_with_other_SimObj()
    except KeyboardInterrupt:
        pass

