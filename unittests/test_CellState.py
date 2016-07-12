#!/usr/bin/env python

import unittest
import sys
import re
import os.path as path

from Sequential_WC_Simulator.core.SimulationObject import (EventQueue, SimulationObject)
from Sequential_WC_Simulator.core.SimulationEngine import SimulationEngine
from Sequential_WC_Simulator.multialgorithm.CellState import (Specie, CellState)
from Sequential_WC_Simulator.multialgorithm.MessageTypes import (MessageTypes, ADJUST_POPULATION_BY_DISCRETE_MODEL_body, 
    Continuous_change, ADJUST_POPULATION_BY_CONTINUOUS_MODEL_body, GET_POPULATION_body, GIVE_POPULATION_body)
from UniversalSenderReceiverSimulationObject import UniversalSenderReceiverSimulationObject
from Sequential_WC_Simulator.core.LoggingConfig import LOGGING_ROOT_DIR

class TestCellState(unittest.TestCase):

    def setUp(self):
        SimulationEngine.reset()

    def test_invalid_event_types(self):
    
        # CellState( name, initial_population, debug=False ) 
        pop = dict( zip( 's1 s2 s3'.split(), range(3) ) )
        cs1 = CellState( 'name', pop, debug=False ) 
        # initial events
        with self.assertRaises(ValueError) as context:
            cs1.send_event( 1.0, cs1, 'init_msg1' )
        self.assertIn( "'CellState' simulation objects not registered to send 'init_msg1' messages", 
            context.exception.message )

    id = 0
    @staticmethod
    def get_name():
        TestCellState.id += 1
        return "CellState_{:d}".format( TestCellState.id )
 
    species = 's1 s2 s3'.split()
    pop = dict( zip( species, map( lambda x: x*7, range(3,6) ) ) )
    fluxes = dict( zip( species, [0] * len(species) ) )

    @staticmethod
    def make_CellState( my_pop, my_fluxes, debug=False, write_plot_output=False, name=None, log=False ):
        if not name:
            name = TestCellState.get_name()
        '''
        print "Creating CellState( {}, --population--, debug={}, write_plot_output={} ) ".format(
            name, debug, write_plot_output )
        '''
        return CellState( name, my_pop, initial_fluxes=my_fluxes, debug=debug, 
            write_plot_output=write_plot_output, log=log ) 

    def test_CellState_debugging(self):
        cs1 = TestCellState.make_CellState( TestCellState.pop, None, debug=False )
        usr = UniversalSenderReceiverSimulationObject( 'usr1' )
        usr.send_event( 1.0, cs1, MessageTypes.GET_POPULATION )
        eq = cs1.event_queue_to_str()
        self.assertIn( 'CellState_1 at 0.000', eq )
        self.assertIn( 'creation_time\tevent_time\tsending_object\treceiving_object\tevent_type', eq )
        
        
    def test_simple_CellState(self):
        SimulationEngine.reset()
        cs1 = TestCellState.make_CellState( TestCellState.pop, TestCellState.fluxes, log=True )
        # initial events
        usr = UniversalSenderReceiverSimulationObject( 'usr1' )
        usr.send_event( 1.0, cs1, MessageTypes.ADJUST_POPULATION_BY_DISCRETE_MODEL, 
            event_body=ADJUST_POPULATION_BY_DISCRETE_MODEL_body(
                dict( zip( TestCellState.species, [1]*len( TestCellState.species ) ) )
            )
        )
        t = 2.0
        d = dict( zip( TestCellState.species,
                map( lambda x: Continuous_change(2.0, 1.0), [1]*len( TestCellState.species ) ) ) )

        usr.send_event( t, cs1, MessageTypes.ADJUST_POPULATION_BY_CONTINUOUS_MODEL, 
            event_body=ADJUST_POPULATION_BY_CONTINUOUS_MODEL_body( d ) )
        SimulationEngine.simulate( 5.0 )
        
        text_in_log_s1_by_line = '''.* : s1 : .* #Sim_time\tAdjustment_type\tNew_population\tNew_flux
.* #1.0\tdiscrete_adjustment\t22\t0
.* #2.0\tcontinuous_adjustment\t24.0\t1.0'''
        expected_patterns = text_in_log_s1_by_line.split('\n')
        log_file = path.join( LOGGING_ROOT_DIR, 's1' + '.log' )
        fh = open( log_file, 'r' )
        for pattern in expected_patterns:
            self.assertRegexpMatches( fh.readline().strip(), pattern )
        fh.close()
        
    # TODO(Arthur): important: test simultaneous recept of ADJUST_POPULATION_BY_DISCRETE_MODEL and ADJUST_POPULATION_BY_CONTINUOUS_MODEL

if __name__ == '__main__':
    try:
        unittest.main()
    except KeyboardInterrupt:
        pass

