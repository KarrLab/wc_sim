import unittest
import sys
import re
import os.path as path
import json

from Sequential_WC_Simulator.core.simulation_object import EventQueue, SimulationObject
from Sequential_WC_Simulator.core.simulation_engine import SimulationEngine, MessageTypesRegistry
from Sequential_WC_Simulator.core.utilities import ReproducibleRandom
from Sequential_WC_Simulator.multialgorithm.cell_state import CellState
from Sequential_WC_Simulator.multialgorithm.message_types import *
from universal_sender_receiver_simulation_object import UniversalSenderReceiverSimulationObject
from Sequential_WC_Simulator.core.logging_config import LOGGING_ROOT_DIR

class TestCellState(unittest.TestCase):

    def setUp(self):
        SimulationEngine.reset()
        ReproducibleRandom.init()

    def test_invalid_event_types(self):
    
        # CellState( name, initial_population, debug=False ) 
        pop = dict( zip( 's1 s2 s3'.split(), range(3) ) )
        cs1 = CellState( 'name', pop, debug=False ) 
        # initial events
        with self.assertRaises(ValueError) as context:
            cs1.send_event( 1.0, cs1, 'init_msg1' )
        self.assertIn( "'CellState' simulation objects not registered to send 'init_msg1' messages", 
            str(context.exception) )

    def test_CellState_debugging(self):
        cs1 = _CellStateMaker.make_CellState( _CellStateMaker.pop, None, debug=False )
        usr = UniversalSenderReceiverSimulationObject( 'usr1' )
        usr.send_event( 1.0, cs1, GET_POPULATION )
        eq = cs1.event_queue_to_str()
        self.assertIn( 'CellState_1 at 0.000', eq )
        self.assertIn( 'creation_time\tevent_time\tsending_object\treceiving_object\tevent_type', eq )
        
        
    def test_CellState_species_logging(self):
        SimulationEngine.reset()
        cs1 = _CellStateMaker.make_CellState( _CellStateMaker.pop, _CellStateMaker.fluxes, log=True )
        # initial events
        usr = UniversalSenderReceiverSimulationObject( 'usr1' )
        usr.send_event( 1.0, cs1, ADJUST_POPULATION_BY_DISCRETE_MODEL, 
            event_body=ADJUST_POPULATION_BY_DISCRETE_MODEL.body(
                dict( zip( _CellStateMaker.species, [1]*len( _CellStateMaker.species ) ) )
            )
        )
        t = 2.0
        d = dict( zip( _CellStateMaker.species,
                map( lambda x: Continuous_change(2.0, 1.0), [1]*len( _CellStateMaker.species ) ) ) )

        usr.send_event( t, cs1, ADJUST_POPULATION_BY_CONTINUOUS_MODEL, 
            event_body=ADJUST_POPULATION_BY_CONTINUOUS_MODEL.body( d ) )
        SimulationEngine.simulate( 5.0 )
        
        text_in_log_s1_by_line = '''.*; s1; .* #Sim_time\tAdjustment_type\tNew_population\tNew_flux
.* #0.0\tinitial_state\t21\t0
.* #1.0\tdiscrete_adjustment\t22\t0
.* #2.0\tcontinuous_adjustment\t24.0\t1.0'''
        expected_patterns = text_in_log_s1_by_line.split('\n')
        log_file = path.join( LOGGING_ROOT_DIR, 's1' + '.log' )
        fh = open( log_file, 'r' )
        for pattern in expected_patterns:
            self.assertRegexpMatches( fh.readline().strip(), pattern )
        fh.close()
        
    def test_CellState_logging(self):
        SimulationEngine.reset()
        cs1 = _CellStateMaker.make_CellState( _CellStateMaker.pop, _CellStateMaker.fluxes, debug=True )
        # TODO(Arthur): avoid copying this code
        # initial events
        usr = UniversalSenderReceiverSimulationObject( 'usr1' )
        usr.send_event( 1.0, cs1, ADJUST_POPULATION_BY_DISCRETE_MODEL, 
            event_body=ADJUST_POPULATION_BY_DISCRETE_MODEL.body(
                dict( zip( _CellStateMaker.species, [1]*len( _CellStateMaker.species ) ) )
            )
        )
        t = 2.0
        d = dict( zip( _CellStateMaker.species,
                map( lambda x: Continuous_change(2.0, 1.0), [1]*len( _CellStateMaker.species ) ) ) )

        usr.send_event( t, cs1, ADJUST_POPULATION_BY_CONTINUOUS_MODEL, 
            event_body=ADJUST_POPULATION_BY_CONTINUOUS_MODEL.body( d ) )
        SimulationEngine.simulate( 5.0 )
        
        expected_initial_population = {'s3': 35, 's2': 28, 's1': 21}
        expected_initial_fluxes = {'s3': 0, 's2': 0, 's1': 0}
        expected_discrete_species_change = {'s3':1.0, 's2':1.0, 's1':1.0}
        expected_continuous_species_change = {'s3':[2.0, 1.0], 's2':[2.0, 1.0], 's1':[2.0, 1.0]}

        text_in_log_CellState_CellState_2_by_line = '''.*initial_population: {'s\d': \d\d, 's\d': \d\d, 's\d': \d\d}
.*initial_fluxes: {'s\d': 0, 's\d': 0, 's\d': 0}
.*write_plot_output: False
.*#debug: True
.*log: False
.*CellState_2 at 1.000
.*creation_time\tevent_time\tsending_object\treceiving_object\tevent_type
.*0.000\t   2.000\tusr1\tCellState_2\tADJUST_POPULATION_BY_CONTINUOUS_MODEL
.*
.*ADJUST_POPULATION_BY_DISCRETE_MODEL: specie:change: s\d:1.0, s\d:1.0, s\d:1.0
.*CellState_2 at 2.000
.*Empty event queue
.*ADJUST_POPULATION_BY_CONTINUOUS_MODEL: specie:\(change,flux\): s\d:\(\d.0,\d.0\), s\d:\(\d.0,\d.0\), s\d:\(\d.0,\d.0\)'''
        expected_patterns = text_in_log_CellState_CellState_2_by_line.split('\n')
        log_file = path.join( LOGGING_ROOT_DIR, cs1.logger_name + '.log' )
        fh = open( log_file, 'r' )
        for pattern in expected_patterns:
            self.assertRegexpMatches( fh.readline().strip(), pattern )

        
        fh.seek(0)
        log_text = fh.read()

        result = re.search("initial_population: ({'s\d': \d\d, 's\d': \d\d, 's\d': \d\d})", log_text, re.MULTILINE)
        self.assertEqual(expected_initial_population, json.loads(result.group(1).replace("'", '"')))

        result = re.search("initial_fluxes: ({'s\d': 0, 's\d': 0, 's\d': 0})", log_text, re.MULTILINE)
        self.assertEqual(expected_initial_fluxes, json.loads(result.group(1).replace("'", '"')))

        result = re.search("ADJUST_POPULATION_BY_DISCRETE_MODEL: specie:change: (s\d:1.0, s\d:1.0, s\d:1.0)", log_text, re.MULTILINE)
        self.assertEqual(expected_discrete_species_change, json.loads(
            '{"' + result.group(1).replace(":", '":').replace(', ', ', "') + '}'
            ))

        result = re.search("ADJUST_POPULATION_BY_CONTINUOUS_MODEL: specie:\(change,flux\): (s\d:\(\d.0,\d.0\), s\d:\(\d.0,\d.0\), s\d:\(\d.0,\d.0\))", log_text, re.MULTILINE)
        self.assertEqual(expected_continuous_species_change,    json.loads(
            '{"' 
            + result.group(1).replace(':', '":').replace(', ', ', "').replace('(', '[').replace(')', ']')
            + '}'
            ))

        fh.close()

        
    # TODO(Arthur): important: test simultaneous recept of ADJUST_POPULATION_BY_DISCRETE_MODEL and ADJUST_POPULATION_BY_CONTINUOUS_MODEL


class _CellStateMaker(object):
    id = 0
    @staticmethod
    def get_name():
        _CellStateMaker.id += 1
        return "CellState_{:d}".format( _CellStateMaker.id )
 
    species = 's1 s2 s3'.split()
    pop = dict( zip( species, map( lambda x: x*7, range(3,6) ) ) )
    fluxes = dict( zip( species, [0] * len(species) ) )

    @staticmethod
    def make_CellState( my_pop, my_fluxes, debug=False, write_plot_output=False, name=None, log=False ):
        if not name:
            name = _CellStateMaker.get_name()
        return CellState( name, my_pop, initial_fluxes=my_fluxes, debug=debug, 
            write_plot_output=write_plot_output, log=log ) 
