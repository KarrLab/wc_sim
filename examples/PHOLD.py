#!/usr/bin/env python

"""
A simple example simulation. Implements PHOLD.
"""

from __future__ import print_function

import random
import argparse
import sys
import datetime

from SequentialSimulator.core.SimulationObject import (EventQueue, SimulationObject)
from SequentialSimulator.core.SimulationEngine import SimulationEngine

def obj_name( obj_num ):
    # create object name from index
    return 'obj_{}'.format( obj_num )
    
def obj_index( obj_name ):
    # get object from name
    return int(obj_name.split('_')[1])
    
def exp_delay( ):
    return random.expovariate( 1.0 )
    
class PHOLDsimulationObject(SimulationObject):

    def __init__( self, name, args, debug=False, write_plot_output=False):
        self.debug = debug
        self.args = args
        super(PHOLDsimulationObject, self).__init__( name, plot_output=write_plot_output )

    def handle_event( self, event_list ):
        """Handle a simulation event."""
        # call handle_event() in class SimulationObject which might produce plotting output or do other things
        super(PHOLDsimulationObject, self).handle_event( event_list )
        
        if self.debug:
            self.print_event_queue( )
        
        # schedule event
        if random.random() < self.args.frac_self_events or self.args.num_PHOLD_procs == 1:
            receiver = self
            if self.debug:
                print( "{:8.3f}: {} sending to self".format( self.time, self.name ))
        else:
            # send to another process; pick process index in [0,num_PHOLD-2], and increment if self
            index = random.randrange(self.args.num_PHOLD_procs-1)
            if index == obj_index( self.name ):
                index += 1
            receiver = SimulationEngine.simulation_objects[ obj_name( index ) ]
            if self.debug:
                print( "{:8.3f}: {} sending to {}".format( self.time, self.name, obj_name( index ) ))

        if receiver == self:
            recipient = 'self'
        else:
            recipient = 'other'
        event_type = "message sent to {}".format( recipient )
        self.send_event( exp_delay(), receiver, event_type )


class runPHOLD(object):

    @staticmethod
    def parseArgs():
        parser = argparse.ArgumentParser( description="Run PHOLD simulation. "
            "Each PHOLD event either schedules an event for 'self' or for some other randomly selected LP, "
            "in either case with an exponentially-distributed time-stamp increment having mean of 1.0. "
            "See R. M. Fujimoto, Performance of Time Warp Under Synthetic Workloads, 1990 Distributed Simulation Conference, pp. 23-28, January 1990 and "
            "Barnes PD, Carothers CD, Jefferson DR, Lapre JM. Warp Speed: Executing Time Warp on 1,966,080 Cores. "
            "SIGSIM-PADS '13. Montreal: Association for Computing Machinery; 2013. p. 327-36. " )
        parser.add_argument( 'num_PHOLD_procs', type=int, help="Number of PHOLD processes to run" )
        parser.add_argument( 'frac_self_events', type=float, help="Fraction of events sent to self" )
        parser.add_argument( 'end_time', type=float, help="End time for the simulation" )
        output_options = parser.add_mutually_exclusive_group()
        output_options.add_argument( '--debug', '-d', action='store_true', help='Print debug output' )
        output_options.add_argument( '--plot', '-p', action='store_true', 
            help='Write plot input for plotSpaceTimeDiagram.py to stdout.' )
        parser.add_argument( '--seed', '-s', type=int, help='Random number seed' )
        args = parser.parse_args()
        if args.num_PHOLD_procs < 1:
            parser.error( "Must create at least 1 PHOLD process." )
        if args.frac_self_events < 0:
            parser.error( "Fraction of events sent to self ({}) should be >= 0.".format( args.frac_self_events ) )
        if 1 < args.frac_self_events:
            parser.error( "Fraction of events sent to self ({}) should be <= 1.".format( args.frac_self_events ) )
        if args.seed:
            random.seed( args.seed )
        return args
    
    @staticmethod
    def main():
    
        args = runPHOLD.parseArgs()
        if args.plot:
            print( '# {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now()) )
            print( "# command line: {}".format( " ".join( sys.argv ) ) )

        # create simulation objects
        for obj_id in range( args.num_PHOLD_procs ):
            PHOLDsimulationObject( obj_name( obj_id ), args, debug=args.debug, write_plot_output=args.plot ) 
        
        # send initial event messages, to self
        for obj_id in range( args.num_PHOLD_procs ):
            obj = SimulationEngine.simulation_objects[ obj_name( obj_id ) ]
            obj.send_event( exp_delay(), obj, 'init_msg' )

        # run the simulation
        event_num = SimulationEngine.simulate( args.end_time )
        sys.stderr.write( "Executed {} events.\n".format( event_num ) )
       

if __name__ == '__main__':
    try:
        runPHOLD.main()
    except KeyboardInterrupt:
        pass
