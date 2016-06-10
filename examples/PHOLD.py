#!/usr/bin/env python

"""
A simple example simulation. Implements PHOLD.
"""

from __future__ import print_function

import random
import argparse
from SequentialSimulator.SimulationObject import (EventQueue, SimulationObject)
from SequentialSimulator.SimulationEngine import SimulationEngine

def obj_name( obj_num ):
    # create object name from index
    return 'obj_{}'.format( obj_num )
    
def obj_index( obj_name ):
    # get object from name
    return int(obj_name.split('_')[1])
    
def exp_delay( ):
    return random.expovariate( 1.0 )
    
class PHOLDsimulationObject(SimulationObject):

    def __init__( self, name, args, debug=False):
        self.debug = debug
        self.args = args
        super(PHOLDsimulationObject, self).__init__( name )

    def handle_event( self, event_list ):
        if self.debug:
            self.print_event_queue( )
        
        # schedule event
        if random.random() < self.args.frac_self_events or self.args.num_PHOLD_procs == 1:
            receiver = self
            print( "{:8.3f}: {} sending to self".format( self.time, self.name ))
        else:
            # send to another process; pick process index in [0,num_PHOLD-2], and increment if self
            index = random.randrange(self.args.num_PHOLD_procs-1)
            if index == obj_index( self.name ):
                index += 1
            receiver = SimulationEngine.simulation_objects[ obj_name( index ) ]
            print( "{:8.3f}: {} sending to {}".format( self.time, self.name, obj_name( index ) ))

        self.send_event( exp_delay(), receiver, 'test event sent by {} at {:6.3f}'.format( 
            self.name, self.time ) )


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
        parser.add_argument( '--debug', '-d', action='store_true', help='Print debug output' )
        parser.add_argument( '--seed', '-s', type=int, help='Random number seed' )
        args = parser.parse_args()
        if args.num_PHOLD_procs < 1:
            parser.error( "Must create at least 1 PHOLD process." )
        if args.seed:
            random.seed( args.seed )
        return args
    
    @staticmethod
    def main():
    
        args = runPHOLD.parseArgs()

        # create simulation objects
        for obj_id in range( args.num_PHOLD_procs ):
            PHOLDsimulationObject( obj_name( obj_id ), args, debug=args.debug ) 
        
        # send initial event messages, to self
        for obj_id in range( args.num_PHOLD_procs ):
            obj = SimulationEngine.simulation_objects[ obj_name( obj_id ) ]
            obj.send_event( exp_delay(), obj, 'init_msg' )

        # run the simulation
        SimulationEngine.simulate( args.end_time )
        

if __name__ == '__main__':
    try:
        runPHOLD.main()
    except KeyboardInterrupt:
        pass
