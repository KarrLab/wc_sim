import unittest
import sys
import re

from wc_utils.util.rand import RandomStateManager

from wc_sim.multialgorithm.specie import Specie
from wc_sim.multialgorithm.shared_memory_cell_state import SharedMemoryCellState

class TestSharedMemoryCellState(unittest.TestCase):

    def setUp(self):
        RandomStateManager.initialize( seed=123 )
        
        species = list( map( lambda x: "specie_{}".format( x ), range( 1, 5 ) ) )
        self.species = species
        init_populations = dict( zip( species, range( 1, 5 ) ) )
        self.init_populations = init_populations
        self.flux = 1
        init_fluxes = dict( zip( species, [self.flux]*len(species) ) )
        self.init_fluxes = init_fluxes
        self.a_SM_CellState = SharedMemoryCellState( None, 'test', init_populations, 
            initial_fluxes=init_fluxes )
        self.a_SM_CellState_no_init_flux = SharedMemoryCellState( None, 'test', init_populations )

    def reusable_assertions(self, the_SM_CellState, flux ):
        # test both discrete and hybrid species
        
        with self.assertRaises(ValueError) as context:
            the_SM_CellState._check_species( 0, 2 )
        self.assertIn( "must be a list", str(context.exception) )
        
        with self.assertRaises(ValueError) as context:
            the_SM_CellState._check_species( 0, ['x'] )
        self.assertIn( "Error: request for population of unknown specie(s):", str(context.exception) )
        
        self.assertEqual( the_SM_CellState.read( 0, self.species ), self.init_populations )
        first_specie = self.species[0]
        the_SM_CellState.adjust_discretely( 0, { first_specie: 3 } )
        self.assertEqual( the_SM_CellState.read( 0, [first_specie] ),  {first_specie: 4} )

        if flux:
            # counts: 1 initialization + 3 discrete adjustment + 2*flux:
            self.assertEqual( the_SM_CellState.read( 2, [first_specie] ),  {first_specie: 4+2*flux} )
            the_SM_CellState.adjust_continuously( 2, {first_specie:( 9, 0) } )
            # counts: 1 initialization + 3 discrete adjustment + 9 continuous adjustment + 0 flux = 13:
            self.assertEqual( the_SM_CellState.read( 2, [first_specie] ),  {first_specie: 13} )

    def test_discrete_and_hyrid(self):
    
        for (SM_CellState, flux) in [(self.a_SM_CellState,self.flux), (self.a_SM_CellState_no_init_flux,None)]:
            self.reusable_assertions( SM_CellState, flux )
            
    def test_init(self):
        a_SM_CellState = SharedMemoryCellState( None, 'test', {}, retain_history=False )
        a_SM_CellState.init_cell_state_specie( 's1', 2 )
        self.assertEqual( a_SM_CellState.read( 0, ['s1'] ),  {'s1': 2} )
        with self.assertRaises(ValueError) as context:
            a_SM_CellState.init_cell_state_specie( 's1', 2 )
        self.assertIn( "Error: specie_name 's1' already stored by this SharedMemoryCellState", str(context.exception) )
        with self.assertRaises(ValueError) as context:
            a_SM_CellState.report_history()
        self.assertIn( "Error: history not recorded", str(context.exception) )
        with self.assertRaises(ValueError) as context:
            a_SM_CellState.history_debug()
        self.assertIn( "Error: history not recorded", str(context.exception) )

    def test_history(self):
        """Test population history."""
        a_SM_CellState_recording_history = SharedMemoryCellState( None, 'test', 
            self.init_populations, None, retain_history=True )
        self.assertTrue( a_SM_CellState_recording_history._recording_history() )
        next_time = 1
        first_specie = self.species[0]
        a_SM_CellState_recording_history.read( next_time, [first_specie])
        a_SM_CellState_recording_history._record_history()
        with self.assertRaises(ValueError) as context:
            a_SM_CellState_recording_history._record_history()
        self.assertIn( "time of previous _record_history() (1) not less than current time",
            str(context.exception) )
        
        history = a_SM_CellState_recording_history.report_history()
        self.assertEqual( history['time'],  [0,next_time] )
        first_specie_history = [1,1]
        self.assertEqual( history['population'][first_specie], first_specie_history )
        
        self.assertIn( 
            '\t'.join( map( lambda x:str(x), [ first_specie, 2, ] + first_specie_history ) ),
            a_SM_CellState_recording_history.history_debug() )
