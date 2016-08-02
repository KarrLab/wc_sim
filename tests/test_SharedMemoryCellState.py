import unittest
import sys
import re

from Sequential_WC_Simulator.core.LoggingConfig import setup_logger
from Sequential_WC_Simulator.multialgorithm.specie import Specie
from Sequential_WC_Simulator.multialgorithm.shared_cell_state import SharedMemoryCellState


class TestSharedMemoryCellState(unittest.TestCase):

    def setUp(self):
        species = map( lambda x: "specie_{}".format( x ), range( 1, 5 ) )
        self.species = species
        init_populations = dict( zip( species, range( 1, 5 ) ) )
        self.init_populations = init_populations
        self.flux = 1
        init_fluxes = dict( zip( species, [self.flux]*len(species) ) )
        self.a_SM_CellState = SharedMemoryCellState( 'test', init_populations, initial_fluxes=init_fluxes,
            debug=False, log=False)
        self.a_SM_CellState_no_init_flux = SharedMemoryCellState( 'test', init_populations, 
            debug=False, log=False)

    def reusable_assertions(self, the_SM_CellState, flux ):
        # test both discrete and hybrid species
        
        with self.assertRaises(ValueError) as context:
            the_SM_CellState._check_species( 0, 2 )
        self.assertIn( "must be a list", context.exception.message )
        
        with self.assertRaises(ValueError) as context:
            the_SM_CellState._check_species( 0, ['x'] )
        self.assertIn( "Error: request for population of unknown specie(s):", context.exception.message )
        
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
        a_SM_CellState = SharedMemoryCellState( 'test', {} )
        a_SM_CellState.init_cell_state_specie( 's1', 2 )
        self.assertEqual( a_SM_CellState.read( 0, ['s1'] ),  {'s1': 2} )
        with self.assertRaises(ValueError) as context:
            a_SM_CellState.init_cell_state_specie( 's1', 2 )
        self.assertIn( "Error: specie_name 's1' already stored by this SharedMemoryCellState", context.exception.message )
