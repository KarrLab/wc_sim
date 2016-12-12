'''
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-12-01
:Copyright: 2016, Karr Lab
:License: MIT
'''
import unittest
import sys
import re

from wc_utils.util.rand import RandomStateManager

from wc_sim.multialgorithm.specie import Specie
from wc_sim.multialgorithm.local_species_population import LocalSpeciesPopulation

class TestLocalSpeciesPopulation(unittest.TestCase):

    def setUp(self):
        RandomStateManager.initialize( seed=123 )

        species_nums = range( 1, 5 )
        species = list( map( lambda x: "specie_{}".format( x ), species_nums ) )
        self.species = species
        self.init_populations = dict( zip( species, species_nums ) )
        self.flux = 1
        init_fluxes = dict( zip( species, [self.flux]*len(species) ) )
        self.init_fluxes = init_fluxes
        self.local_species_pop = LocalSpeciesPopulation( None, 'test', self.init_populations,
            initial_fluxes=init_fluxes )
        self.local_species_pop_no_init_flux = LocalSpeciesPopulation( None, 'test', self.init_populations )

    def reusable_assertions(self, the_local_species_pop, flux ):
        # test both discrete and hybrid species

        with self.assertRaises(ValueError) as context:
            the_local_species_pop._check_species( 0, 2 )
        self.assertIn( "must be a set", str(context.exception) )

        with self.assertRaises(ValueError) as context:
            the_local_species_pop._check_species( 0, {'x'} )
        self.assertIn( "Error: request for population of unknown specie(s):", str(context.exception) )

        self.assertEqual( the_local_species_pop.read( 0, set(self.species) ), self.init_populations )
        first_specie = self.species[0]
        the_local_species_pop.adjust_discretely( 0, { first_specie: 3 } )
        self.assertEqual( the_local_species_pop.read( 0, {first_specie} ),  {first_specie: 4} )

        if flux:
            # counts: 1 initialization + 3 discrete adjustment + 2*flux:
            self.assertEqual( the_local_species_pop.read( 2, {first_specie} ),  {first_specie: 4+2*flux} )
            the_local_species_pop.adjust_continuously( 2, {first_specie:( 9, 0) } )
            # counts: 1 initialization + 3 discrete adjustment + 9 continuous adjustment + 0 flux = 13:
            self.assertEqual( the_local_species_pop.read( 2, {first_specie} ),  {first_specie: 13} )

    def test_read_one(self):
        self.assertEqual(self.local_species_pop.read_one(1,'specie_3'), 4)
        with self.assertRaises(ValueError) as context:
            self.local_species_pop.read_one(2, 's1')
        self.assertIn( "request for population of unknown specie(s): 's1'", str(context.exception) )
        with self.assertRaises(ValueError) as context:
            self.local_species_pop.read_one(0, 'specie_3')
        self.assertIn( "earlier access of specie(s): ['specie_3']", str(context.exception) )

    def test_discrete_and_hyrid(self):

        for (local_species_pop, flux) in [(self.local_species_pop,self.flux),
            (self.local_species_pop_no_init_flux,None)]:
            self.reusable_assertions(local_species_pop, flux)

    def test_init(self):
        an_LSP = LocalSpeciesPopulation( None, 'test', {}, retain_history=False )
        an_LSP.init_cell_state_specie( 's1', 2 )
        self.assertEqual( an_LSP.read( 0, {'s1'} ),  {'s1': 2} )
        with self.assertRaises(ValueError) as context:
            an_LSP.init_cell_state_specie( 's1', 2 )
        self.assertIn( "Error: specie_id 's1' already stored by this LocalSpeciesPopulation",
            str(context.exception) )
        with self.assertRaises(ValueError) as context:
            an_LSP.report_history()
        self.assertIn( "Error: history not recorded", str(context.exception) )
        with self.assertRaises(ValueError) as context:
            an_LSP.history_debug()
        self.assertIn( "Error: history not recorded", str(context.exception) )

    def test_history(self):
        """Test population history."""
        an_LSP_recording_history = LocalSpeciesPopulation( None, 'test',
            self.init_populations, None, retain_history=False )
        with self.assertRaises(ValueError) as context:
            an_LSP_recording_history.report_history()
        self.assertIn("Error: history not recorded", str(context.exception))
        with self.assertRaises(ValueError) as context:
            an_LSP_recording_history.history_debug()
        self.assertIn("Error: history not recorded", str(context.exception))

        an_LSP_recording_history = LocalSpeciesPopulation( None, 'test',
            self.init_populations, None, retain_history=True )
        self.assertTrue( an_LSP_recording_history._recording_history() )
        next_time = 1
        first_specie = self.species[0]
        an_LSP_recording_history.read( next_time, {first_specie})
        an_LSP_recording_history._record_history()
        with self.assertRaises(ValueError) as context:
            an_LSP_recording_history._record_history()
        self.assertIn( "time of previous _record_history() (1) not less than current time",
            str(context.exception) )

        history = an_LSP_recording_history.report_history()
        self.assertEqual( history['time'],  [0,next_time] )
        first_specie_history = [1.0,1.0]
        self.assertEqual( history['population'][first_specie], first_specie_history )

        self.assertIn(
            '\t'.join( map( lambda x:str(x), [ first_specie, 2, ] + first_specie_history ) ),
            an_LSP_recording_history.history_debug() )
