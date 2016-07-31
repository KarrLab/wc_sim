"""

@author Jonathan Karr, karr@mssm.edu
Created 2016/07/14, built FBA use of Cobra
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
built DES simulation, and converted FBA for DES

"""
    
import sys
import logging
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from cobra import Metabolite as CobraMetabolite
    from cobra import Model as CobraModel
    from cobra import Reaction as CobraReaction

from Sequential_WC_Simulator.core.LoggingConfig import setup_logger
from Sequential_WC_Simulator.core.SimulationObject import (EventQueue, SimulationObject)
from Sequential_WC_Simulator.core.SimulationEngine import MessageTypesRegistry
from Sequential_WC_Simulator.core.utilities import N_AVOGADRO, compare_name_with_class
import Sequential_WC_Simulator.core.utilities as utilities

from Sequential_WC_Simulator.multialgorithm.submodels.submodel import Submodel
from Sequential_WC_Simulator.multialgorithm.MessageTypes import *
from Sequential_WC_Simulator.multialgorithm.model_representation import *

class FbaSubmodel(Submodel):
    """
    FbaSubmodel employs Flus Balance Analysis to predict the reaction fluxes of 
    a set of chemical species in a 'well-mixed' container constrained by maximizing
    biomass increase. 
    
    # TODO(Arthur): expand description

    Attributes:
        random: random object; private PRNG
        time_step: float; time between FBA executions
        metabolismProductionReaction
        exchangedSpecies
        cobraModel
        thermodynamicBounds
        exchangeRateBounds
        defaultFbaBound
        reactionFluxes
        Plus see superclasses.

    Event messages:
        RUN_FBA
        # messages after future enhancement
        ADJUST_POPULATION_BY_CONTINUOUS_MODEL
        GET_POPULATION
        GIVE_POPULATION
    """

    SENT_MESSAGE_TYPES = [ RUN_FBA, 
        ADJUST_POPULATION_BY_CONTINUOUS_MODEL, 
        GET_POPULATION ]

    MessageTypesRegistry.set_sent_message_types( 'FbaSubmodel', SENT_MESSAGE_TYPES )

    # at any time instant, process messages in this order
    MESSAGE_TYPES_BY_PRIORITY = [ 
        GIVE_POPULATION, 
        RUN_FBA ]

    MessageTypesRegistry.set_receiver_priorities( 'FbaSubmodel', MESSAGE_TYPES_BY_PRIORITY )
    
    # COMMENT(Arthur): I want to understand this better.
    
    def __init__(self, model, name, id, private_cell_state, shared_cell_states, 
        reactions, species, time_step, random_seed=None, debug=False, write_plot_output=False ):        
        """Initialize a simple_SSA_submodel object.
        
        # TODO(Arthur): expand description
        
        Args:
            See pydocs of super classes.
            time_step: float; time between FBA executions
            debug: boolean; log debugging output
            write_plot_output: boolean; log output for plotting simulation; simply passed to SimulationObject
        """
        Submodel.__init__( self, model, name, id, private_cell_state, shared_cell_states,
            reactions, species, debug=debug, write_plot_output=write_plot_output )

        self.algorithm = 'FBA'
        self.time_step = time_step

        self.logger_name = "FbaSubmodel_{}".format( name )
        if debug:
            # make a logger
            # TODO(Arthur): eventually control logging when creating SimulationObjects, and pass in the logger
            setup_logger( self.logger_name, level=logging.DEBUG )
            mylog = logging.getLogger(self.logger_name)
            # log initialization data
            mylog.debug( "init: name: {}".format( name ) )
            mylog.debug( "init: id: {}".format( id ) )
            mylog.debug( "init: species: {}".format( str([s.name for s in species]) ) )
            mylog.debug( "init: time_step: {}".format( str(time_step) ) )
        
        self.metabolismProductionReaction = None 
        self.exchangedSpecies = None
        
        self.cobraModel = None
        self.thermodynamicBounds = None
        self.exchangeRateBounds = None
        
        self.defaultFbaBound = 1e15
        self.reactionFluxes = np.zeros(0)

        self.set_up_fba_submodel()
        
    def set_up_fba_submodel( self ):
        """Set up this FBA submodel for simulation.
        
        Setup species fluxes, reaction participant, enzyme counts matrices. 
        Create initial event for this FBA submodel.
        """

        cobraModel = CobraModel(self.id)
        self.cobraModel = cobraModel
            
        # setup metabolites
        cbMets = []
        for species in self.species:
            cbMets.append(CobraMetabolite(id = species.id, name = species.name))
        cobraModel.add_metabolites(cbMets)
        
        # setup reactions
        for rxn in self.reactions:            
            cbRxn = CobraReaction(
                id = rxn.id,
                name = rxn.name,
                lower_bound = -self.defaultFbaBound if rxn.reversible else 0,
                upper_bound =  self.defaultFbaBound,
                objective_coefficient = 1 if rxn.id == 'MetabolismProduction' else 0,
                )
            cobraModel.add_reaction(cbRxn)

            cbMets = {}
            for part in rxn.participants:
                cbMets[part.id] = part.coefficient
            cbRxn.add_metabolites(cbMets)            
        
        # add external exchange reactions
        self.exchangedSpecies = []
        for species in self.species:
            if species.compartment.id == 'e':                
                cbRxn = CobraReaction(
                    id = '%sEx' % species.species.id,
                    name = '%s exchange' % species.species.name,
                    lower_bound = -self.defaultFbaBound,
                    upper_bound =  self.defaultFbaBound,
                    objective_coefficient = 0,
                    )
                cobraModel.add_reaction(cbRxn)
                cbRxn.add_metabolites({species.id: 1})
                
                self.exchangedSpecies.append(ExchangedSpecies(id = species.id, 
                    reactionIndex = cobraModel.reactions.index(cbRxn)))
        
        # add biomass exchange reaction
        cbRxn = CobraReaction(
            id = 'BiomassEx',
            name = 'Biomass exchange',
            lower_bound = 0,
            upper_bound = self.defaultFbaBound,
            objective_coefficient = 0,
            )
        cobraModel.add_reaction(cbRxn)
        cbRxn.add_metabolites({'Biomass[c]': -1})
        
        '''Bounds'''
        # thermodynamic       
        arrayCobraModel = cobraModel.to_array_based_model()
        self.thermodynamicBounds = {
            'lower': np.array(arrayCobraModel.lower_bounds.tolist()),
            'upper': np.array(arrayCobraModel.upper_bounds.tolist()),
            }
        
        # exchange reactions
        carbonExRate = self.getComponentById('carbonExchangeRate', self.model.parameters).value
        nonCarbonExRate = self.getComponentById('nonCarbonExchangeRate', self.model.parameters).value
        self.exchangeRateBounds = {
            'lower': np.full(len(cobraModel.reactions), -np.nan),
            'upper': np.full(len(cobraModel.reactions),  np.nan),
            }

        for exSpecies in self.exchangedSpecies:
            if self.getComponentById(exSpecies.id, self.species).species.containsCarbon():
                self.exchangeRateBounds['lower'][exSpecies.reactionIndex] = -carbonExRate
                self.exchangeRateBounds['upper'][exSpecies.reactionIndex] =  carbonExRate
            else:
                self.exchangeRateBounds['lower'][exSpecies.reactionIndex] = -nonCarbonExRate
                self.exchangeRateBounds['upper'][exSpecies.reactionIndex] =  nonCarbonExRate
            
        '''Setup reactions'''
        self.metabolismProductionReaction = {
            'index': cobraModel.reactions.index(cobraModel.reactions.get_by_id('MetabolismProduction')),
            'reaction': self.getComponentById('MetabolismProduction', self.reactions),
            }

        self.schedule_next_FBA_analysis()
        
    def schedule_next_FBA_analysis(self):
        """Schedule the next analysis by this FBA submodel.
        """
        self.send_event( self.time_step, self, RUN_FBA )

    def calcReactionFluxes(self):
        """calculate growth rate.
        
        Args:
            ?
        
        Returns:
            ?
        """
        self.cobraModel.optimize()
        self.reactionFluxes = self.cobraModel.solution.x
        '''
        print 'self.reactionFluxes', self.reactionFluxes
        print 'lengths'
        print 'reactionFluxes', len(self.reactionFluxes)
        print 'species', len(self.species)
        '''
        self.model.growth = self.reactionFluxes[self.metabolismProductionReaction['index']] #fraction cell/s
        
    def updateMetabolites(self):
        """Update species (metabolite) counts and fluxes.
        """
        # biomass production
        # TODO(Arthur): IMPORTANT; incorporate fluxes!
        adjustments={}
        for participant in self.metabolismProductionReaction['reaction'].participants:
            # directly reference global state
            adjustments[participant.id] = (-self.model.growth * participant.coefficient * self.time_step, 0)
            # was: self.speciesCounts[part.id] -= self.model.growth * part.coefficient * timeStep
        
        # external nutrients
        for exSpecies in self.exchangedSpecies:
            # was: self.speciesCounts[exSpecies.id] += self.reactionFluxes[exSpecies.reactionIndex] * timeStep
            adjustments[exSpecies.id] = (self.reactionFluxes[exSpecies.reactionIndex] * self.time_step, 0)

        cts = self.get_specie_counts()
        for s in self.metabolismProductionReaction['reaction'].participants:    # self.species:
            if cts[s.id]+adjustments[s.id][0] < 0:
                pass
        self.model.the_SharedMemoryCellState.adjust_continuously( self.time, adjustments )
        
    def calcReactionBounds(self):
        """Compute FBA reaction bounds.
        """
        # thermodynamics
        lowerBounds = self.thermodynamicBounds['lower'].copy()
        upperBounds = self.thermodynamicBounds['upper'].copy()
        
        # rate laws
        upperBounds[0:len(self.reactions)] = utilities.nanminimum(
            upperBounds[0:len(self.reactions)], 
            Submodel.calcReactionRates(self.reactions, self.get_specie_concentrations()) 
                * self.model.volume * N_AVOGADRO )
        
        # external nutrients availability
        specie_counts = self.get_specie_counts()
        for exSpecies in self.exchangedSpecies:
            upperBounds[exSpecies.reactionIndex] = max(0, 
                np.minimum(upperBounds[exSpecies.reactionIndex], specie_counts[exSpecies.id]) 
                / self.time_step)
        
        # exchange bounds
        lowerBounds = utilities.nanminimum(lowerBounds, self.model.dryWeight / 3600 * N_AVOGADRO 
            * 1e-3 * self.exchangeRateBounds['lower'])
        upperBounds = utilities.nanminimum(upperBounds, self.model.dryWeight / 3600 * N_AVOGADRO 
            * 1e-3 * self.exchangeRateBounds['upper'])
        
        # return
        arrCbModel = self.cobraModel.to_array_based_model()
        arrCbModel.lower_bounds = lowerBounds
        arrCbModel.upper_bounds = upperBounds

    def handle_event( self, event_list ):
        """Handle a FbaSubmodel simulation event.
        
        In this shared-memory FBA, the only event is RUN_FBA, and event_list should
        always contain one event.
        
        Args:
            event_list: list of event messages to process
        """
        # call handle_event() in class SimulationObject which performs generic tasks on the event list
        SimulationObject.handle_event( self, event_list )
        if not self.num_events % 100:
            print "{:7.1f}: submodel {}, event {}".format( self.time, self.name, self.num_events )

        for event_message in event_list:
            if compare_name_with_class( event_message.event_type, GIVE_POPULATION ):
                
                pass
                # TODO(Arthur): add this functionality; currently, handling RUN_FBA accesses memory directly

                # population_values is a GIVE_POPULATION_body object
                population_values = event_message.event_body

                logging.getLogger( self.logger_name ).debug( "GIVE_POPULATION: {}".format( str(population_values) ) ) 
                # store population_values in some local cache ...
                    
            elif compare_name_with_class( event_message.event_type, RUN_FBA ):
            
                logging.getLogger( self.logger_name ).debug( "{:8.3f}: {} submodel: "
                "executing".format( self.time, self.name ) ) 

                # run the FBA analysis
                self.calcReactionBounds()
                self.calcReactionFluxes()
                self.updateMetabolites()
                self.schedule_next_FBA_analysis()

            else:
                assert False, "Error: the 'if' statement should handle "\
                "event_message.event_type '{}'".format(event_message.event_type)
        
