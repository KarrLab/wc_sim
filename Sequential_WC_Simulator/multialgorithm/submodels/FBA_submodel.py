"""

@author Jonathan Karr, karr@mssm.edu
Created 2016/07/14, built FBA use of Cobra
@author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
built DES simulation, and converted FBA for DES

"""
    
import sys
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from cobra import Metabolite as CobraMetabolite
    from cobra import Model as CobraModel
    from cobra import Reaction as CobraReaction

from scipy.constants import Avogadro
from wc_utils.util.misc import isclass_by_name

from Sequential_WC_Simulator.core.simulation_object import (EventQueue, SimulationObject)
from Sequential_WC_Simulator.core.simulation_engine import MessageTypesRegistry

from Sequential_WC_Simulator.multialgorithm.submodels.submodel import Submodel
from Sequential_WC_Simulator.multialgorithm.message_types import *
from Sequential_WC_Simulator.multialgorithm.model_representation import *

class FbaSubmodel(Submodel):
    """
    FbaSubmodel employs Flus Balance Analysis to predict the reaction fluxes of 
    a set of chemical species in a 'well-mixed' container constrained by maximizing
    biomass increase. 
    
    # TODO(Arthur): expand description
    # TODO(Arthur): change variable names to lower_with_under style

    Attributes:
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
        RunFBA
        # messages after future enhancement
        AdjustPopulationByContinuousModel
        GetPopulation
        GivePopulation
    """

    SENT_MESSAGE_TYPES = [ RunFBA, 
        AdjustPopulationByContinuousModel, 
        GetPopulation ]

    MessageTypesRegistry.set_sent_message_types( 'FbaSubmodel', SENT_MESSAGE_TYPES )

    # at any time instant, process messages in this order
    MESSAGE_TYPES_BY_PRIORITY = [ 
        GivePopulation, 
        RunFBA ]

    MessageTypesRegistry.set_receiver_priorities( 'FbaSubmodel', MESSAGE_TYPES_BY_PRIORITY )


    def __init__(self, model, name, id, private_cell_state, shared_cell_states, 
        reactions, species, time_step ):
        """Initialize a FbaSubmodel object.
        
        # TODO(Arthur): expand description
        
        Args:
            See pydocs of super classes.
            time_step: float; time between FBA executions
        """
        Submodel.__init__( self, model, name, id, private_cell_state, shared_cell_states,
            reactions, species )

        self.algorithm = 'FBA'
        self.time_step = time_step

        # log initialization data
        self.log_with_time( "init: name: {}".format( name ) )
        self.log_with_time( "init: id: {}".format( id ) )
        self.log_with_time( "init: species: {}".format( str([s.name for s in species]) ) )
        self.log_with_time( "init: time_step: {}".format( str(time_step) ) )
        
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
                    id = '{}Ex'.format(species.species.id),
                    name = '{} exchange'.format(species.species.name),
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
        self.send_event( self.time_step, self, RunFBA )

    def calcReactionFluxes(self):
        """calculate growth rate.
        """

        '''
        assertion because 
            arrCbModel = self.cobraModel.to_array_based_model()
            arrCbModel.lower_bounds = lowerBounds
            arrCbModel.upper_bounds = upperBounds
        was assigning a list to the bound for each reaction
        '''
        for r in self.cobraModel.reactions:
            assert ( type(r.lower_bound) is np.float64 and type(r.upper_bound) is np.float64)

        self.cobraModel.optimize()
        self.reactionFluxes = self.cobraModel.solution.x
        self.model.growth = self.reactionFluxes[self.metabolismProductionReaction['index']] #fraction cell/s
        
    def updateMetabolites(self):
        """Update species (metabolite) counts and fluxes.
        """
        # biomass production
        adjustments={}
        local_fluxes={}
        for participant in self.metabolismProductionReaction['reaction'].participants:
            # was: self.speciesCounts[part.id] -= self.model.growth * part.coefficient * timeStep
            adjustments[participant.id] = (-self.model.growth * participant.coefficient * self.time_step,
                -self.model.growth * participant.coefficient )
        
        # external nutrients
        for exSpecies in self.exchangedSpecies:
            # was: self.speciesCounts[exSpecies.id] += self.reactionFluxes[exSpecies.reactionIndex] * timeStep
            adjustments[exSpecies.id] = (self.reactionFluxes[exSpecies.reactionIndex] * self.time_step,
                self.reactionFluxes[exSpecies.reactionIndex])
        
        self.model.the_SharedMemoryCellState.adjust_continuously( self.time, adjustments )
        
        
    def calcReactionBounds(self):
        """Compute FBA reaction bounds.
        """
        # thermodynamics
        lowerBounds = self.thermodynamicBounds['lower'].copy()
        upperBounds = self.thermodynamicBounds['upper'].copy()
        
        # rate laws
        upperBounds[0:len(self.reactions)] = np.fmin(
            upperBounds[0:len(self.reactions)], 
            Submodel.calcReactionRates(self.reactions, self.get_specie_concentrations()) 
                * self.model.volume * Avogadro )
        
        # external nutrients availability
        specie_counts = self.get_specie_counts()
        for exSpecies in self.exchangedSpecies:
            upperBounds[exSpecies.reactionIndex] = max(0, 
                np.minimum(upperBounds[exSpecies.reactionIndex], specie_counts[exSpecies.id]) 
                / self.time_step)
        
        # exchange bounds
        lowerBounds = np.fmin(lowerBounds, self.model.dryWeight / 3600 * Avogadro 
            * 1e-3 * self.exchangeRateBounds['lower'])
        upperBounds = np.fmin(upperBounds, self.model.dryWeight / 3600 * Avogadro 
            * 1e-3 * self.exchangeRateBounds['upper'])
        
        for i_rxn, rxn in enumerate(self.cobraModel.reactions):
            rxn.lower_bound = lowerBounds[i_rxn]
            rxn.upper_bound = upperBounds[i_rxn]


    def handle_event( self, event_list ):
        """Handle a FbaSubmodel simulation event.
        
        In this shared-memory FBA, the only event is RunFBA, and event_list should
        always contain one event.
        
        Args:
            event_list: list of event messages to process
        """
        # call handle_event() in class SimulationObject which performs generic tasks on the event list
        SimulationObject.handle_event( self, event_list )
        if not self.num_events % 100:
            print( "{:7.1f}: submodel {}, event {}".format( self.time, self.name, self.num_events ) )

        for event_message in event_list:
            if isclass_by_name( event_message.event_type, GivePopulation ):
                
                pass
                # TODO(Arthur): add this functionality; currently, handling RunFBA accesses memory directly

                # population_values is a GivePopulation body attribute
                population_values = event_message.event_body.population

                self.log_with_time( "GivePopulation: {}".format( str(event_message.event_body) ) )
                # store population_values in some local cache ...
                    
            elif isclass_by_name( event_message.event_type, RunFBA ):
            
                self.log_with_time( "submodel '{}' executing".format( self.name ) )

                # run the FBA analysis
                self.calcReactionBounds()
                self.calcReactionFluxes()
                self.updateMetabolites()
                self.schedule_next_FBA_analysis()

            else:
                assert False, "Error: the 'if' statement should handle "\
                "event_message.event_type '{}'".format(event_message.event_type)
        
