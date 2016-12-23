'''Maintain the population of a set of species in a simulation object.

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016_11_29
:Copyright: 2016, Karr Lab
:License: MIT
'''

import sys

from wc_utils.util.misc import isclass_by_name as check_class
from wc_sim.core.simulation_object import SimulationObject
from wc_sim.core.simulation_engine import MessageTypesRegistry
from wc_sim.multialgorithm import message_types
from wc_sim.multialgorithm.local_species_population import LocalSpeciesPopulation
from wc_sim.multialgorithm.multialgorithm_errors import SpeciesPopulationError

# logging
from wc_sim.multialgorithm.debug_logs import logs as debug_logs
debug_log = debug_logs.get_log( 'wc.debug.file' )

class SpeciesPopSimObject(LocalSpeciesPopulation,SimulationObject):
    '''Maintain the population of a set of species in a simulation object that can be parallelized.

    A whole-cell PDES must run multiple submodels in parallel. These share cell state, such as
    species populations, by accessing shared simulation objects. A SpeciesPopSimObject provides that
    functionality by wrapping a LocalSpeciesPopulation as a DES object accessed only by
    simulation event messages.
    '''

    def __init__(self, name, initial_population, initial_fluxes=None, retain_history=True ):
        '''Initialize a SpeciesPopSimObject object.

        Initialize a SpeciesPopSimObject object. Initialize its base classes.

        Args:
            name (str): the name of the simulation object and local species population object.

        For remaining args and exceptions, see `__init__()` documentation for
        `wc_sim.multialgorithm.SimulationObject` and `wc_sim.multialgorithm.LocalSpeciesPopulation`.
        (Perhaps Sphinx can automate this, but the documentation is unclear.)
        '''
        SimulationObject.__init__(self, name)
        LocalSpeciesPopulation.__init__(self, None, name, initial_population, initial_fluxes)

    def handle_event(self, event_list):
        '''Handle a SpeciesPopSimObject simulation event.

        Process event messages for this SpeciesPopSimObject.

        Args:
            event_list (:obj:`list` of :obj:`wc_sim.core.Event`): list of Events to process.

        Raises:
            SpeciesPopulationError: if a GetPopulation message requests the population of an
                unknown species.
            SpeciesPopulationError: if an AdjustPopulationByContinuousModel event acts on a
                non-existent species.
        '''
        # call handle_event() in class SimulationObject to perform generic tasks on the event list
        super(SpeciesPopSimObject, self).handle_event(event_list)
        for event_message in event_list:

            # switch/case on event message type
            if check_class(event_message.event_type, message_types.AdjustPopulationByDiscreteModel):
                population_change = event_message.event_body.population_change
                self.adjust_discretely( self.time, population_change )

            elif check_class(event_message.event_type, message_types.AdjustPopulationByContinuousModel):
                population_change = event_message.event_body.population_change
                self.adjust_continuously( self.time, population_change )

            elif check_class(event_message.event_type, message_types.GetPopulation):
                species = event_message.event_body.species
                self.send_event( 0, event_message.sending_object, message_types.GivePopulation,
                    event_body=self.read( self.time, species) )

            else:
                assert False, "Shouldn't get here - {} should be covered"\
                    " in the if statement above".format(event_message.event_type)

# Register sent message types
SENT_MESSAGE_TYPES = [ message_types.GivePopulation ]
MessageTypesRegistry.set_sent_message_types(SpeciesPopSimObject, SENT_MESSAGE_TYPES)

# At any time instant, process messages in this order
MESSAGE_TYPES_BY_PRIORITY = [
    message_types.AdjustPopulationByDiscreteModel,
    message_types.AdjustPopulationByContinuousModel,
    message_types.GetPopulation ]
MessageTypesRegistry.set_receiver_priorities(SpeciesPopSimObject, MESSAGE_TYPES_BY_PRIORITY)
