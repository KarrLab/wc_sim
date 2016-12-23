'''
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2016-10-01
:Copyright: 2016, Karr Lab
:License: MIT
'''

from wc_sim.core.simulation_object import SimulationObject
from wc_sim.core.simulation_engine import MessageTypesRegistry
from wc_sim.multialgorithm.message_types import ALL_MESSAGE_TYPES


class UniversalSenderReceiverSimulationObject(SimulationObject):
    pass

MessageTypesRegistry.set_sent_message_types( UniversalSenderReceiverSimulationObject,
    ALL_MESSAGE_TYPES )
MessageTypesRegistry.set_receiver_priorities( UniversalSenderReceiverSimulationObject,
    ALL_MESSAGE_TYPES )
