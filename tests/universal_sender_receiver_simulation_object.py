from wc_sim.core.simulation_object import SimulationObject
from wc_sim.core.simulation_engine import MessageTypesRegistry
from wc_sim.multialgorithm.message_types import ALL_MESSAGE_TYPES


class UniversalSenderReceiverSimulationObject(SimulationObject):

    MessageTypesRegistry.set_sent_message_types( 'UniversalSenderReceiverSimulationObject', ALL_MESSAGE_TYPES )
    MessageTypesRegistry.set_receiver_priorities( 'UniversalSenderReceiverSimulationObject', ALL_MESSAGE_TYPES )
