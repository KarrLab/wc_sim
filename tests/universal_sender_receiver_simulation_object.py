from Sequential_WC_Simulator.core.simulation_object import SimulationObject
from Sequential_WC_Simulator.core.simulation_engine import MessageTypesRegistry
from Sequential_WC_Simulator.multialgorithm.message_types import *

class UniversalSenderReceiverSimulationObject(SimulationObject):

    ALL_MESSAGE_TYPES = [ADJUST_POPULATION_BY_DISCRETE_MODEL, ADJUST_POPULATION_BY_CONTINUOUS_MODEL, GET_POPULATION, GIVE_POPULATION, EXECUTE_SSA_REACTION, SSA_WAIT ]
    MessageTypesRegistry.set_sent_message_types( 'UniversalSenderReceiverSimulationObject', ALL_MESSAGE_TYPES )
    MessageTypesRegistry.set_receiver_priorities( 'UniversalSenderReceiverSimulationObject', ALL_MESSAGE_TYPES )
