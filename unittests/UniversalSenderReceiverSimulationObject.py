from Sequential_WC_Simulator.core.SimulationObject import SimulationObject
from Sequential_WC_Simulator.multialgorithm.MessageTypes import MessageTypes
from Sequential_WC_Simulator.core.SimulationEngine import MessageTypesRegistry

class UniversalSenderReceiverSimulationObject(SimulationObject):

    ALL_MESSAGE_TYPES = 'ADJUST_POPULATION_BY_DISCRETE_MODEL ADJUST_POPULATION_BY_CONTINUOUS_MODEL GET_POPULATION GIVE_POPULATION EXECUTE_SSA_REACTION'.split()
    MessageTypesRegistry.set_sent_message_types( 'UniversalSenderReceiverSimulationObject', ALL_MESSAGE_TYPES )
    MessageTypesRegistry.set_receiver_priorities( 'UniversalSenderReceiverSimulationObject', ALL_MESSAGE_TYPES )
