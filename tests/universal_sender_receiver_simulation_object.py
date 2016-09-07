from Sequential_WC_Simulator.core.simulation_object import SimulationObject
from Sequential_WC_Simulator.core.simulation_engine import MessageTypesRegistry
from Sequential_WC_Simulator.multialgorithm.message_types import *

class UniversalSenderReceiverSimulationObject(SimulationObject):

    ALL_MESSAGE_TYPES = [AdjustPopulationByDiscreteModel, AdjustPopulationByContinuousModel, GetPopulation, GivePopulation, ExecuteSSAReaction, SSAWait ]
    MessageTypesRegistry.set_sent_message_types( 'UniversalSenderReceiverSimulationObject', ALL_MESSAGE_TYPES )
    MessageTypesRegistry.set_receiver_priorities( 'UniversalSenderReceiverSimulationObject', ALL_MESSAGE_TYPES )
