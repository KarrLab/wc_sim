from SequentialSimulator.core.SimulationObject import SimulationObject
from SequentialSimulator.multialgorithm.MessageTypes import MessageTypes

class UniversalSenderReceiverSimulationObject(SimulationObject):

    ALL_MESSAGE_TYPES = 'ADJUST_POPULATION_BY_DISCRETE_MODEL ADJUST_POPULATION_BY_CONTINUOUS_MODEL GET_POPULATION GIVE_POPULATION'.split()
    MessageTypes.set_sent_message_types( 'UniversalSenderReceiverSimulationObject', ALL_MESSAGE_TYPES )
    MessageTypes.set_receiver_priorities( 'UniversalSenderReceiverSimulationObject', ALL_MESSAGE_TYPES )
