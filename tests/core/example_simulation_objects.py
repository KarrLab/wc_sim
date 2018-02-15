"""
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-02-15
:Copyright: 2018, Karr Lab
:License: MIT
"""

from wc_sim.core.simulation_object import EventQueue, SimulationObject, SimulationObjectInterface
from tests.core.some_message_types import InitMsg, Eg1, UnregisteredMsg
from wc_utils.util.misc import most_qual_cls_name

ALL_MESSAGE_TYPES = [InitMsg, Eg1]
TEST_SIM_OBJ_STATE = 'Test SimulationObject state'


class ExampleSimulationObject(SimulationObject, SimulationObjectInterface):

    def __init__(self, name):
        super().__init__(name)

    def send_initial_events(self, *args): pass

    def get_state(self):
        return TEST_SIM_OBJ_STATE

    def handler(self, event): pass

    @classmethod
    def register_subclass_handlers(this_class):
        SimulationObject.register_handlers(this_class,
            [(sim_msg_type, this_class.handler) for sim_msg_type in ALL_MESSAGE_TYPES])

    @classmethod
    def register_subclass_sent_messages(this_class):
        SimulationObject.register_sent_messages(this_class, ALL_MESSAGE_TYPES)
