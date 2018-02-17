# example message types:
from wc_sim.core.simulation_message import SimulationMessageFactory

InitMsg = SimulationMessageFactory.create('InitMsg', 'An InitMsg message')
Eg1 = SimulationMessageFactory.create('Eg1', 'Eg1 simulation message')
MsgWithAttrs = SimulationMessageFactory.create('MsgWithAttrs', 'MsgWithAttrs simulation message', ['attr1', 'attr2'])
UnregisteredMsg = SimulationMessageFactory.create('UnregisteredMsg', 'Unregistered simulation message')
