# example message types:
from wc_sim.core.simulation_message import SimulationMsgUtils

InitMsg = SimulationMsgUtils.create('InitMsg', 'An InitMsg message', ['reaction_index'])
Eg1 = SimulationMsgUtils.create('Eg1', 'Eg1 simulation message')
UnregisteredMsg = SimulationMsgUtils.create('UnregisteredMsg', 'Unregistered simulation message')