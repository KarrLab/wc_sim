# example message types:
from wc_sim.core.simulation_message import SimulationMessageFactory

InitMsg = SimulationMessageFactory.create('InitMsg', 'An InitMsg message', ['reaction_index'])
Eg1 = SimulationMessageFactory.create('Eg1', 'Eg1 simulation message')
UnregisteredMsg = SimulationMessageFactory.create('UnregisteredMsg', 'Unregistered simulation message')
