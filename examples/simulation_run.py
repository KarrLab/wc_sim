""" Simple example simulation run

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-05-22
:Copyright: 2018, Karr Lab
:License: MIT
"""

import os
import matplotlib.pyplot as plt
import tempfile

# import simulation and run results
from wc_sim.simulation import Simulation
from wc_sim.run_results import RunResults

# setup inputs
# use a toy model
model_filename = os.path.join(os.path.dirname(__file__), '../tests/fixtures',
                                   '2_species_1_reaction.xlsx')
results_dir = tempfile.mkdtemp()

# create and run simulation
simulation = Simulation(model_filename)
simulation_rv = simulation.run(max_time=30, results_dir=results_dir, checkpoint_period=10)
results_dir = simulation_rv.results_dir
run_results = RunResults(results_dir)

# view results
# run_results contains 'populations', 'observables', 'functions', 'aggregate_states', 'random_states', and 'metadata'
for component in RunResults.COMPONENTS:
    print('\ncomponent:', component)
    print(run_results.get(component))

# get the mass of compartment 'c' at time 10
aggregate_states_df = run_results.get('aggregate_states')
print("\naggregate_states_df.loc[10, ('c', 'mass')] =", aggregate_states_df.loc[10, ('c', 'mass')])

fig = run_results.get('populations').plot().get_figure()
fig.savefig(os.path.join(results_dir, 'population_dynamics.pdf'))
plt.close(fig)
