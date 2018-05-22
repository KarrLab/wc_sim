""" Simple example simulation run

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2018-05-22
:Copyright: 2018, Karr Lab
:License: MIT
"""

import os
from argparse import Namespace

# import simulate and run results
from wc_sim.multialgorithm.__main__ import SimController

# select tiny model
# tood: more interesting model
model_filename = os.path.join(os.path.dirname(__file__), '../tests/multialgorithm/fixtures',
                                   '2_species_1_reaction.xlsx')

# setup inputs
checkpoints_dir = os.path.expanduser('tmp/checkpoints_dir')
# tood: build into simulator: make dir if doesn't exist, use if empty, error otherwise
if not os.path.isdir(checkpoints_dir):
    os.makedirs(checkpoints_dir)

args = Namespace(
    model_file=model_filename,
    end_time=100,
    checkpoint_period=3,
    checkpoints_dir=checkpoints_dir,
    fba_time_step=5     # although the model doesn't have an FBA submodel
)
SimController.process_and_validate_args(args)

# simulate
num_events, results_dir = SimController.simulate(args)

# view results

