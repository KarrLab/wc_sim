#!/usr/bin/env python3
""" Command line program for WC simulation

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2018-04-18
:Copyright: 2018, Karr Lab
:License: MIT
"""

import argparse
from argparse import Namespace
import cement
import os
import getpass
import sys
import socket
import datetime
import warnings
import pandas

from wc_sim.core.sim_metadata import SimulationMetadata, ModelMetadata, AuthorMetadata, RunMetadata
from wc_sim.core.sim_config import SimulationConfig
from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_lang.io import Reader
from wc_lang.prepare import PrepareModel, CheckModel
from wc_sim.multialgorithm.multialgorithm_checkpointing import MultialgorithmCheckpoint
from wc_sim.multialgorithm.simulation import Simulation
from wc_sim.multialgorithm.run_results import RunResults

# ignore 'setting concentration' warnings
warnings.filterwarnings('ignore', '.*setting concentration.*', )

# get config
from .config.core import get_config
config = get_config()['wc_sim']['multialgorithm']


class SimController(cement.Controller):
    class Meta:
        label = 'sim'
        description = 'Simulate a wc-lang model'
        stacked_on = 'base'
        stacked_type = 'nested'
        arguments = [
            (['model_file'], dict(
                type=str,
                help="an Excel file containing a model, or a glob matching tab- or comma-delimited files storing a model")),
            (['end_time'], dict(
                type=float,
                help="end time for the simulation (sec)")),
            (['--results-dir'], dict(
                type=str,
                help="store simulation results; a timestamped sub-directory of end_time-dir will hold results, "
                "including an HDF5 file combining all run results")),
            (['--checkpoint-period'], dict(
                type=float,
                default=config['checkpoint_period'],
                help="checkpointing period (sec)")),
            (['--fba-time-step'], dict(
                type=float,
                default=config['fba_time_step'],
                help="timestep for FBA submodel(s) (sec)")),
        ]

    @cement.ex(hide=True)
    def _default(self):
        args = self.app.pargs
        simulation = Simulation(args.model_file)
        simulation.run(end_time=args.end_time, results_dir=args.results_dir,
            checkpoint_period=args.checkpoint_period)

handlers = [
    SimController,
]
