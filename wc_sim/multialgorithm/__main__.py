#!/usr/bin/env python3
""" Command line program for WC simulation

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2018-04-18
:Copyright: 2018, Karr Lab
:License: MIT
"""

import argparse
import os
import getpass
import sys
import socket
import datetime
import warnings
import pandas
from cement.core.controller import CementBaseController, expose

from wc_sim.core.sim_metadata import SimulationMetadata, ModelMetadata, AuthorMetadata, RunMetadata
from wc_sim.sim_config import SimulationConfig
from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_lang.io import Reader
from wc_lang.prepare import PrepareModel, CheckModel
from wc_sim.multialgorithm.multialgorithm_checkpointing import MultialgorithmCheckpoint
from wc_sim.multialgorithm.run_results import RunResults

# ignore 'setting concentration' warnings
warnings.filterwarnings('ignore', '.*setting concentration.*', )

# get config
from .config.core import get_config
config = get_config()['wc_sim']['multialgorithm']


class SimController(CementBaseController):
    class Meta:
        label = 'sim'
        description = 'Simulate model'
        stacked_on = 'base'
        stacked_type = 'nested'
        arguments = [
            (['model_file'], dict(
                type=str,
                help="wc_lang model: File containing Excel model, or glob matching tab- or comma-delimited model")),
            (['end_time'], dict(
                type=float,
                help="End time for the simulation (sec)")),
            (['--checkpoints-dir'], dict(
                type=str,
                help="Store simulation results; if provided, a timestamped sub-directory will hold results, "
                "including an HDF5 file that can be accessed through a RunResults object")),
            (['--checkpoint-period'], dict(
                type=float,
                default=config['checkpoint_period'],
                help="Checkpointing period (sec)")),
            (['--fba-time-step'], dict(
                type=float,
                default=config['fba_time_step'],
                help="Timestep for FBA submodel(s) (sec)")),
        ]

    @expose(hide=True)
    def default(self):
        args = self.app.pargs
        self.process_and_validate_args(args)
        self.simulate(args)

    @staticmethod
    def process_and_validate_args(args):
        """ Process and validate command line arguments

        Args:
            args (:obj:`object`): parsed command line arguments

        Raises:
            :obj:`ValueError`: if any of the command line arguments are invalid
        """

        # create results directory
        if args.checkpoints_dir:
            results_sup_dir = os.path.abspath(os.path.expanduser(args.checkpoints_dir))
            args.checkpoints_dir = os.path.join(results_sup_dir, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
            if not os.path.isdir(args.checkpoints_dir):
                os.makedirs(args.checkpoints_dir)

        # validate args
        if args.end_time <= 0:
            raise ValueError("End time ({}) must be positive".format(args.end_time))

        if args.checkpoint_period <= 0 or args.end_time < args.checkpoint_period:
            raise ValueError("Checkpointing period ({}) must be positive and less than or equal to end time".format(
                args.checkpoint_period))

        if args.fba_time_step <= 0.0 or args.end_time < args.fba_time_step:
            raise ValueError("Timestep for FBA submodels ({}) must be positive and less than or equal to end time".format(
                args.fba_time_step))

    @staticmethod
    def create_metadata(args):
        """ Record metadata for this simulation run

        Args:
            args (:obj:`Namespace`): parsed command line arguments for this simulation run

        Returns:
            :obj:`SimulationMetadata`: a metadata record for this simulation run, but missing
                the simulation `run_time`
        """
        # print('type(args)', type(args))
        model = ModelMetadata.create_from_repository()

        # author metadata
        # TODO: collect more comprehensive and specific author information
        ip_address = socket.gethostbyname(socket.gethostname())
        try:
            username = getpass.getuser()
        except:
            username = 'Unknown username'
        author = AuthorMetadata(name='Unknown name', email='Unknown email', username=username,
                                organization='Unknown organization', ip_address=ip_address)

        # simulation config metadata
        time_step = None
        if args.fba_time_step:
            time_step = args.fba_time_step
        simulation_config = SimulationConfig(time_max=args.end_time, time_step=time_step)

        # run metadata
        run = RunMetadata()
        run.record_start()
        run.record_ip_address()

        simulation_metadata = SimulationMetadata(model, simulation_config, run, author)
        return simulation_metadata

    @staticmethod
    def simulate(args):
        """ Run multialgorithmic simulation of a wc_lang model

        Args:
            args (:obj:`object`): parsed command line arguments

        Raises:
            :obj:`MultialgorithmError`: if a model cannot be read from the model file
        """
        # read model
        model_file = os.path.abspath(os.path.expanduser(args.model_file))
        model = Reader().run(model_file, strict=False)
        if model is None:
            raise MultialgorithmError("No model found in model file '{}'".format(model_file))

        # prepare & check the model
        PrepareModel(model).run()
        CheckModel(model).run()

        # create metadata
        simulation_metadata = SimController.create_metadata(args)

        # create a multi-algorithmic simulator
        simulation_args = dict(
            checkpoint_dir=args.checkpoints_dir,
            checkpoint_period=args.checkpoint_period,
            metadata=simulation_metadata
        )

        # run simulation
        multialgorithm_simulation = MultialgorithmSimulation(model, simulation_args)
        simulation_engine, dynamic_model = multialgorithm_simulation.build_simulation()
        simulation_engine.initialize()

        # run simulation
        num_events = simulation_engine.simulate(args.end_time)
        simulation_metadata.run.record_end()

        print('Simulated {} events'.format(num_events))
        if args.checkpoints_dir:
            # use RunResults to summarize results in an HDF5 file in args.checkpoints_dir
            # print('type(simulation_metadata)', type(simulation_metadata))
            RunResults(args.checkpoints_dir, simulation_metadata)
            print("Saved chcekpoints in '{}'".format(args.checkpoints_dir))

        return (num_events, args.checkpoints_dir)

handlers = [
    SimController,
]
