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
                help="Store simulation results; if provided, a timestamped sub-directory will hold results")),
            (['--checkpoint-period'], dict(
                type=float,
                default=config['checkpoint_period'],
                help="Checkpointing period (sec)")),
            (['--dataframe-file'], dict(
                type=str,
                help="File for storing Pandas DataFrame of checkpoints; written in HDF5; requires checkpoints_dir")),
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

        # process dataframe_file
        if args.dataframe_file and not args.checkpoints_dir:
            raise ValueError("dataframe_file cannot be specified unless checkpoints_dir is provided")

        # TODO: convert files specified relative to home directory

        # dataframe_file cannot be in checkpoints dir
        if args.dataframe_file and not args.checkpoints_dir:
            raise ValueError("dataframe_file cannot be specified unless checkpoints_dir is provided")

        # suffix for HDF5 dataframe_file
        if args.dataframe_file and not args.dataframe_file.endswith('.h5'):
            args.dataframe_file = args.dataframe_file + '.h5'

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
        """ Initialize metadata for this simulation run

        Args:
            args (:obj:`object`): parsed command line arguments
        """
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

        # create results directory
        results_dir = None
        if args.checkpoints_dir:
            results_sup_dir = os.path.abspath(os.path.expanduser(args.checkpoints_dir))
            results_dir = os.path.join(results_sup_dir, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)

        # create metadata
        simulation_metadata = SimController.create_metadata(args)

        # create a multi-algorithmic simulator
        simulation_args = dict(
            checkpoint_dir=results_dir,
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

        if args.dataframe_file:
            pred_species_pops = MultialgorithmCheckpoint.convert_checkpoints(results_dir)
            store = pandas.HDFStore(args.dataframe_file)
            store['dataframe'] = pred_species_pops
            store.close()

        print('Simulated {} events'.format(num_events))
        if args.checkpoints_dir:
            print("Saved chcekpoints in '{}'".format(results_dir))

        return (num_events, results_dir)

handlers = [
    SimController,
]
