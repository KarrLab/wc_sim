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
import os
import getpass
import sys
import socket
import datetime
import warnings
import pandas
from cement.core.controller import CementBaseController, expose

from wc_sim.core.sim_metadata import SimulationMetadata, ModelMetadata, AuthorMetadata, RunMetadata
from wc_sim.core.sim_config import SimulationConfig
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
            (['--checkpoints-dir'], dict(
                type=str,
                help="store simulation results; a timestamped sub-directory of checkpoints-dir will hold results, "
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

    @expose(hide=True)
    def default(self):
        args = self.app.pargs
        self.process_and_validate_args(args)
        self.simulate(args)

    # TODO(Arthur): remove code that's been copied to Simulation
    @staticmethod
    def process_and_validate_args(args):
        """ Process and validate command line arguments

        Args:
            args (:obj:`Namespace`): parsed command line arguments

        Raises:
            :obj:`ValueError`: if any of the command line arguments are invalid
        """

        # process results directory
        if args.checkpoints_dir:
            results_sup_dir = os.path.abspath(os.path.expanduser(args.checkpoints_dir))

            # if results_sup_dir is a file, raise error
            if os.path.isfile(results_sup_dir):
                raise ValueError("checkpoints-dir ({}) is a file, not a dir".format(results_sup_dir))

            # if results_sup_dir does not exist, make it
            if not os.path.exists(results_sup_dir):
                os.makedirs(results_sup_dir)

            # make a time-stamped sub-dir for this run
            time_stamped_sub_dir = os.path.join(results_sup_dir, datetime.datetime.now().strftime(
                '%Y-%m-%d-%H-%M-%S'))
            if os.path.exists(time_stamped_sub_dir):
                raise ValueError("timestamped sub-directory of checkpoints-dir ({}) already exists".format(
                    time_stamped_sub_dir))
            else:
                os.makedirs(time_stamped_sub_dir)
            args.checkpoints_dir = time_stamped_sub_dir

        # validate args
        if args.end_time <= 0:
            raise ValueError("End time ({}) must be positive".format(args.end_time))

        if args.checkpoint_period <= 0 or args.end_time < args.checkpoint_period:
            raise ValueError("Checkpointing period ({}) must be positive and less than or equal to end time".format(
                args.checkpoint_period))

        if args.end_time / args.checkpoint_period % 1 != 0:
            raise ValueError('end_time ({}) must be a multiple of checkpoint_period ({})'.format(
                args.end_time, args.checkpoint_period))

        if hasattr(args, 'fba_time_step') and (args.fba_time_step <= 0.0 or args.end_time < args.fba_time_step):
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
        model = ModelMetadata.create_from_repository()

        # author metadata
        # TODO: collect more comprehensive and specific author information
        try:
            username = getpass.getuser()
        except:     # pragma: no cover
            username = 'Unknown username'
        author = AuthorMetadata(name='Unknown name', email='Unknown email', username=username,
                                organization='Unknown organization')

        # simulation config metadata
        sim_args = {'time_max':args.end_time}
        if  hasattr(args, 'fba_time_step') and args.fba_time_step:
            sim_args['time_step'] = args.fba_time_step
        simulation_config = SimulationConfig(**sim_args)

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
            args (:obj:`Namespace`): parsed command line arguments

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
            RunResults(args.checkpoints_dir, simulation_metadata)
            print("Saved checkpoints and run results in '{}'".format(args.checkpoints_dir))

        return (num_events, args.checkpoints_dir)

handlers = [
    SimController,
]
