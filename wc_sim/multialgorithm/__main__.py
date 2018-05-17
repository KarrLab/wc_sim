#!/usr/bin/env python3
""" Command line program for WC simulation

:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2018-04-18
:Copyright: 2018, Karr Lab
:License: MIT
"""

import argparse
import os
import sys
import datetime
import warnings
import pandas

from .config.core import get_config
from cement.core.controller import CementBaseController, expose
from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_lang.io import Reader
from wc_lang.prepare import PrepareModel, CheckModel
from wc_sim.multialgorithm.multialgorithm_checkpointing import MultialgorithmCheckpoint

# ignore 'setting concentration' warnings
warnings.filterwarnings('ignore', '.*setting concentration.*', )

# get config
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
            (['--checkpoint-period'], dict(
                type=float,
                default=config['checkpoint_period'],
                help="Checkpointing period (sec)")),
            (['--checkpoints-dir'], dict(
                type=str,
                help="Store simulation results; if provided, a timestamped sub-directory will hold results")),
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
            :obj:`ValueError`: if any fo the command line arguments are invalid
        """

        # process dataframe_file
        if args.dataframe_file:
            if not args.dataframe_file.endswith('.h5'):
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

        if args.dataframe_file and not args.checkpoints_dir:
            raise ValueError("dataframe_file cannot be specified unless checkpoints_dir is provided")

    @staticmethod
    def simulate(args):
        """ Run multialgorithmic simulation of a wc_lang model

        Args:
            args (:obj:`object`): parsed command line arguments

        Raises:
            :obj:`MultialgorithmError`: if a model cannot be read from the model file, or ...
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
        results_sup_dir = os.path.abspath(os.path.expanduser(args.checkpoints_dir))
        res_dirname = os.path.join(results_sup_dir, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        if not os.path.isdir(res_dirname):
            os.makedirs(res_dirname)

        # create a multi-algorithmic simulator
        simulation_args = dict(
            checkpoint_dir=res_dirname,
            checkpoint_period=args.checkpoint_period,
            # TODO(Arthur): provide metadata
            metadata={},
        )

        # run simulation        
        multialgorithm_simulation = MultialgorithmSimulation(model, simulation_args)        
        simulation_engine, dynamic_model = multialgorithm_simulation.build_simulation()
        simulation_engine.initialize()        

        # run simulation
        num_events = simulation_engine.simulate(args.end_time)

        if args.dataframe_file:
            pred_species_pops = MultialgorithmCheckpoint.convert_checkpoints(res_dirname)
            store = pandas.HDFStore(args.dataframe_file)
            store['dataframe'] = pred_species_pops
            store.close()

        print('Saved {} events to {}'.format(num_events, res_dirname))


handlers = [
    SimController,
]
