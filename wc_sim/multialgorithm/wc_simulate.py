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

from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_lang.io import Reader
from wc_lang.prepare import PrepareModel, CheckModel
from wc_sim.multialgorithm.multialgorithm_checkpointing import MultialgorithmCheckpoint

# ignore 'setting concentration' warnings
warnings.filterwarnings('ignore', '.*setting concentration.*', )

# TODO(Arthur): move to config data
DEFAULTS = dict(
    checkpoint_period=10,
    FBA_time_step=1,
)


class RunSimulation(object):

    @staticmethod
    def parse_args(cli_args):
        """ Parse command line arguments

        Args:
            cli_args (:obj:`list`): command line args

        Returns:
            :obj:`argparse.Namespace`: parsed command line arguements
        """
        parser = argparse.ArgumentParser(description="Multialgorithmic whole-cell simulation")
        parser.add_argument('model_file', type=str, help="wc_lang model: "
            "File containing Excel model, or glob matching tab- or comma-delimited model")
        parser.add_argument('end_time', type=float, help="End time for the simulation (sec)")
        parser.add_argument('--checkpoint_period', type=float,
            default=DEFAULTS['checkpoint_period'], help="Checkpointing period (sec)")
        parser.add_argument('--checkpoints_dir', type=str,
            help="Store simulation results; if provided, a timestamped sub-directory will hold results")
        parser.add_argument('--dataframe_file', type=str,
            help="File for storing Pandas DataFrame of checkpoints; written in HDF5; requires checkpoints_dir")
        parser.add_argument('--FBA_time_step', type=float,
            default=DEFAULTS['FBA_time_step'], help="Timestep for FBA submodel(s) (sec)")
        args = parser.parse_args(cli_args)

        if args.dataframe_file:
            if not args.dataframe_file.endswith('.h5'):
                args.dataframe_file = args.dataframe_file + '.h5'

        # validate args
        if args.end_time <= 0:
            parser.error("End time ({}) must be positive".format(args.end_time))
        if args.checkpoint_period <= 0 or args.end_time < args.checkpoint_period:
            parser.error("Checkpointing period ({}) must be positive and less than or equal to end time".format(
                args.checkpoint_period))
        if args.FBA_time_step <= 0.0 or args.end_time < args.FBA_time_step:
            parser.error("Timestep for FBA submodels ({}) must be positive and less than or equal to end time".format(
                args.FBA_time_step))
        if args.dataframe_file and not args.checkpoints_dir:
            parser.error("dataframe_file cannot be specified unless checkpoints_dir is provided")

        return args

    @staticmethod
    def run(args):
        """ Run multialgorithmic simulations of a wc_lang model

        Args:
            args (:obj:`argparse.Namespace`): parsed command line arguments

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
            metadata={}
        )

        # run simulation
        multialgorithm_simulation = MultialgorithmSimulation(model, simulation_args)
        simulation_engine, dynamic_model = multialgorithm_simulation.build_simulation()
        # print("Simulating '{}'".format(model.name))
        simulation_engine.initialize()
        # run simulation
        num_events = simulation_engine.simulate(args.end_time)
        # print("{} events".format(num_events))

        if args.dataframe_file:
            pred_species_pops = MultialgorithmCheckpoint.convert_checkpoints(res_dirname)
            '''
            print('pred_species_pops\n', pred_species_pops)
            print(pred_species_pops.shape)
            print(pred_species_pops.dtypes)
            '''
            store = pandas.HDFStore(args.dataframe_file)
            print('store', store)
            # store['dataframe'] = pred_species_pops
            # print("Wrote dataframe to '{}'".format(args.dataframe_file))
            store.close()

        # print("Results in '{}'".format(res_dirname))
        return (res_dirname, num_events)

    @staticmethod
    def main():
        args = RunSimulation.parse_args(sys.argv[1:])
        RunSimulation.run(args)


if __name__ == '__main__':  # pragma: no cover     # run from the command line
    try:
        RunSimulation.main()
    except KeyboardInterrupt:
        pass
