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
    checkpoints_dir='wc_sim_results'
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
        parser.add_argument('--checkpoints_dir', type=str, default=DEFAULTS['checkpoints_dir'],
            help="Directory for storing simulation results; a timestamped sub directory will hold results")
        parser.add_argument('--dataframe_file', type=str,
            help="File for storing Pandas DataFrame of checkpoints; written in HDF5; requires checkpoints_dir")
        parser.add_argument('--FBA_time_step', type=float,
            default=DEFAULTS['FBA_time_step'], help="Timestep for FBA submodel(s) (sec)")
        parser.add_argument('--num_simulations', type=int, default=1, help="Number of simulation runs")
        args = parser.parse_args(cli_args)

        # validate args
        if args.end_time <= 0:
            parser.error("End time ({}) must be positive".format(args.end_time))
        if args.checkpoint_period <= 0 or args.end_time < args.checkpoint_period:
            parser.error("Checkpointing period ({}) must be positive and less than or equal to end time".format(
                args.checkpoint_period))
        if args.FBA_time_step <= 0.0 or args.end_time < args.FBA_time_step:
            parser.error("Timestep for FBA submodels ({}) must be positive and less than or equal to end time".format(
                args.FBA_time_step))
        if args.num_simulations <= 0:
            parser.error("Number of simulation runs ({}) must be positive".format(args.num_simulations))
        if args.dataframe_file and not args.checkpoints_dir:
            parser.error("Use of --dataframe_file requires specification of --checkpoints_dir")

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

        # run simulation(s)
        if 1 == args.num_simulations:
            multialgorithm_simulation = MultialgorithmSimulation(model, simulation_args)
            simulation_engine, dynamic_model = multialgorithm_simulation.build_simulation()
            print("Simulating '{}'".format(model.name))
            simulation_engine.initialize()
            # run simulation
            num_events = simulation_engine.simulate(args.end_time)
            print("{} events".format(num_events))

            if args.dataframe_file:
                if not args.dataframe_file.endswith('.h5'):
                    args.dataframe_file = args.dataframe_file + '.h5'
                pred_species_pops = MultialgorithmCheckpoint.convert_checkpoints(res_dirname)
                store = pandas.HDFStore(args.dataframe_file)
                store['dataframe'] = pred_species_pops
                print("wrote dataframe to '{}'".format(args.dataframe_file))

        elif 1 < args.num_simulations:
            for run_index in range(args.num_simulations):
                # TODO(Arthur): modify simulation_args for each run
                multialgorithm_simulation = MultialgorithmSimulation(model, simulation_args)
                simulation_engine, dynamic_model = multialgorithm_simulation.build_simulation()
                simulation_run = run_index+1
                print("Simulating '{}', run {}".format(model.name, simulation_run))
                simulation_engine.initialize()
                # run simulation
                num_events = simulation_engine.simulate(args.end_time)
                print("{} events".format(num_events))
        print("Results in '{}'".format(res_dirname))

    @staticmethod
    def main():
        args = RunSimulation.parse_args(sys.argv[1:])
        RunSimulation.run(args)


if __name__ == '__main__':  # pragma: no cover     # run from the command line
    try:
        RunSimulation.main()
    except KeyboardInterrupt:
        pass
