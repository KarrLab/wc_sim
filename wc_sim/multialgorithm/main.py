""" Command line program for WC simulation
:Author: Arthur Goldberg, Arthur.Goldberg@mssm.edu
:Date: 2018-04-18
:Copyright: 2018, Karr Lab
:License: MIT
"""

import argparse
import os
import datetime
# ignore 'setting concentration' warnings
import warnings
warnings.filterwarnings('ignore', '.*setting concentration.*', )

from wc_sim.multialgorithm.multialgorithm_simulation import MultialgorithmSimulation
from wc_sim.multialgorithm.multialgorithm_errors import MultialgorithmError
from wc_lang.io import Reader
from wc_lang.prepare import PrepareModel, CheckModel


'''
* Other inputs
    * results directory
* Outputs
    * Simulation results
    * Error report(s)
'''
# TODO(Arthur): move to config data
DEFAULTS = dict(
    prediction_recording_interval=1,
    FBA_time_step=1,
    results_dir='wc_sim_results'
)
class RunSimulation(object):

    @staticmethod
    def parse_args(cli_args=None):
        """ Parse command line arguments

        Args:
            cli_args (:obj:`list`, optional): if provided, use to test command line parsing

        Returns:
            :obj:`argparse.Namespace`: parsed command line arguements
        """
        parser = argparse.ArgumentParser(description="Multialgorithmic whole-cell simulation")
        parser.add_argument('model_file', type=str, help="wc_lang model: "
            "File containing Excel model, or glob matching tab- or comma-delimited model")
        parser.add_argument('end_time', type=float, help="End time for the simulation (sec)")
        parser.add_argument('--FBA_time_step', type=float,
            default=DEFAULTS['FBA_time_step'],
            help="Timestep for FBA submodel(s) (sec)")
        parser.add_argument('--prediction_recording_interval', type=float,
            default=DEFAULTS['prediction_recording_interval'],
            help="Interval between recording of simulation results (sec) [to be implemented]")
        parser.add_argument('--results_dir', type=str, default=DEFAULTS['results_dir'],
            help="Directory for storing simulation results; a timestamped subdir will hold results")
        parser.add_argument('--num_simulations', type=float, default=1,
            help="Number of simulation runs")
        if cli_args is not None:
            args = parser.parse_args(cli_args)
        else:    # pragma: no cover     # reachable only from command line
            args = parser.parse_args()
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
        model = Reader().run(model_file)
        if model is None:
            raise MultialgorithmError("no model found in model file '{}'".format(model_file))
        # prepare & check the model
        PrepareModel(model).run()
        CheckModel(model).run()

        # create results directory
        results_sup_dir = os.path.abspath(os.path.expanduser(args.results_dir))
        res_dirname = os.path.join(results_sup_dir, datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S'))
        if not os.path.isdir(res_dirname):
            os.makedirs(res_dirname)
        # todo: create results object

        # create a multi-algorithmic simulator
        # todo: have MultialgorithmSimulation take kwargs, not Namespace args
        multialgorithm_simulation = MultialgorithmSimulation(model, args)
        multialgorithm_simulation.initialize()
        simulation_engine, dynamic_model = multialgorithm_simulation.build_simulation()

        # run simulation(s)
        for run_index in range(args.num_simulations):
            simulation_run = run_index+1
            print("Simulating {}, run {}".format(model.name, simulation_run))
            simulation_engine.initialize()
            # run simulation
            simulation_engine(args.end_time)
            # todo: save final results
            # reset simulation
            simulation_engine.reset()

if __name__ == '__main__':  # pragma: no cover     # reachable only from command line
    try:
        args = RunSimulation.parse_args()
        print('args', args)
        RunSimulation.run(args)
    except KeyboardInterrupt:
        pass
