""" Command line programs for simulating whole-cell models

:Author: Jonathan Karr <karr@mssm.edu>
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-05-15
:Copyright: 2018, Karr Lab
:License: MIT
"""

import cement
import warnings

from obj_tables.migrate import data_repo_migration_controllers
import wc_sim
from wc_sim.simulation import Simulation

# ignore 'setting concentration' warnings
warnings.filterwarnings('ignore', '.*setting concentration.*', )

# get config
from .config.core import get_config
config = get_config()['wc_sim']['multialgorithm']


class BaseController(cement.Controller):
    """ Base controller for command line application """

    class Meta:
        label = 'base'
        description = "Whole-cell model simulator"
        help = "Whole-cell model simulator"
        arguments = [
            (['-v', '--version'], dict(action='version', version=wc_sim.__version__)),
        ]

    @cement.ex(hide=True)
    def _default(self):
        self._parser.print_help()


class SimController(cement.Controller):
    class Meta:
        label = 'sim'
        description = 'Simulate a wc-lang model'
        help = 'Simulate a wc-lang model'
        stacked_on = 'base'
        stacked_type = 'nested'
        arguments = [
            (['model_file'], dict(
                type=str,
                help="an Excel file containing a model, or a glob matching tab- or comma-delimited files storing a model")),
            (['max_time'], dict(
                type=float,
                help="maximum time for the simulation (sec)")),
            (['--results-dir'], dict(
                type=str,
                help="store simulation results in results-dir, including an HDF5 file combining all run results")),
            (['--checkpoint-period'], dict(
                type=float,
                default=config['checkpoint_period'],
                help="checkpointing period (sec)")),
            (['--dfba-time-step'], dict(
                type=float,
                default=config['dfba_time_step'],
                help="timestep for dFBA submodel(s) (sec)")),
            (['-p', '--profile'], dict(
                default=False,
                action='store_true',
                help="profile the simulation's performance; saved in results-dir if provided, else on stdout")),
            (['-m', '--memory_profile'], dict(
                type=float,
                default=0,
                help=("event interval in a memory profile of the simulation; saved in results-dir if provided, "
                     "else on stdout"))),
            (['-v', '--verbose'], dict(
                default=False,
                action='store_true',
                help="verbose output")),
        ]

    @cement.ex(hide=True)
    def _default(self):
        args = self.app.pargs
        simulation = Simulation(args.model_file)
        simulation.run(max_time=args.max_time,
                       results_dir=args.results_dir,
                       checkpoint_period=args.checkpoint_period,
                       dfba_time_step=args.dfba_time_step,
                       profile=args.profile,
                       verbose=args.verbose)


class App(cement.App):
    """ Command line application """
    class Meta:
        label = 'wc-sim'
        base_controller = 'base'
        handlers = [BaseController, SimController] + data_repo_migration_controllers


def main():
    with App() as app:
        app.run()
