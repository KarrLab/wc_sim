""" Command line programs for simulating whole-cell models

:Author: Jonathan Karr <karr@mssm.edu>
:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2018-05-15
:Copyright: 2018, Karr Lab
:License: MIT
"""

from obj_model.migrate import data_repo_migration_controllers
from obj_model.migrate import CementControllers
from .multialgorithm.__main__ import handlers as multialgorithm_handlers
import cement
import wc_sim


class BaseController(cement.Controller):
    """ Base controller for command line application """

    class Meta:
        label = 'base'
        description = "Whole-cell model simulator"
        arguments = [
            (['-v', '--version'], dict(action='version', version=wc_sim.__version__)),
        ]

    @cement.ex(hide=True)
    def _default(self):
        self._parser.print_help()


class App(cement.App):
    """ Command line application """
    class Meta:
        label = 'wc-sim'
        base_controller = 'base'
        handlers = [BaseController] + multialgorithm_handlers + data_repo_migration_controllers


def main():
    with App() as app:
        app.run()
