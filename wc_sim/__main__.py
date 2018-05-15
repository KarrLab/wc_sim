""" Command line programs for simulating whole-cell models

:Author: Jonathan Karr <karr@mssm.edu>
:Date: 2018-05-15
:Copyright: 2018, Karr Lab
:License: MIT
"""

from cement.core.foundation import CementApp
from cement.core.controller import CementBaseController, expose
import wc_sim


class BaseController(CementBaseController):
    """ Base controller for command line application """

    class Meta:
        label = 'base'
        description = "Whole-cell model simulator"
        arguments = [
            (['-v', '--version'], dict(action='version', version=wc_sim.__version__)),
        ]


class App(CementApp):
    """ Command line application """
    class Meta:
        label = 'wc_sim'
        base_controller = 'base'
        handlers = [
            BaseController,
        ]


def main():
    with App() as app:
        app.run()
