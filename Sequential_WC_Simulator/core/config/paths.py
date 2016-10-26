""" Configuration

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2016-09-19
:Copyright: 2016, Karr Lab
:License: MIT
"""

from wc_utils.config.core import ConfigPaths
from wc_utils.debug_logs.config import paths as debug_logs_default_paths
import os


core = ConfigPaths(
    default=os.path.join(os.path.dirname(__file__), 'core.default.cfg'),
    schema=os.path.join(os.path.dirname(__file__), 'core.schema.cfg'),
    user=(
        'Sequential_WC_Simulator.core.core.cfg',
        os.path.expanduser('~/.wc/Sequential_WC_Simulator.core.core.cfg'),
    ),
)

debug_logs = debug_logs_default_paths.deepcopy()
debug_logs.default = os.path.join(os.path.dirname(__file__), 'debug.default.cfg')
debug_logs.user = (
    'Sequential_WC_Simulator.core.debug.cfg',
    os.path.expanduser('~/.wc/Sequential_WC_Simulator.core.debug.cfg'),
)
