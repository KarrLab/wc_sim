""" Configuration

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2016-09-19
:Copyright: 2016, Karr Lab
:License: MIT
"""

from pkg_resources import resource_filename
from wc_utils.config.core import ConfigPaths
from wc_utils.debug_logs.config import paths as debug_logs_default_paths
import os


core = ConfigPaths(
    default=resource_filename('wc_sim', 'core/config/core.default.cfg'),
    schema=resource_filename('wc_sim', 'core/config/core.schema.cfg'),
    user=(
        'wc_sim.core.core.cfg',
        os.path.expanduser('~/.wc/wc_sim.core.core.cfg'),
    ),
)

debug_logs = debug_logs_default_paths.deepcopy()
debug_logs.default = resource_filename('wc_sim', 'core/config/debug.default.cfg')
debug_logs.user = (
    'wc_sim.core.debug.cfg',
    os.path.expanduser('~/.wc/wc_sim.core.debug.cfg'),
)
