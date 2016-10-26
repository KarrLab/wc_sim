""" PHOLD configuration

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2016-09-21
:Copyright: 2016, Karr Lab
:License: MIT
"""

from wc_utils.debug_logs.config import paths as debug_logs_default_paths
import os

debug_logs = debug_logs_default_paths.deepcopy()
debug_logs.default = os.path.join(os.path.dirname(__file__), 'debug.default.cfg')
