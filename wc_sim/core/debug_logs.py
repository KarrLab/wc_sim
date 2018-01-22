""" Setup simulator core configuration

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2016-10-05
:Copyright: 2016-2018, Karr Lab
:License: MIT
"""

from .config import paths as config_paths
from wc_utils.config.core import ConfigManager
from wc_utils.debug_logs.core import DebugLogsManager

# setup debug logs
config = ConfigManager(config_paths.debug_logs).get_config()
logs = DebugLogsManager().setup_logs(config)
