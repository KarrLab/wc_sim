""" Setup simulation example configuration

:Author: Arthur Goldberg <Arthur.Goldberg@mssm.edu>
:Date: 2016-10-03
:Copyright: 2016, Karr Lab
:License: MIT
"""

# setup logging
from examples.config import config_constants
from wc_utilities.config.config import ConfigAll
debug_log = ConfigAll.setup_logger( config_constants )
