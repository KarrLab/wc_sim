import pkg_resources

with open(pkg_resources.resource_filename('wc_sim', 'VERSION'), 'r') as file:
    __version__ = file.read().strip()
# :obj:`str`: version

# API
from . import log
from . import config

from . import sim_config
from . import aggregate_distributed_props
from . import debug_logs
from . import distributed_properties
from . import dynamic_elements
from . import message_types
from . import model_utilities
from . import multialgorithm_errors
from . import multialgorithm_simulation
from . import species_populations
from . import utils
